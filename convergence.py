"""
Convergence Engine: the core of the Data State-First approach.

Responsibilities:
1. Diff: Compare desired schema (from YAML) against materialized data in Lance
2. Plan: Identify which fragments are missing which derived columns ("dirty")
3. Execute: Dispatch Ray tasks to compute missing columns, merge results back
4. Compact: Merge fragments when threshold is exceeded
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import lance
import pyarrow as pa
import ray

from config import WorkflowConfig, ColumnDef, LookupConfig
from processors import REGISTRY


@dataclass
class ColumnTask:
    """A single column that needs computation."""
    col_def: ColumnDef
    mode: str  # "add_new" (column missing) or "fill_nulls" (column exists, has NULLs)
    null_count: int = 0


@dataclass
class ConvergencePlan:
    """What needs to be computed in this convergence pass."""
    tasks: list[ColumnTask]
    dirty_row_count: int
    fragment_count: int

    @property
    def columns_to_compute(self) -> list[ColumnDef]:
        return [t.col_def for t in self.tasks]


@dataclass
class ConvergenceResult:
    columns_computed: list[str] = field(default_factory=list)
    rows_processed: int = 0
    fragments_before: int = 0
    fragments_after: int = 0
    compaction_ran: bool = False
    duration_seconds: float = 0.0


class ConvergenceEngine:
    def __init__(self, config: WorkflowConfig):
        self.config = config
        self.dataset_path = config.dataset_path

    def dataset_exists(self) -> bool:
        try:
            lance.dataset(self.dataset_path)
            return True
        except Exception:
            return False

    def _open_dataset(self) -> lance.LanceDataset:
        return lance.dataset(self.dataset_path)

    def diff(self) -> ConvergencePlan:
        """
        Compare desired schema against materialized data.
        Returns a plan of what columns need to be computed.
        """
        if not self.dataset_exists():
            return ConvergencePlan(
                columns_to_compute=[],
                dirty_row_count=0,
                fragment_count=0,
            )

        ds = self._open_dataset()
        existing_columns = set(ds.schema.names)
        row_count = ds.count_rows()
        fragment_count = len(ds.get_fragments())

        ordered_derived = self.config.derived_columns_in_order()
        tasks = []
        for col_def in ordered_derived:
            if col_def.name not in existing_columns:
                tasks.append(ColumnTask(col_def=col_def, mode="add_new"))
            else:
                # Column exists — check for NULL rows (stale/incomplete)
                tbl = ds.to_table(columns=[col_def.name])
                null_count = tbl.column(col_def.name).null_count
                if null_count > 0:
                    tasks.append(ColumnTask(
                        col_def=col_def, mode="fill_nulls", null_count=null_count,
                    ))

        return ConvergencePlan(
            tasks=tasks,
            dirty_row_count=row_count,
            fragment_count=fragment_count,
        )

    def converge(self, plan: ConvergencePlan | None = None) -> ConvergenceResult:
        """
        Execute a convergence pass: compute all missing/stale derived columns
        and merge them back into the Lance dataset.
        """
        start = time.time()

        if plan is None:
            plan = self.diff()

        if not plan.columns_to_compute:
            return ConvergenceResult(
                duration_seconds=time.time() - start,
                fragments_before=plan.fragment_count,
                fragments_after=plan.fragment_count,
            )

        ds = self._open_dataset()
        result = ConvergenceResult(
            fragments_before=plan.fragment_count,
        )

        # Process columns in topological order — each column may depend on
        # a column computed in a previous iteration of this loop
        for task in plan.tasks:
            col_def = task.col_def
            # Re-open dataset to pick up columns merged in previous iterations
            ds = self._open_dataset()
            existing_columns = set(ds.schema.names)

            # Check that all dependencies are available
            deps_available = all(d in existing_columns for d in col_def.derived_from)
            if not deps_available:
                print(f"  [SKIP] {col_def.name}: dependencies {col_def.derived_from} not yet available")
                continue

            if task.mode == "add_new":
                print(f"  [ADD] {col_def.name} <- {col_def.derived_from} "
                      f"(processor={col_def.processor})")
                rows_touched = self._add_new_column(ds, col_def)
            else:
                print(f"  [FILL] {col_def.name} <- {col_def.derived_from} "
                      f"({task.null_count} null rows, processor={col_def.processor})")
                rows_touched = self._fill_null_column(ds, col_def)

            result.columns_computed.append(col_def.name)
            result.rows_processed += rows_touched

        # Check if compaction is needed
        ds = self._open_dataset()
        result.fragments_after = len(ds.get_fragments())
        threshold = self.config.execution_config.get("compaction_fragment_threshold", 20)

        if result.fragments_after > threshold:
            print(f"  [COMPACT] {result.fragments_after} fragments > threshold {threshold}")
            ds.compact_files()
            ds = self._open_dataset()
            result.fragments_after = len(ds.get_fragments())
            result.compaction_ran = True
            print(f"  [COMPACT] After compaction: {result.fragments_after} fragments")

        result.duration_seconds = time.time() - start
        return result

    def _add_new_column(self, ds: lance.LanceDataset, col_def: ColumnDef) -> int:
        """Add a column that doesn't exist yet. Uses Lance merge."""
        pk = self.config.primary_key
        read_cols = [pk] + col_def.derived_from
        input_table = ds.to_table(columns=read_cols)
        derived_array = self._compute_column(col_def, input_table)
        merge_table = pa.table({
            pk: input_table.column(pk),
            col_def.name: derived_array,
        })
        ds.merge(merge_table, left_on=pk, right_on=pk)
        return len(input_table)

    def _fill_null_column(self, ds: lance.LanceDataset, col_def: ColumnDef) -> int:
        """
        Fill NULL values in an existing column.
        Strategy: read rows where column is null, compute values,
        then delete those rows and re-append with filled values.
        """
        pk = self.config.primary_key
        # Read all data — filter to rows where this column is null
        all_cols = list(set([pk] + col_def.derived_from + [col_def.name]))
        full_table = ds.to_table(columns=all_cols)

        # Find null mask
        col_array = full_table.column(col_def.name)
        null_mask = col_array.is_null()

        # Filter to only null rows
        null_table = full_table.filter(null_mask)
        if len(null_table) == 0:
            return 0

        null_ids = null_table.column(pk).to_pylist()
        print(f"    -> {len(null_ids)} rows need computation")

        # Compute the derived values for null rows
        input_cols_table = null_table.select([pk] + col_def.derived_from)
        derived_array = self._compute_column(col_def, input_cols_table)

        # Build complete rows for the null entries (all existing columns)
        ds_reopen = self._open_dataset()

        # Read the full row data for these primary key values
        id_list = ", ".join(f"'{mid}'" for mid in null_ids)
        full_rows = ds_reopen.to_table(filter=f"{pk} IN ({id_list})")

        # Replace the null column with computed values
        col_idx = full_rows.schema.get_field_index(col_def.name)
        full_rows = full_rows.set_column(col_idx, col_def.name, derived_array)

        # Delete old rows and append updated ones
        ds_reopen.delete(f"{pk} IN ({id_list})")
        lance.write_dataset(full_rows, self.dataset_path, mode="append")

        return len(null_ids)

    def _compute_column(self, col_def: ColumnDef, input_table: pa.Table) -> pa.Array:
        """
        Compute a single derived column. Uses Ray for parallelism.
        Handles batching if configured.
        """
        processor_fn = REGISTRY[col_def.processor]
        batch_size = col_def.scaling.batch_size
        parallelism = col_def.scaling.parallelism
        num_cpus = col_def.scaling.num_cpus

        total_rows = len(input_table)

        if batch_size > 0 and total_rows > batch_size:
            # Split into batches and process in parallel via Ray
            return self._compute_batched(
                processor_fn, col_def, input_table,
                batch_size, parallelism, num_cpus,
            )
        elif total_rows > 1000 and parallelism > 1:
            # Split into chunks for parallelism even without explicit batch_size
            chunk_size = max(1, total_rows // parallelism)
            return self._compute_batched(
                processor_fn, col_def, input_table,
                chunk_size, parallelism, num_cpus,
            )
        else:
            # Small enough to process directly
            return processor_fn(input_table, col_def.derived_from, lookup=col_def.lookup)

    def _compute_batched(
        self, processor_fn, col_def: ColumnDef,
        input_table: pa.Table, batch_size: int,
        parallelism: int, num_cpus: int,
    ) -> pa.Array:
        """Split table into batches, dispatch to Ray, reassemble."""
        total_rows = len(input_table)
        batches = []
        for start in range(0, total_rows, batch_size):
            end = min(start + batch_size, total_rows)
            batches.append(input_table.slice(start, end - start))

        # Create Ray remote function with resource constraints
        @ray.remote(num_cpus=num_cpus)
        def process_batch(batch_table, derived_from, processor_name, lookup_config):
            from processors import REGISTRY as reg
            fn = reg[processor_name]
            return fn(batch_table, derived_from, lookup=lookup_config)

        # Submit batches with parallelism limit
        results = []
        in_flight = []
        for batch in batches:
            if len(in_flight) >= parallelism:
                # Wait for at least one to finish
                done, in_flight = ray.wait(in_flight, num_returns=1)
                results.extend(ray.get(done))

            ref = process_batch.remote(batch, col_def.derived_from, col_def.processor, col_def.lookup)
            in_flight.append(ref)

        # Collect remaining
        if in_flight:
            results.extend(ray.get(in_flight))

        # Concatenate all result arrays
        combined = pa.concat_arrays(results)
        return combined


def ingest_raw_data(
    dataset_path: str,
    records: list[dict],
    mode: str = "append",
) -> int:
    """
    Write raw records to the Lance dataset.
    Handles the buffering concern: callers should batch before calling this.
    Returns fragment count after write.
    """
    table = pa.Table.from_pylist(records)

    try:
        lance.dataset(dataset_path)
        exists = True
    except Exception:
        exists = False

    if not exists or mode == "overwrite":
        lance.write_dataset(table, dataset_path, mode="overwrite")
    else:
        lance.write_dataset(table, dataset_path, mode="append")

    ds = lance.dataset(dataset_path)
    return len(ds.get_fragments())


def upsert_raw_data(
    dataset_path: str,
    records: list[dict],
    primary_key: str = "member_id",
) -> int:
    """
    Upsert records: new primary keys are appended, existing ones are updated
    using deletion vectors + append.
    Returns fragment count after write.
    """
    new_table = pa.Table.from_pylist(records)
    new_ids = set(new_table.column(primary_key).to_pylist())

    try:
        ds = lance.dataset(dataset_path)
    except Exception:
        # Dataset doesn't exist yet, just write
        lance.write_dataset(new_table, dataset_path, mode="overwrite")
        return 1

    # Find existing rows that match incoming primary keys
    existing_table = ds.to_table(columns=[primary_key])
    existing_ids = set(existing_table.column(primary_key).to_pylist())
    overlap_ids = new_ids & existing_ids

    if overlap_ids:
        # Delete old versions of overlapping rows using a filter
        id_list = ", ".join(f"'{mid}'" for mid in overlap_ids)
        ds.delete(f"{primary_key} IN ({id_list})")
        print(f"  [UPSERT] Deleted {len(overlap_ids)} existing rows (deletion vectors)")

    # Append all new records (including updates)
    # We need to match the schema of the existing dataset
    existing_schema_names = set(ds.schema.names)
    new_schema_names = set(new_table.schema.names)

    # Add null columns for any derived columns that exist in the dataset but not in new data
    for col_name in existing_schema_names - new_schema_names:
        col_type = ds.schema.field(col_name).type
        null_array = pa.nulls(len(new_table), type=col_type)
        new_table = new_table.append_column(col_name, null_array)

    # Only keep columns that exist in the dataset schema
    cols_to_write = [c for c in ds.schema.names if c in new_table.schema.names]
    new_table = new_table.select(cols_to_write)

    lance.write_dataset(new_table, dataset_path, mode="append")
    ds = lance.dataset(dataset_path)
    print(f"  [UPSERT] Appended {len(records)} rows. Fragments: {len(ds.get_fragments())}")
    return len(ds.get_fragments())
