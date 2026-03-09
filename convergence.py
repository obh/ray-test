"""
Convergence Engine: the core of the Data State-First approach.

Responsibilities:
1. Diff: Compare desired schema (from YAML) against materialized data in Lance
2. Plan: Identify which fragments are missing which derived columns ("dirty")
3. Execute: Dispatch Ray tasks to compute missing columns, merge results back
4. Retry: Failed column tasks are retried up to max_retries before giving up
5. Re-converge: After a pass, re-diff to catch rows changed during execution
6. Compact: Merge fragments when threshold is exceeded
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

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
    columns_failed: list[str] = field(default_factory=list)
    rows_processed: int = 0
    fragments_before: int = 0
    fragments_after: int = 0
    compaction_ran: bool = False
    passes: int = 0
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
                tasks=[],
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
        Execute convergence with re-diff loop.

        After each pass, re-diffs the dataset. If new dirty rows appeared
        (e.g. data ingested while we were computing), runs another pass.
        Stops when the dataset is fully converged or max_passes is reached.

        Within each pass, columns are grouped into levels by dependency depth.
        Columns at the same level run in parallel. Failed tasks are retried.
        """
        start = time.time()
        max_passes = self.config.execution_config.get("max_convergence_passes", 3)
        max_retries = self.config.execution_config.get("max_retries", 2)

        result = ConvergenceResult()

        # Track null counts before each pass to detect "stuck" columns
        # (processors that intentionally return null for some rows)
        prev_null_counts: dict[str, int] = {}

        for pass_num in range(1, max_passes + 1):
            if plan is None:
                plan = self.diff()

            # Filter out columns that are "stuck" — same null count as last pass,
            # meaning the processor already ran and intentionally returned null
            if prev_null_counts:
                plan.tasks = [
                    t for t in plan.tasks
                    if t.mode == "add_new"
                    or t.col_def.name not in prev_null_counts
                    or t.null_count != prev_null_counts[t.col_def.name]
                ]

            if not plan.columns_to_compute:
                if pass_num == 1:
                    result.fragments_before = plan.fragment_count
                    result.fragments_after = plan.fragment_count
                break

            if pass_num == 1:
                result.fragments_before = plan.fragment_count

            # Snapshot null counts before this pass
            prev_null_counts = {
                t.col_def.name: t.null_count for t in plan.tasks if t.mode == "fill_nulls"
            }

            print(f"  [PASS {pass_num}] {len(plan.columns_to_compute)} columns, "
                  f"{plan.dirty_row_count} rows")

            pass_result = self._execute_pass(plan, max_retries)
            result.columns_computed.extend(pass_result["computed"])
            result.columns_failed.extend(pass_result["failed"])
            result.rows_processed += pass_result["rows"]
            result.passes = pass_num

            # Re-diff: did new data arrive while we were computing?
            plan = self.diff()
            if not plan.columns_to_compute:
                print(f"  [PASS {pass_num}] Converged — no more dirty columns")
                break
            else:
                dirty_names = [t.col_def.name for t in plan.tasks]
                print(f"  [PASS {pass_num}] Re-diffing, dirty: {dirty_names}")

        # Compaction
        if self.dataset_exists():
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

    def _execute_pass(self, plan: ConvergencePlan, max_retries: int) -> dict:
        """
        Execute one convergence pass: process all levels, with retries.
        Returns dict with computed/failed column names and row count.
        """
        computed = []
        failed = []
        total_rows = 0

        task_map = {t.col_def.name: t for t in plan.tasks}
        levels = self.config.derived_columns_by_level()

        for level_idx, level_cols in enumerate(levels):
            level_tasks = [task_map[c.name] for c in level_cols if c.name in task_map]
            if not level_tasks:
                continue

            ds = self._open_dataset()
            existing_columns = set(ds.schema.names)

            ready_tasks = []
            for task in level_tasks:
                deps_available = all(d in existing_columns for d in task.col_def.derived_from)
                if deps_available:
                    ready_tasks.append(task)
                else:
                    print(f"  [SKIP] {task.col_def.name}: dependencies "
                          f"{task.col_def.derived_from} not yet available")

            if not ready_tasks:
                continue

            level_names = [t.col_def.name for t in ready_tasks]
            print(f"  [LEVEL {level_idx}] Computing {level_names} in parallel")

            if len(ready_tasks) == 1:
                task = ready_tasks[0]
                rows, ok = self._execute_task_with_retry(ds, task, max_retries)
                (computed if ok else failed).append(task.col_def.name)
                total_rows += rows
            else:
                level_computed, level_failed, rows = self._execute_level_parallel(
                    ds, ready_tasks, max_retries,
                )
                computed.extend(level_computed)
                failed.extend(level_failed)
                total_rows += rows

        return {"computed": computed, "failed": failed, "rows": total_rows}

    def _execute_task_with_retry(
        self, ds: lance.LanceDataset, task: ColumnTask, max_retries: int,
    ) -> tuple[int, bool]:
        """Execute a single column task with retries. Returns (rows_touched, success)."""
        col_def = task.col_def
        label = "ADD" if task.mode == "add_new" else "FILL"

        for attempt in range(1, max_retries + 2):  # +2: 1 initial + max_retries
            try:
                if task.mode == "add_new":
                    print(f"    [{label}] {col_def.name} <- {col_def.derived_from} "
                          f"(processor={col_def.processor})"
                          f"{f' [retry {attempt-1}]' if attempt > 1 else ''}")
                    rows = self._add_new_column(ds, col_def)
                else:
                    print(f"    [{label}] {col_def.name} <- {col_def.derived_from} "
                          f"({task.null_count} null rows, processor={col_def.processor})"
                          f"{f' [retry {attempt-1}]' if attempt > 1 else ''}")
                    rows = self._fill_null_column(ds, col_def)
                return rows, True
            except Exception as e:
                print(f"    [ERROR] {col_def.name} attempt {attempt}: {e}")
                if attempt <= max_retries:
                    time.sleep(min(2 ** attempt, 10))  # exponential backoff, cap 10s
                    ds = self._open_dataset()  # re-open in case state changed
                else:
                    print(f"    [FAILED] {col_def.name} after {max_retries + 1} attempts")
                    return 0, False
        return 0, False

    def _execute_level_parallel(
        self, ds: lance.LanceDataset, tasks: list[ColumnTask], max_retries: int,
    ) -> tuple[list[str], list[str], int]:
        """
        Compute multiple independent columns in parallel via Ray.
        Uses ray.wait() to merge results as they complete (not in submission order).
        Failed tasks are retried up to max_retries times.

        Returns (computed_names, failed_names, total_rows).
        """
        pk = self.config.primary_key
        computed = []
        failed = []
        total_rows = 0

        # Dispatch all columns in this level concurrently
        ref_to_task = {}
        retry_counts = {}
        for task in tasks:
            col_def = task.col_def
            print(f"    [{'ADD' if task.mode == 'add_new' else 'FILL'}] {col_def.name} "
                  f"<- {col_def.derived_from} (processor={col_def.processor})")
            ref = self._dispatch_column_task(task)
            ref_to_task[ref] = task
            retry_counts[task.col_def.name] = 0

        # Collect results as they complete using ray.wait()
        pending = list(ref_to_task.keys())
        while pending:
            done, pending = ray.wait(pending, num_returns=1)
            ref = done[0]
            task = ref_to_task[ref]
            col_def = task.col_def

            try:
                merge_table, mode, rows_touched = ray.get(ref)
                total_rows += rows_touched

                # Merge into Lance (sequential — Lance merge is not concurrent-safe)
                if merge_table is not None:
                    self._merge_result(task, merge_table, mode)
                    print(f"    [DONE] {col_def.name} ({rows_touched} rows)")

                computed.append(col_def.name)

            except Exception as e:
                retries = retry_counts[col_def.name]
                if retries < max_retries:
                    retry_counts[col_def.name] += 1
                    wait_secs = min(2 ** (retries + 1), 10)
                    print(f"    [RETRY] {col_def.name} attempt {retries + 1}/{max_retries}: "
                          f"{e} (waiting {wait_secs}s)")
                    time.sleep(wait_secs)
                    # Re-dispatch
                    new_ref = self._dispatch_column_task(task)
                    ref_to_task[new_ref] = task
                    pending.append(new_ref)
                else:
                    print(f"    [FAILED] {col_def.name} after {max_retries + 1} attempts: {e}")
                    failed.append(col_def.name)

        return computed, failed, total_rows

    def _dispatch_column_task(self, task: ColumnTask) -> ray.ObjectRef:
        """Submit a column computation as a Ray task. Returns an ObjectRef."""
        col_def = task.col_def
        pk = self.config.primary_key
        return _compute_column_remote.remote(
            self.dataset_path, pk, col_def.name, col_def.processor,
            col_def.derived_from, col_def.lookup, task.mode,
        )

    def _merge_result(self, task: ColumnTask, merge_table: pa.Table, mode: str):
        """Merge a computed column result back into the Lance dataset."""
        pk = self.config.primary_key
        col_def = task.col_def

        if mode == "add_new":
            ds = self._open_dataset()
            ds.merge(merge_table, left_on=pk, right_on=pk)
        else:
            # fill_nulls: delete old rows and append with filled values
            null_ids = merge_table.column(pk).to_pylist()
            ds = self._open_dataset()
            id_list = ", ".join(f"'{mid}'" for mid in null_ids)
            full_rows = ds.to_table(filter=f"{pk} IN ({id_list})")
            col_idx = full_rows.schema.get_field_index(col_def.name)
            full_rows = full_rows.set_column(
                col_idx, col_def.name, merge_table.column(col_def.name),
            )
            ds.delete(f"{pk} IN ({id_list})")
            lance.write_dataset(full_rows, self.dataset_path, mode="append")

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
        all_cols = list(set([pk] + col_def.derived_from + [col_def.name]))
        full_table = ds.to_table(columns=all_cols)

        col_array = full_table.column(col_def.name)
        null_mask = col_array.is_null()
        null_table = full_table.filter(null_mask)
        if len(null_table) == 0:
            return 0

        null_ids = null_table.column(pk).to_pylist()
        print(f"    -> {len(null_ids)} rows need computation")

        input_cols_table = null_table.select([pk] + col_def.derived_from)
        derived_array = self._compute_column(col_def, input_cols_table)

        ds_reopen = self._open_dataset()
        id_list = ", ".join(f"'{mid}'" for mid in null_ids)
        full_rows = ds_reopen.to_table(filter=f"{pk} IN ({id_list})")

        col_idx = full_rows.schema.get_field_index(col_def.name)
        full_rows = full_rows.set_column(col_idx, col_def.name, derived_array)

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
            return self._compute_batched(
                processor_fn, col_def, input_table,
                batch_size, parallelism, num_cpus,
            )
        elif total_rows > 1000 and parallelism > 1:
            chunk_size = max(1, total_rows // parallelism)
            return self._compute_batched(
                processor_fn, col_def, input_table,
                chunk_size, parallelism, num_cpus,
            )
        else:
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

        @ray.remote(num_cpus=num_cpus)
        def process_batch(batch_table, derived_from, processor_name, lookup_config):
            from processors import REGISTRY as reg
            fn = reg[processor_name]
            return fn(batch_table, derived_from, lookup=lookup_config)

        results = []
        in_flight = []
        for batch in batches:
            if len(in_flight) >= parallelism:
                done, in_flight = ray.wait(in_flight, num_returns=1)
                results.extend(ray.get(done))

            ref = process_batch.remote(batch, col_def.derived_from, col_def.processor, col_def.lookup)
            in_flight.append(ref)

        if in_flight:
            results.extend(ray.get(in_flight))

        combined = pa.concat_arrays(results)
        return combined


# --- Ray remote function for level-parallel column computation ---

@ray.remote
def _compute_column_remote(dataset_path, primary_key, col_name, col_processor,
                            derived_from, lookup_config, mode):
    """
    Standalone Ray task that computes a single column.
    Defined at module level so Ray can serialize it once (not per-call).
    """
    import lance as _lance
    import pyarrow as _pa
    from processors import REGISTRY as reg

    ds = _lance.dataset(dataset_path)
    fn = reg[col_processor]

    if mode == "add_new":
        read_cols = [primary_key] + derived_from
        input_table = ds.to_table(columns=read_cols)
        derived_array = fn(input_table, derived_from, lookup=lookup_config)
        return _pa.table({
            primary_key: input_table.column(primary_key),
            col_name: derived_array,
        }), mode, len(input_table)
    else:  # fill_nulls
        all_cols = list(set([primary_key] + derived_from + [col_name]))
        full_table = ds.to_table(columns=all_cols)
        col_array = full_table.column(col_name)
        null_mask = col_array.is_null()
        null_table = full_table.filter(null_mask)
        if len(null_table) == 0:
            return None, mode, 0
        input_cols_table = null_table.select([primary_key] + derived_from)
        derived_array = fn(input_cols_table, derived_from, lookup=lookup_config)
        return _pa.table({
            primary_key: null_table.column(primary_key),
            col_name: derived_array,
        }), mode, len(null_table)


# --- Data ingestion functions ---

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
    lock_manager=None,
    workflow_name: str | None = None,
) -> int:
    """
    Upsert records: new primary keys are appended, existing ones are updated
    using deletion vectors + append.

    If lock_manager is provided, checks for locked rows and skips them
    to avoid overwriting data being converged.

    Returns fragment count after write.
    """
    new_table = pa.Table.from_pylist(records)
    new_ids = set(new_table.column(primary_key).to_pylist())

    try:
        ds = lance.dataset(dataset_path)
    except Exception:
        lance.write_dataset(new_table, dataset_path, mode="overwrite")
        return 1

    existing_table = ds.to_table(columns=[primary_key])
    existing_ids = set(existing_table.column(primary_key).to_pylist())
    overlap_ids = new_ids & existing_ids

    # Check Redis locks — skip rows currently being converged
    if overlap_ids and lock_manager and workflow_name:
        locked = lock_manager.check_locked(workflow_name, list(overlap_ids))
        if locked:
            locked_set = set(locked)
            print(f"  [UPSERT] Skipping {len(locked)} locked rows (being converged)")
            overlap_ids = overlap_ids - locked_set
            # Also remove locked rows from new_table
            all_new_ids = new_table.column(primary_key).to_pylist()
            keep_mask = pa.array([rid not in locked_set for rid in all_new_ids])
            new_table = new_table.filter(keep_mask)

    if overlap_ids:
        id_list = ", ".join(f"'{mid}'" for mid in overlap_ids)
        ds.delete(f"{primary_key} IN ({id_list})")
        print(f"  [UPSERT] Deleted {len(overlap_ids)} existing rows (deletion vectors)")

    existing_schema_names = set(ds.schema.names)
    new_schema_names = set(new_table.schema.names)

    for col_name in existing_schema_names - new_schema_names:
        col_type = ds.schema.field(col_name).type
        null_array = pa.nulls(len(new_table), type=col_type)
        new_table = new_table.append_column(col_name, null_array)

    cols_to_write = [c for c in ds.schema.names if c in new_table.schema.names]
    new_table = new_table.select(cols_to_write)

    lance.write_dataset(new_table, dataset_path, mode="append")
    ds = lance.dataset(dataset_path)
    print(f"  [UPSERT] Appended {len(new_table)} rows. Fragments: {len(ds.get_fragments())}")
    return len(ds.get_fragments())
