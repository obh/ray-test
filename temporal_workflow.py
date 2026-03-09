"""
Temporal workflow for durable convergence orchestration.

Replaces the in-process convergence loop with a Temporal workflow that:
- Survives process restarts (durable execution)
- Has built-in retries with backoff per activity
- Provides visibility into running/completed workflows
- Acquires Redis row locks before computing, releases after merging

Workflow structure:
  ConvergeWorkflow
    ├── activity: diff_dataset       → returns plan (column tasks)
    ├── for each pass:
    │   ├── for each level:
    │   │   ├── activity: acquire_locks     → lock row IDs
    │   │   ├── activity: compute_column    → run processor (parallel per level)
    │   │   ├── activity: merge_column      → merge result into Lance
    │   │   └── activity: release_locks     → unlock row IDs
    │   └── activity: diff_dataset          → re-diff for next pass
    └── activity: compact_dataset    → compaction if needed
"""
from __future__ import annotations

import os
import uuid
import time
from dataclasses import dataclass
from datetime import timedelta

from temporalio import activity, workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    import lance
    import pyarrow as pa
    from config import WorkflowConfig, parse_config
    from processors import REGISTRY
    from locks import RowLockManager


# ---------------------------------------------------------------------------
# Data classes for serialization between workflow and activities
# ---------------------------------------------------------------------------

@dataclass
class ColumnTaskInput:
    col_name: str
    processor: str
    derived_from: list[str]
    mode: str  # "add_new" or "fill_nulls"
    null_count: int
    # Lookup fields (flat, since dataclasses serialize better in Temporal)
    lookup_workflow: str | None = None
    lookup_dataset_path: str | None = None
    lookup_match_columns: list[str] | None = None


@dataclass
class DiffOutput:
    tasks: list[ColumnTaskInput]
    dirty_row_count: int
    fragment_count: int


@dataclass
class ComputeColumnInput:
    dataset_path: str
    primary_key: str
    task: ColumnTaskInput
    owner: str  # lock owner ID


@dataclass
class ComputeColumnOutput:
    col_name: str
    rows_processed: int
    success: bool
    error: str | None = None


@dataclass
class ConvergeWorkflowInput:
    workflow_name: str
    config_path: str  # path to the YAML file
    max_passes: int = 3


@dataclass
class ConvergeWorkflowOutput:
    columns_computed: list[str]
    columns_failed: list[str]
    rows_processed: int
    passes: int
    fragments_before: int
    fragments_after: int
    compaction_ran: bool
    duration_seconds: float


# ---------------------------------------------------------------------------
# Activities
# ---------------------------------------------------------------------------

def _get_lock_manager() -> RowLockManager:
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    return RowLockManager(redis_url=redis_url)


def _get_config(config_path: str) -> WorkflowConfig:
    return parse_config(config_path)


@activity.defn
async def diff_dataset(config_path: str) -> DiffOutput:
    """Diff the dataset against the desired schema. Returns tasks to execute."""
    config = _get_config(config_path)
    try:
        ds = lance.dataset(config.dataset_path)
    except Exception:
        return DiffOutput(tasks=[], dirty_row_count=0, fragment_count=0)

    existing_columns = set(ds.schema.names)
    row_count = ds.count_rows()
    fragment_count = len(ds.get_fragments())

    ordered_derived = config.derived_columns_in_order()
    tasks = []
    for col_def in ordered_derived:
        lookup_wf = lookup_dp = None
        lookup_mc = None
        if col_def.lookup:
            lookup_wf = col_def.lookup.workflow
            lookup_dp = col_def.lookup.dataset_path
            lookup_mc = col_def.lookup.match_columns

        if col_def.name not in existing_columns:
            tasks.append(ColumnTaskInput(
                col_name=col_def.name,
                processor=col_def.processor,
                derived_from=col_def.derived_from,
                mode="add_new",
                null_count=0,
                lookup_workflow=lookup_wf,
                lookup_dataset_path=lookup_dp,
                lookup_match_columns=lookup_mc,
            ))
        else:
            tbl = ds.to_table(columns=[col_def.name])
            null_count = tbl.column(col_def.name).null_count
            if null_count > 0:
                tasks.append(ColumnTaskInput(
                    col_name=col_def.name,
                    processor=col_def.processor,
                    derived_from=col_def.derived_from,
                    mode="fill_nulls",
                    null_count=null_count,
                    lookup_workflow=lookup_wf,
                    lookup_dataset_path=lookup_dp,
                    lookup_match_columns=lookup_mc,
                ))

    return DiffOutput(tasks=tasks, dirty_row_count=row_count, fragment_count=fragment_count)


@activity.defn
async def compute_and_merge_column(inp: ComputeColumnInput) -> ComputeColumnOutput:
    """
    Compute a single column and merge it back into Lance.
    Acquires row locks before reading, releases after merging.

    This is a single activity so that lock lifetime is bounded to one
    Temporal activity execution (with heartbeating for long-running ones).
    """
    config_path = None
    # Reconstruct lookup config if needed
    from config import LookupConfig
    lookup = None
    if inp.task.lookup_dataset_path:
        lookup = LookupConfig(
            workflow=inp.task.lookup_workflow or "",
            dataset_path=inp.task.lookup_dataset_path,
            match_columns=inp.task.lookup_match_columns or [],
        )

    task = inp.task
    pk = inp.primary_key
    ds = lance.dataset(inp.dataset_path)
    fn = REGISTRY[task.processor]

    lock_mgr = _get_lock_manager()
    workflow_name = inp.dataset_path.replace("/", "_").replace(".", "_")
    locked = []  # track for cleanup on error

    try:
        if task.mode == "add_new":
            read_cols = [pk] + task.derived_from
            input_table = ds.to_table(columns=read_cols)
            row_ids = input_table.column(pk).to_pylist()

            # Acquire locks
            locked, skipped = lock_mgr.acquire(workflow_name, row_ids, inp.owner)
            if skipped:
                activity.logger.info(f"Skipping {len(skipped)} locked rows for {task.col_name}")
                # Filter to only locked rows
                locked_set = set(locked)
                mask = pa.array([rid in locked_set for rid in row_ids])
                input_table = input_table.filter(mask)
                if len(input_table) == 0:
                    return ComputeColumnOutput(col_name=task.col_name, rows_processed=0, success=True)

            activity.heartbeat(f"Computing {task.col_name} for {len(input_table)} rows")
            derived_array = fn(input_table, task.derived_from, lookup=lookup)
            merge_table = pa.table({
                pk: input_table.column(pk),
                task.col_name: derived_array,
            })
            ds = lance.dataset(inp.dataset_path)  # re-open
            ds.merge(merge_table, left_on=pk, right_on=pk)

            # Release locks
            lock_mgr.release(workflow_name, locked, inp.owner)
            return ComputeColumnOutput(
                col_name=task.col_name, rows_processed=len(input_table), success=True
            )

        else:  # fill_nulls
            all_cols = list(set([pk] + task.derived_from + [task.col_name]))
            full_table = ds.to_table(columns=all_cols)
            col_array = full_table.column(task.col_name)
            null_mask = col_array.is_null()
            null_table = full_table.filter(null_mask)
            if len(null_table) == 0:
                return ComputeColumnOutput(col_name=task.col_name, rows_processed=0, success=True)

            null_ids = null_table.column(pk).to_pylist()

            # Acquire locks on null rows
            locked, skipped = lock_mgr.acquire(workflow_name, null_ids, inp.owner)
            if skipped:
                activity.logger.info(f"Skipping {len(skipped)} locked rows for {task.col_name}")
                locked_set = set(locked)
                mask = pa.array([rid in locked_set for rid in null_ids])
                null_table = null_table.filter(mask)
                null_ids = [rid for rid in null_ids if rid in locked_set]
                if len(null_table) == 0:
                    return ComputeColumnOutput(col_name=task.col_name, rows_processed=0, success=True)

            activity.heartbeat(f"Filling {task.col_name} for {len(null_table)} null rows")
            input_cols_table = null_table.select([pk] + task.derived_from)
            derived_array = fn(input_cols_table, task.derived_from, lookup=lookup)

            ds = lance.dataset(inp.dataset_path)  # re-open
            id_list = ", ".join(f"'{mid}'" for mid in null_ids)
            full_rows = ds.to_table(filter=f"{pk} IN ({id_list})")
            col_idx = full_rows.schema.get_field_index(task.col_name)
            full_rows = full_rows.set_column(col_idx, task.col_name, derived_array)

            ds.delete(f"{pk} IN ({id_list})")
            lance.write_dataset(full_rows, inp.dataset_path, mode="append")

            # Release locks
            lock_mgr.release(workflow_name, locked, inp.owner)
            return ComputeColumnOutput(
                col_name=task.col_name, rows_processed=len(null_ids), success=True
            )

    except Exception as e:
        # Release any locks we acquired before failing
        try:
            if locked:
                lock_mgr.release(workflow_name, locked, inp.owner)
        except Exception:
            pass
        return ComputeColumnOutput(
            col_name=task.col_name, rows_processed=0, success=False, error=str(e)
        )


@activity.defn
async def compact_dataset(dataset_path: str, threshold: int) -> tuple[bool, int]:
    """Run compaction if fragments exceed threshold. Returns (ran, fragments_after)."""
    try:
        ds = lance.dataset(dataset_path)
        frag_count = len(ds.get_fragments())
        if frag_count > threshold:
            ds.compact_files()
            ds = lance.dataset(dataset_path)
            return True, len(ds.get_fragments())
        return False, frag_count
    except Exception:
        return False, 0


# ---------------------------------------------------------------------------
# Temporal Workflow
# ---------------------------------------------------------------------------

@workflow.defn
class ConvergeWorkflow:
    """
    Durable convergence workflow.

    Orchestrates: diff → [per-level: lock → compute → merge → unlock] → compact
    with multi-pass re-diff loop and stuck-column detection.
    """

    @workflow.run
    async def run(self, inp: ConvergeWorkflowInput) -> ConvergeWorkflowOutput:
        start = time.time()
        owner = f"temporal-{workflow.info().workflow_id}"

        result = ConvergeWorkflowOutput(
            columns_computed=[], columns_failed=[], rows_processed=0,
            passes=0, fragments_before=0, fragments_after=0,
            compaction_ran=False, duration_seconds=0.0,
        )

        # Load config to get level info
        config = _get_config(inp.config_path)
        prev_null_counts: dict[str, int] = {}

        for pass_num in range(1, inp.max_passes + 1):
            # Diff
            diff_out: DiffOutput = await workflow.execute_activity(
                diff_dataset,
                inp.config_path,
                start_to_close_timeout=timedelta(minutes=5),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )

            if pass_num == 1:
                result.fragments_before = diff_out.fragment_count

            # Filter stuck columns
            if prev_null_counts:
                diff_out.tasks = [
                    t for t in diff_out.tasks
                    if t.mode == "add_new"
                    or t.col_name not in prev_null_counts
                    or t.null_count != prev_null_counts[t.col_name]
                ]

            if not diff_out.tasks:
                if pass_num == 1:
                    result.fragments_after = diff_out.fragment_count
                break

            # Snapshot null counts
            prev_null_counts = {
                t.col_name: t.null_count for t in diff_out.tasks if t.mode == "fill_nulls"
            }

            workflow.logger.info(
                f"Pass {pass_num}: {len(diff_out.tasks)} columns, "
                f"{diff_out.dirty_row_count} rows"
            )

            # Group tasks by dependency level
            task_by_name = {t.col_name: t for t in diff_out.tasks}
            levels = config.derived_columns_by_level()

            for level_idx, level_cols in enumerate(levels):
                level_tasks = [task_by_name[c.name] for c in level_cols if c.name in task_by_name]
                if not level_tasks:
                    continue

                # Execute all columns in this level concurrently
                handles = []
                for task in level_tasks:
                    compute_input = ComputeColumnInput(
                        dataset_path=config.dataset_path,
                        primary_key=config.primary_key,
                        task=task,
                        owner=owner,
                    )
                    handle = workflow.execute_activity(
                        compute_and_merge_column,
                        compute_input,
                        start_to_close_timeout=timedelta(minutes=30),
                        heartbeat_timeout=timedelta(minutes=5),
                        retry_policy=RetryPolicy(
                            maximum_attempts=3,
                            initial_interval=timedelta(seconds=2),
                            backoff_coefficient=2.0,
                            maximum_interval=timedelta(seconds=30),
                        ),
                    )
                    handles.append(handle)

                # Wait for all columns in this level to complete (barrier)
                import asyncio
                outputs: list[ComputeColumnOutput] = await asyncio.gather(*handles)
                for out in outputs:
                    if out.success:
                        result.columns_computed.append(out.col_name)
                    else:
                        result.columns_failed.append(out.col_name)
                        workflow.logger.warning(f"Column {out.col_name} failed: {out.error}")
                    result.rows_processed += out.rows_processed

            result.passes = pass_num

        # Compaction
        threshold = config.execution_config.get("compaction_fragment_threshold", 20)
        compacted, frags_after = await workflow.execute_activity(
            compact_dataset,
            args=[config.dataset_path, threshold],
            start_to_close_timeout=timedelta(minutes=10),
            retry_policy=RetryPolicy(maximum_attempts=2),
        )
        result.compaction_ran = compacted
        result.fragments_after = frags_after
        result.duration_seconds = round(time.time() - start, 3)

        return result


# ---------------------------------------------------------------------------
# Worker entrypoint
# ---------------------------------------------------------------------------

async def run_worker(task_queue: str = "convergence-queue"):
    """Start a Temporal worker that hosts convergence activities + workflow."""
    from temporalio.client import Client
    from temporalio.worker import Worker

    temporal_url = os.environ.get("TEMPORAL_URL", "localhost:7233")
    client = await Client.connect(temporal_url)

    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[ConvergeWorkflow],
        activities=[diff_dataset, compute_and_merge_column, compact_dataset],
    )
    print(f"Starting Temporal worker on queue '{task_queue}' (temporal={temporal_url})")
    await worker.run()


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_worker())
