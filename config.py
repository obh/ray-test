"""
Parses the workflow YAML into a DAG of column definitions.
Performs topological sort to determine execution order for derived columns.
"""
from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict


@dataclass
class ScalingConfig:
    parallelism: int = 1
    num_cpus: int = 1
    num_gpus: float = 0
    batch_size: int = 0  # 0 means no batching


@dataclass
class LookupConfig:
    workflow: str        # logical name of external workflow
    dataset_path: str    # Lance path to query
    match_columns: list[str]  # columns to read from external dataset


@dataclass
class ColumnDef:
    name: str
    type: str = "string"
    source: str | None = None  # "raw" for base columns, None for derived
    processor: str | None = None
    derived_from: list[str] = field(default_factory=list)
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    lookup: LookupConfig | None = None

    @property
    def is_derived(self) -> bool:
        return self.processor is not None


@dataclass
class WorkflowConfig:
    name: str
    dataset_path: str
    workflow_version: str
    columns: list[ColumnDef]
    execution_config: dict
    sources: dict
    primary_key: str = "member_id"

    @property
    def raw_columns(self) -> list[ColumnDef]:
        return [c for c in self.columns if not c.is_derived]

    @property
    def derived_columns(self) -> list[ColumnDef]:
        return [c for c in self.columns if c.is_derived]

    def derived_columns_in_order(self) -> list[ColumnDef]:
        """Topological sort of derived columns based on derived_from dependencies."""
        col_map = {c.name: c for c in self.columns}
        derived = {c.name for c in self.derived_columns}

        # Build adjacency list: edge from dependency -> dependent
        in_degree = defaultdict(int)
        dependents = defaultdict(list)
        for c in self.derived_columns:
            in_degree[c.name]  # ensure key exists
            for dep in c.derived_from:
                if dep in derived:
                    dependents[dep].append(c.name)
                    in_degree[c.name] += 1

        # Kahn's algorithm
        queue = [name for name in derived if in_degree[name] == 0]
        result = []
        while queue:
            node = queue.pop(0)
            result.append(col_map[node])
            for dependent in dependents[node]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(result) != len(derived):
            missing = derived - {c.name for c in result}
            raise ValueError(f"Cycle detected in column dependencies: {missing}")

        return result


def workflow_dependency_order(configs: dict[str, WorkflowConfig]) -> list[str]:
    """Topological sort of workflows based on cross-dataset lookup dependencies."""
    in_degree = defaultdict(int)
    dependents = defaultdict(list)
    all_names = set(configs.keys())

    for name, cfg in configs.items():
        in_degree[name]  # ensure key exists
        for col in cfg.columns:
            if col.lookup and col.lookup.workflow in all_names:
                dependents[col.lookup.workflow].append(name)
                in_degree[name] += 1

    queue = [name for name in all_names if in_degree[name] == 0]
    result = []
    while queue:
        node = queue.pop(0)
        result.append(node)
        for dep in dependents[node]:
            in_degree[dep] -= 1
            if in_degree[dep] == 0:
                queue.append(dep)

    if len(result) != len(all_names):
        raise ValueError("Cycle detected in workflow dependencies")

    return result


def parse_config(path: str | Path) -> WorkflowConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    columns = []
    for col_raw in raw["columns"]:
        scaling_raw = col_raw.get("scaling", {})
        scaling = ScalingConfig(
            parallelism=scaling_raw.get("parallelism", 1),
            num_cpus=scaling_raw.get("num_cpus", 1),
            num_gpus=scaling_raw.get("num_gpus", 0),
            batch_size=scaling_raw.get("batch_size", 0),
        )
        derived_from = col_raw.get("derived_from", [])
        if isinstance(derived_from, str):
            derived_from = [derived_from]

        lookup = None
        lookup_raw = col_raw.get("lookup")
        if lookup_raw:
            lookup = LookupConfig(
                workflow=lookup_raw["workflow"],
                dataset_path=lookup_raw["dataset_path"],
                match_columns=lookup_raw["match_columns"],
            )

        columns.append(ColumnDef(
            name=col_raw["name"],
            type=col_raw.get("type", "string"),
            source=col_raw.get("source"),
            processor=col_raw.get("processor"),
            derived_from=derived_from,
            scaling=scaling,
            lookup=lookup,
        ))

    return WorkflowConfig(
        name=raw["name"],
        dataset_path=raw["dataset_path"],
        workflow_version=raw["workflow_version"],
        columns=columns,
        execution_config=raw.get("execution_config", {}),
        sources=raw.get("sources", {}),
        primary_key=raw.get("primary_key", "member_id"),
    )
