"""
HTTP server exposing the convergence engine for multiple workflows.

Endpoints:
  POST /ingest?workflow=<name>&count=500  — Generate and ingest random profiles
  POST /upsert?workflow=<name>&count=100&updates=50 — Ingest new + update existing
  POST /converge?workflow=<name>          — Run convergence (direct or via Temporal)
  POST /converge-all                      — Converge all workflows in dependency order
  GET  /status?workflow=<name>            — Dataset stats (rows, fragments, columns)
  GET  /sample?workflow=<name>&n=5        — Sample rows from the dataset
  GET  /workflows                         — List available workflows
  POST /reset?workflow=<name>             — Delete a workflow's dataset

Set USE_TEMPORAL=1 to route convergence through Temporal instead of in-process.
"""
import os
import random
import shutil
import uuid
from pathlib import Path

import ray
import lance
import pyarrow as pa
from ray import serve
from starlette.requests import Request
from starlette.responses import JSONResponse

from config import parse_config, workflow_dependency_order, WorkflowConfig
from convergence import ConvergenceEngine, ingest_raw_data, upsert_raw_data
import user_data_pb2

USE_TEMPORAL = os.environ.get("USE_TEMPORAL", "").lower() in ("1", "true", "yes")

# Optional Redis lock manager for upsert safety
LOCK_MANAGER = None
REDIS_URL = os.environ.get("REDIS_URL", "")
if REDIS_URL:
    try:
        from locks import RowLockManager
        LOCK_MANAGER = RowLockManager(redis_url=REDIS_URL)
        if LOCK_MANAGER.ping():
            print(f"[INIT] Redis lock manager connected: {REDIS_URL}")
        else:
            LOCK_MANAGER = None
            print("[INIT] Redis not reachable, running without locks")
    except Exception as e:
        print(f"[INIT] Redis unavailable ({e}), running without locks")

WORKFLOWS_DIR = Path(__file__).parent / "workflows"

# Load all workflow configs
CONFIGS: dict[str, WorkflowConfig] = {}
ENGINES: dict[str, ConvergenceEngine] = {}
for yaml_file in sorted(WORKFLOWS_DIR.glob("*.yaml")):
    cfg = parse_config(yaml_file)
    CONFIGS[cfg.name] = cfg
    ENGINES[cfg.name] = ConvergenceEngine(cfg)

# Also support the legacy single-file config as fallback
LEGACY_CONFIG_PATH = Path(__file__).parent / "workflow.yaml"
if LEGACY_CONFIG_PATH.exists() and not CONFIGS:
    cfg = parse_config(LEGACY_CONFIG_PATH)
    CONFIGS[cfg.name] = cfg
    ENGINES[cfg.name] = ConvergenceEngine(cfg)

DEFAULT_WORKFLOW = next(iter(CONFIGS)) if CONFIGS else None

# --- Fake data generators ---

FIRST_NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace",
    "Hank", "Iris", "Jack", "Karen", "Leo", "Mona", "Nate", "Olivia",
    "Paul", "Quinn", "Rosa", "Sam", "Tina", "Uma", "Victor", "Wendy",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
    "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez",
    "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore",
]
SKILLS = [
    "Python", "Java", "SQL", "Machine Learning", "Data Science",
    "React", "Node.js", "AWS", "Docker", "Kubernetes", "Go",
    "Rust", "TypeScript", "GraphQL", "PostgreSQL", "Redis",
    "Kafka", "Spark", "Airflow", "dbt", "Terraform", "CI/CD",
]
REPOS = [
    "awesome-ml", "dotfiles", "react-dashboard", "api-gateway",
    "data-pipeline", "cli-tool", "terraform-modules", "k8s-operator",
    "web-scraper", "chat-bot", "blog", "leetcode-solutions",
]


def _next_id_offset(config: WorkflowConfig) -> int:
    pk = config.primary_key
    try:
        ds = lance.dataset(config.dataset_path)
        ids = ds.to_table(columns=[pk]).column(pk).to_pylist()
        if not ids:
            return 0
        # Try to extract numeric suffix
        nums = []
        for mid in ids:
            parts = mid.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                nums.append(int(parts[1]))
        return max(nums) + 1 if nums else len(ids)
    except Exception:
        return 0


def generate_linkedin_profiles(n: int, id_offset: int = 0) -> list[dict]:
    profiles = []
    for i in range(n):
        mid = f"member_{id_offset + i:06d}"
        name = f"  {random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}  "
        num_skills = random.randint(2, 6)
        skills = ", ".join(random.sample(SKILLS, num_skills))
        if random.random() > 0.5:
            skills = skills.replace(",", " , ").upper()
        profiles.append({"member_id": mid, "name": name, "skills": skills})
    return profiles


def generate_github_profiles(n: int, id_offset: int = 0) -> list[dict]:
    profiles = []
    for i in range(n):
        first = random.choice(FIRST_NAMES)
        last = random.choice(LAST_NAMES)
        username = f"{first.lower()}_{last.lower()}_{id_offset + i}"
        name = f"  {first} {last}  "
        num_repos = random.randint(1, 5)
        repos = ", ".join(random.sample(REPOS, num_repos))
        profiles.append({"github_username": username, "name": name, "repos": repos})
    return profiles


def _generate_for_workflow(config: WorkflowConfig, count: int, id_offset: int) -> list[dict]:
    if config.primary_key == "github_username":
        return generate_github_profiles(count, id_offset)
    return generate_linkedin_profiles(count, id_offset)


# --- Ray Serve deployment ---

@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 1})
class ConvergenceServer:
    def __init__(self):
        pass

    def _get_workflow(self, request: Request) -> tuple[str, WorkflowConfig, ConvergenceEngine] | None:
        name = request.query_params.get("workflow", DEFAULT_WORKFLOW)
        if name not in CONFIGS:
            return None
        return name, CONFIGS[name], ENGINES[name]

    async def __call__(self, request: Request) -> JSONResponse:
        path = request.url.path
        method = request.method

        if path == "/workflows" and method == "GET":
            return await self._list_workflows(request)
        elif path == "/converge-all" and method == "POST":
            return await self._converge_all(request)

        if path == "/ingest" and method == "POST":
            return await self._ingest(request)
        elif path == "/upsert" and method == "POST":
            return await self._upsert(request)
        elif path == "/converge" and method == "POST":
            return await self._converge(request)
        elif path == "/status" and method == "GET":
            return await self._status(request)
        elif path == "/sample" and method == "GET":
            return await self._sample(request)
        elif path == "/reset" and method == "POST":
            return await self._reset(request)
        else:
            return JSONResponse({"error": f"Unknown route: {method} {path}"}, status_code=404)

    async def _list_workflows(self, request: Request):
        order = workflow_dependency_order(CONFIGS)
        workflows = []
        for name in order:
            cfg = CONFIGS[name]
            workflows.append({
                "name": cfg.name,
                "dataset_path": cfg.dataset_path,
                "primary_key": cfg.primary_key,
                "columns": [c.name for c in cfg.columns],
            })
        return JSONResponse({"workflows": workflows, "dependency_order": order})

    async def _ingest(self, request: Request):
        wf = self._get_workflow(request)
        if not wf:
            return JSONResponse({"error": "Unknown workflow"}, status_code=400)
        name, config, engine = wf

        content_type = request.headers.get("content-type", "")

        if "application/x-protobuf" in content_type:
            body = await request.body()
            if config.primary_key == "github_username":
                batch_pb = user_data_pb2.GithubProfileBatch()
                batch_pb.ParseFromString(body)
                profiles = [
                    {"github_username": p.github_username, "name": p.name, "repos": p.repos}
                    for p in batch_pb.profiles
                ]
            else:
                batch_pb = user_data_pb2.ProfileBatch()
                batch_pb.ParseFromString(body)
                profiles = [
                    {"member_id": p.member_id, "name": p.name, "skills": p.skills}
                    for p in batch_pb.profiles
                ]
        else:
            count = int(request.query_params.get("count", 500))
            offset = _next_id_offset(config)
            profiles = _generate_for_workflow(config, count, id_offset=offset)

        batch_size = 500
        total_fragments = 0
        for i in range(0, len(profiles), batch_size):
            batch = profiles[i:i + batch_size]
            total_fragments = ingest_raw_data(config.dataset_path, batch)

        pk = config.primary_key
        first_id = profiles[0][pk] if profiles else ""
        last_id = profiles[-1][pk] if profiles else ""

        return JSONResponse({
            "action": "ingest",
            "workflow": name,
            "count": len(profiles),
            "id_range": [first_id, last_id],
            "fragments": total_fragments,
        })

    async def _upsert(self, request: Request):
        wf = self._get_workflow(request)
        if not wf:
            return JSONResponse({"error": "Unknown workflow"}, status_code=400)
        name, config, engine = wf

        count = int(request.query_params.get("count", 100))
        updates = int(request.query_params.get("updates", 50))

        offset = _next_id_offset(config)
        new_profiles = _generate_for_workflow(config, count, id_offset=offset)

        update_profiles = []
        pk = config.primary_key
        for i in range(updates):
            if pk == "github_username":
                first = random.choice(FIRST_NAMES)
                last = random.choice(LAST_NAMES)
                update_profiles.append({
                    "github_username": f"{first.lower()}_{last.lower()}_{i}",
                    "name": f"  UPDATED {first} {last}  ",
                    "repos": ", ".join(random.sample(REPOS, random.randint(2, 5))),
                })
            else:
                mid = f"member_{i:06d}"
                update_profiles.append({
                    "member_id": mid,
                    "name": f"  UPDATED {random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}  ",
                    "skills": ", ".join(random.sample(SKILLS, random.randint(3, 7))),
                })

        all_records = new_profiles + update_profiles
        random.shuffle(all_records)
        wf_lock_name = config.dataset_path.replace("/", "_").replace(".", "_")
        frag_count = upsert_raw_data(
            config.dataset_path, all_records, primary_key=pk,
            lock_manager=LOCK_MANAGER, workflow_name=wf_lock_name,
        )

        return JSONResponse({
            "action": "upsert",
            "workflow": name,
            "new": count,
            "updates": updates,
            "total_records": len(all_records),
            "fragments": frag_count,
        })

    async def _converge(self, request: Request):
        wf = self._get_workflow(request)
        if not wf:
            return JSONResponse({"error": "Unknown workflow"}, status_code=400)
        name, config, engine = wf

        if USE_TEMPORAL:
            return await self._converge_via_temporal(name, config)

        plan = engine.diff()
        result = engine.converge(plan)

        return JSONResponse({
            "action": "converge",
            "workflow": name,
            "columns_computed": result.columns_computed,
            "columns_failed": result.columns_failed,
            "rows_processed": result.rows_processed,
            "passes": result.passes,
            "fragments_before": result.fragments_before,
            "fragments_after": result.fragments_after,
            "compaction_ran": result.compaction_ran,
            "duration_seconds": round(result.duration_seconds, 3),
        })

    async def _converge_via_temporal(self, name: str, config: WorkflowConfig):
        """Start a Temporal workflow for convergence and wait for result."""
        from temporalio.client import Client
        from temporal_workflow import ConvergeWorkflowInput, ConvergeWorkflow

        temporal_url = os.environ.get("TEMPORAL_URL", "localhost:7233")
        client = await Client.connect(temporal_url)

        config_path = str(WORKFLOWS_DIR / f"{name.replace('-', '_')}_workflow.yaml")
        # Fallback: search for matching config file
        for yaml_file in sorted(WORKFLOWS_DIR.glob("*.yaml")):
            cfg = parse_config(yaml_file)
            if cfg.name == name:
                config_path = str(yaml_file)
                break

        workflow_id = f"converge-{name}-{uuid.uuid4().hex[:8]}"
        result = await client.execute_workflow(
            ConvergeWorkflow.run,
            ConvergeWorkflowInput(
                workflow_name=name,
                config_path=config_path,
                max_passes=config.execution_config.get("max_convergence_passes", 3),
            ),
            id=workflow_id,
            task_queue="convergence-queue",
        )

        return JSONResponse({
            "action": "converge",
            "workflow": name,
            "temporal_workflow_id": workflow_id,
            "columns_computed": result.columns_computed,
            "columns_failed": result.columns_failed,
            "rows_processed": result.rows_processed,
            "passes": result.passes,
            "fragments_before": result.fragments_before,
            "fragments_after": result.fragments_after,
            "compaction_ran": result.compaction_ran,
            "duration_seconds": result.duration_seconds,
        })

    async def _converge_all(self, request: Request):
        order = workflow_dependency_order(CONFIGS)
        results = {}

        if USE_TEMPORAL:
            from temporalio.client import Client
            from temporal_workflow import ConvergeWorkflowInput, ConvergeWorkflow

            temporal_url = os.environ.get("TEMPORAL_URL", "localhost:7233")
            client = await Client.connect(temporal_url)

            for name in order:
                config = CONFIGS[name]
                config_path = str(WORKFLOWS_DIR / f"{name.replace('-', '_')}_workflow.yaml")
                for yaml_file in sorted(WORKFLOWS_DIR.glob("*.yaml")):
                    cfg = parse_config(yaml_file)
                    if cfg.name == name:
                        config_path = str(yaml_file)
                        break

                workflow_id = f"converge-{name}-{uuid.uuid4().hex[:8]}"
                result = await client.execute_workflow(
                    ConvergeWorkflow.run,
                    ConvergeWorkflowInput(
                        workflow_name=name,
                        config_path=config_path,
                        max_passes=config.execution_config.get("max_convergence_passes", 3),
                    ),
                    id=workflow_id,
                    task_queue="convergence-queue",
                )
                results[name] = {
                    "temporal_workflow_id": workflow_id,
                    "columns_computed": result.columns_computed,
                    "columns_failed": result.columns_failed,
                    "rows_processed": result.rows_processed,
                    "passes": result.passes,
                    "fragments_before": result.fragments_before,
                    "fragments_after": result.fragments_after,
                    "compaction_ran": result.compaction_ran,
                    "duration_seconds": result.duration_seconds,
                }
        else:
            for name in order:
                engine = ENGINES[name]
                plan = engine.diff()
                result = engine.converge(plan)
                results[name] = {
                    "columns_computed": result.columns_computed,
                    "columns_failed": result.columns_failed,
                    "rows_processed": result.rows_processed,
                    "passes": result.passes,
                    "fragments_before": result.fragments_before,
                    "fragments_after": result.fragments_after,
                    "compaction_ran": result.compaction_ran,
                    "duration_seconds": round(result.duration_seconds, 3),
                }

        return JSONResponse({
            "action": "converge-all",
            "dependency_order": order,
            "results": results,
        })

    async def _status(self, request: Request):
        wf = self._get_workflow(request)
        if not wf:
            return JSONResponse({"error": "Unknown workflow"}, status_code=400)
        name, config, engine = wf

        try:
            ds = lance.dataset(config.dataset_path)
            return JSONResponse({
                "workflow": name,
                "exists": True,
                "rows": ds.count_rows(),
                "fragments": len(ds.get_fragments()),
                "columns": ds.schema.names,
            })
        except Exception:
            return JSONResponse({"workflow": name, "exists": False, "rows": 0, "fragments": 0, "columns": []})

    async def _sample(self, request: Request):
        wf = self._get_workflow(request)
        if not wf:
            return JSONResponse({"error": "Unknown workflow"}, status_code=400)
        name, config, engine = wf

        n = int(request.query_params.get("n", 5))
        try:
            ds = lance.dataset(config.dataset_path)
            table = ds.to_table().slice(0, n)
            rows = table.to_pylist()
            for row in rows:
                for k, v in row.items():
                    if hasattr(v, "tolist"):
                        row[k] = v.tolist()
            return JSONResponse({"workflow": name, "count": len(rows), "rows": rows})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    async def _reset(self, request: Request):
        wf = self._get_workflow(request)
        if not wf:
            return JSONResponse({"error": "Unknown workflow"}, status_code=400)
        name, config, engine = wf

        path = Path(config.dataset_path)
        if path.exists():
            shutil.rmtree(path)
        return JSONResponse({"action": "reset", "workflow": name, "message": "Dataset deleted"})


app = ConvergenceServer.bind()
