"""
Test client for the multi-workflow convergence server.

Usage:
  python test_load.py                              # Full lifecycle: both workflows
  python test_load.py ingest 1000                   # Ingest 1000 linkedin profiles (protobuf)
  python test_load.py ingest 500 github              # Ingest 500 github users (protobuf)
  python test_load.py converge                      # Converge default workflow
  python test_load.py converge github                # Converge github workflow
  python test_load.py converge-all                  # Converge all in dependency order
  python test_load.py upsert 200 50                 # 200 new + 50 updates (linkedin)
  python test_load.py status                        # Status of default workflow
  python test_load.py status github                 # Status of github workflow
  python test_load.py sample 10                     # 10 sample rows (linkedin)
  python test_load.py sample 10 github              # 10 sample rows (github)
  python test_load.py workflows                     # List available workflows
  python test_load.py reset                         # Delete default workflow dataset
"""
import sys
import time
import random
import httpx
import json

import user_data_pb2

BASE_URL = "http://localhost:8000"

WORKFLOW_LINKEDIN = "linkedin-person-enrichment"
WORKFLOW_GITHUB = "github-user-enrichment"

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


def pp(data: dict):
    print(json.dumps(data, indent=2))


def wait_for_server(timeout_s=30):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = httpx.get(f"{BASE_URL}/workflows", timeout=5)
            if r.status_code < 500:
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError(f"Server at {BASE_URL} not ready after {timeout_s}s")


def _build_linkedin_batch_proto(count: int, id_offset: int = 0) -> bytes:
    batch = user_data_pb2.ProfileBatch()
    for i in range(count):
        p = batch.profiles.add()
        p.member_id = f"member_{id_offset + i:06d}"
        p.name = f"  {random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}  "
        num_skills = random.randint(2, 6)
        skills = ", ".join(random.sample(SKILLS, num_skills))
        if random.random() > 0.5:
            skills = skills.replace(",", " , ").upper()
        p.skills = skills
    return batch.SerializeToString()


def _build_github_batch_proto(count: int, id_offset: int = 0) -> bytes:
    batch = user_data_pb2.GithubProfileBatch()
    for i in range(count):
        p = batch.profiles.add()
        first = random.choice(FIRST_NAMES)
        last = random.choice(LAST_NAMES)
        p.github_username = f"{first.lower()}_{last.lower()}_{id_offset + i}"
        p.name = f"  {first} {last}  "
        num_repos = random.randint(1, 5)
        p.repos = ", ".join(random.sample(REPOS, num_repos))
    return batch.SerializeToString()


def _get_next_offset(workflow: str = WORKFLOW_LINKEDIN) -> int:
    try:
        r = httpx.get(f"{BASE_URL}/status?workflow={workflow}", timeout=5)
        return r.json().get("rows", 0)
    except Exception:
        return 0


def cmd_ingest(count=500, workflow=WORKFLOW_LINKEDIN):
    print(f"\n--- Ingesting {count} profiles into {workflow} (protobuf) ---")
    offset = _get_next_offset(workflow)

    if workflow == WORKFLOW_GITHUB:
        proto_bytes = _build_github_batch_proto(count, id_offset=offset)
    else:
        proto_bytes = _build_linkedin_batch_proto(count, id_offset=offset)

    print(f"    Protobuf payload: {len(proto_bytes)} bytes")
    r = httpx.post(
        f"{BASE_URL}/ingest?workflow={workflow}",
        content=proto_bytes,
        headers={"Content-Type": "application/x-protobuf"},
        timeout=60,
    )
    pp(r.json())


def cmd_converge(workflow=WORKFLOW_LINKEDIN):
    print(f"\n--- Running convergence for {workflow} ---")
    r = httpx.post(f"{BASE_URL}/converge?workflow={workflow}", timeout=120)
    pp(r.json())


def cmd_converge_all():
    print("\n--- Running converge-all (dependency order) ---")
    r = httpx.post(f"{BASE_URL}/converge-all", timeout=300)
    pp(r.json())


def cmd_upsert(count=100, updates=50, workflow=WORKFLOW_LINKEDIN):
    print(f"\n--- Upserting {count} new + {updates} updates in {workflow} ---")
    r = httpx.post(
        f"{BASE_URL}/upsert?workflow={workflow}&count={count}&updates={updates}",
        timeout=60,
    )
    pp(r.json())


def cmd_status(workflow=WORKFLOW_LINKEDIN):
    print(f"\n--- Dataset status ({workflow}) ---")
    r = httpx.get(f"{BASE_URL}/status?workflow={workflow}", timeout=10)
    pp(r.json())


def cmd_sample(n=5, workflow=WORKFLOW_LINKEDIN):
    print(f"\n--- Sample ({n} rows from {workflow}) ---")
    r = httpx.get(f"{BASE_URL}/sample?workflow={workflow}&n={n}", timeout=10)
    data = r.json()
    if "rows" in data:
        for row in data["rows"]:
            emb = row.get("profile_embedding")
            emb_str = f"[{len(emb)}d]" if emb else "null"
            match = row.get("linkedin_match", "n/a")
            pk = row.get("member_id") or row.get("github_username", "?")
            print(f"  {pk}: {row.get('cleaned_name', row.get('name', ''))!r} "
                  f"emb={emb_str} match={match}")
    else:
        pp(data)


def cmd_workflows():
    print("\n--- Available workflows ---")
    r = httpx.get(f"{BASE_URL}/workflows", timeout=10)
    pp(r.json())


def cmd_reset(workflow=WORKFLOW_LINKEDIN):
    print(f"\n--- Resetting {workflow} ---")
    r = httpx.post(f"{BASE_URL}/reset?workflow={workflow}", timeout=10)
    pp(r.json())


def full_lifecycle():
    """Run full multi-workflow demo via API calls."""
    print("=== Full Multi-Workflow Lifecycle Demo ===")

    cmd_workflows()

    # LinkedIn: ingest + converge
    cmd_ingest(500, WORKFLOW_LINKEDIN)
    cmd_converge(WORKFLOW_LINKEDIN)
    cmd_sample(3, WORKFLOW_LINKEDIN)

    # GitHub: ingest + converge (cross-dataset lookup)
    cmd_ingest(300, WORKFLOW_GITHUB)
    cmd_converge(WORKFLOW_GITHUB)
    cmd_sample(5, WORKFLOW_GITHUB)

    # Converge-all (idempotency)
    print("\n--- Idempotency check (converge-all) ---")
    cmd_converge_all()

    # Status for both
    cmd_status(WORKFLOW_LINKEDIN)
    cmd_status(WORKFLOW_GITHUB)


def _resolve_workflow(arg: str | None) -> str:
    if arg and "github" in arg.lower():
        return WORKFLOW_GITHUB
    return WORKFLOW_LINKEDIN


def main():
    args = sys.argv[1:]

    print(f"Waiting for server at {BASE_URL}...")
    wait_for_server()
    print("Server ready.\n")

    if not args:
        full_lifecycle()
    elif args[0] == "ingest":
        count = int(args[1]) if len(args) > 1 else 500
        workflow = _resolve_workflow(args[2] if len(args) > 2 else None)
        cmd_ingest(count, workflow)
    elif args[0] == "converge":
        workflow = _resolve_workflow(args[1] if len(args) > 1 else None)
        cmd_converge(workflow)
    elif args[0] == "converge-all":
        cmd_converge_all()
    elif args[0] == "upsert":
        count = int(args[1]) if len(args) > 1 else 100
        updates = int(args[2]) if len(args) > 2 else 50
        workflow = _resolve_workflow(args[3] if len(args) > 3 else None)
        cmd_upsert(count, updates, workflow)
    elif args[0] == "status":
        workflow = _resolve_workflow(args[1] if len(args) > 1 else None)
        cmd_status(workflow)
    elif args[0] == "sample":
        n = int(args[1]) if len(args) > 1 else 5
        workflow = _resolve_workflow(args[2] if len(args) > 2 else None)
        cmd_sample(n, workflow)
    elif args[0] == "workflows":
        cmd_workflows()
    elif args[0] == "reset":
        workflow = _resolve_workflow(args[1] if len(args) > 1 else None)
        cmd_reset(workflow)
    else:
        print(f"Unknown command: {args[0]}")
        print(__doc__)


if __name__ == "__main__":
    main()
