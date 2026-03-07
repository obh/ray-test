"""
End-to-end demo of the Data State-First convergence approach
with multi-workflow support and cross-dataset lookups.

Demonstrates:
1. LinkedIn workflow: ingest profiles, converge derived columns
2. GitHub workflow: ingest users, converge with cross-dataset lookup
3. linkedin_match column populated via name similarity matching
4. Convergence respects workflow dependency order
"""
import shutil
import random
from pathlib import Path

import ray
import lance

from config import parse_config, workflow_dependency_order
from convergence import ConvergenceEngine, ingest_raw_data

# ---------------------------------------------------------------------------
# Fake data generation
# ---------------------------------------------------------------------------
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


def generate_linkedin_profiles(n: int, id_offset: int = 0) -> list[dict]:
    profiles = []
    for i in range(n):
        mid = f"member_{id_offset + i:06d}"
        first = random.choice(FIRST_NAMES)
        last = random.choice(LAST_NAMES)
        name = f"  {first} {last}  "
        num_skills = random.randint(2, 6)
        skills = ", ".join(random.sample(SKILLS, num_skills))
        if random.random() > 0.5:
            skills = skills.replace(",", " , ").upper()
        profiles.append({"member_id": mid, "name": name, "skills": skills})
    return profiles


def generate_github_profiles(n: int, linkedin_profiles: list[dict], overlap: int = 0) -> list[dict]:
    """Generate github profiles. `overlap` of them will share names with linkedin profiles."""
    profiles = []
    # Overlapping users (same names as linkedin)
    for i in range(min(overlap, len(linkedin_profiles))):
        ln = linkedin_profiles[i]
        raw_name = ln["name"]  # messy whitespace version
        username = f"gh_{raw_name.strip().lower().replace(' ', '_')}_{i}"
        num_repos = random.randint(1, 5)
        repos = ", ".join(random.sample(REPOS, num_repos))
        profiles.append({"github_username": username, "name": raw_name, "repos": repos})

    # Non-overlapping users
    for i in range(n - overlap):
        first = random.choice(FIRST_NAMES)
        last = random.choice(LAST_NAMES)
        username = f"gh_unique_{first.lower()}_{last.lower()}_{i}"
        name = f"  {first} {last}  "
        num_repos = random.randint(1, 5)
        repos = ", ".join(random.sample(REPOS, num_repos))
        profiles.append({"github_username": username, "name": name, "repos": repos})

    return profiles


def print_separator(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def main():
    workflows_dir = Path(__file__).parent / "workflows"

    # Load configs
    configs = {}
    for yaml_file in sorted(workflows_dir.glob("*.yaml")):
        cfg = parse_config(yaml_file)
        configs[cfg.name] = cfg

    # Clean up from previous runs
    for cfg in configs.values():
        if Path(cfg.dataset_path).exists():
            shutil.rmtree(cfg.dataset_path)

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=4)

    engines = {name: ConvergenceEngine(cfg) for name, cfg in configs.items()}
    dep_order = workflow_dependency_order(configs)
    print(f"Workflow dependency order: {dep_order}")

    linkedin_cfg = configs["linkedin-person-enrichment"]
    github_cfg = configs["github-user-enrichment"]
    linkedin_engine = engines["linkedin-person-enrichment"]
    github_engine = engines["github-user-enrichment"]

    # -----------------------------------------------------------------------
    # PHASE 1: Ingest LinkedIn profiles
    # -----------------------------------------------------------------------
    print_separator("PHASE 1: Ingest LinkedIn Profiles (1000 profiles)")

    linkedin_profiles = generate_linkedin_profiles(1000)
    batch_size = 500
    for i in range(0, len(linkedin_profiles), batch_size):
        batch = linkedin_profiles[i:i + batch_size]
        frag_count = ingest_raw_data(linkedin_cfg.dataset_path, batch)
        print(f"  Batch {i // batch_size + 1}: wrote {len(batch)} records "
              f"(fragments: {frag_count})")

    ds = lance.dataset(linkedin_cfg.dataset_path)
    print(f"\n  LinkedIn dataset: {ds.count_rows()} rows, "
          f"{len(ds.get_fragments())} fragments")

    # -----------------------------------------------------------------------
    # PHASE 2: Converge LinkedIn
    # -----------------------------------------------------------------------
    print_separator("PHASE 2: Converge LinkedIn (compute derived columns)")

    plan = linkedin_engine.diff()
    print(f"  Plan: {len(plan.columns_to_compute)} columns to compute: "
          f"{[c.name for c in plan.columns_to_compute]}")

    result = linkedin_engine.converge(plan)
    print(f"\n  Result: computed={result.columns_computed}, "
          f"rows={result.rows_processed}, duration={result.duration_seconds:.2f}s")

    ds = lance.dataset(linkedin_cfg.dataset_path)
    print(f"  Final columns: {ds.schema.names}")

    # -----------------------------------------------------------------------
    # PHASE 3: Ingest GitHub users (some overlap with LinkedIn names)
    # -----------------------------------------------------------------------
    print_separator("PHASE 3: Ingest GitHub Users (500 total, 200 overlap with LinkedIn)")

    github_profiles = generate_github_profiles(500, linkedin_profiles, overlap=200)
    for i in range(0, len(github_profiles), batch_size):
        batch = github_profiles[i:i + batch_size]
        frag_count = ingest_raw_data(github_cfg.dataset_path, batch)
        print(f"  Batch {i // batch_size + 1}: wrote {len(batch)} records "
              f"(fragments: {frag_count})")

    ds = lance.dataset(github_cfg.dataset_path)
    print(f"\n  GitHub dataset: {ds.count_rows()} rows, "
          f"{len(ds.get_fragments())} fragments")

    # -----------------------------------------------------------------------
    # PHASE 4: Converge GitHub (cross-dataset lookup)
    # -----------------------------------------------------------------------
    print_separator("PHASE 4: Converge GitHub (cleaned_name + linkedin_match)")

    plan = github_engine.diff()
    print(f"  Plan: {len(plan.columns_to_compute)} columns to compute: "
          f"{[c.name for c in plan.columns_to_compute]}")

    result = github_engine.converge(plan)
    print(f"\n  Result: computed={result.columns_computed}, "
          f"rows={result.rows_processed}, duration={result.duration_seconds:.2f}s")

    # -----------------------------------------------------------------------
    # PHASE 5: Show cross-dataset links
    # -----------------------------------------------------------------------
    print_separator("PHASE 5: Cross-Dataset Links")

    ds = lance.dataset(github_cfg.dataset_path)
    print(f"  GitHub columns: {ds.schema.names}")
    sample = ds.to_table(columns=["github_username", "name", "cleaned_name", "linkedin_match"])
    rows = sample.to_pylist()

    matched = [r for r in rows if r["linkedin_match"] is not None]
    unmatched = [r for r in rows if r["linkedin_match"] is None]

    print(f"\n  Total GitHub users: {len(rows)}")
    print(f"  Matched to LinkedIn: {len(matched)}")
    print(f"  Unmatched: {len(unmatched)}")

    print(f"\n  Sample matched users:")
    for row in matched[:5]:
        print(f"    {row['github_username']}: "
              f"name={row['cleaned_name']!r} -> "
              f"linkedin={row['linkedin_match']}")

    if unmatched:
        print(f"\n  Sample unmatched users:")
        for row in unmatched[:3]:
            print(f"    {row['github_username']}: "
                  f"name={row['cleaned_name']!r} -> no match")

    # -----------------------------------------------------------------------
    # PHASE 6: Converge-all (dependency order)
    # -----------------------------------------------------------------------
    print_separator("PHASE 6: Converge-All (idempotency check)")

    for name in dep_order:
        engine = engines[name]
        plan = engine.diff()
        result = engine.converge(plan)
        print(f"  {name}: computed={result.columns_computed}, "
              f"rows={result.rows_processed}")

    # -----------------------------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------------------------
    print_separator("SUMMARY")
    for name, cfg in configs.items():
        try:
            ds = lance.dataset(cfg.dataset_path)
            print(f"  {name}:")
            print(f"    Dataset: {cfg.dataset_path}")
            print(f"    Primary key: {cfg.primary_key}")
            print(f"    Rows: {ds.count_rows()}")
            print(f"    Fragments: {len(ds.get_fragments())}")
            print(f"    Schema: {ds.schema.names}")
        except Exception:
            print(f"  {name}: no data")

    print(f"\n  Key points demonstrated:")
    print(f"    1. Multiple workflows with independent datasets")
    print(f"    2. Cross-dataset lookup (github -> linkedin by name)")
    print(f"    3. Dependency ordering (linkedin converges before github)")
    print(f"    4. Convergence is idempotent (pass #2 was a no-op)")
    print()

    ray.shutdown()


if __name__ == "__main__":
    main()
