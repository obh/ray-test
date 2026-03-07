# Ray Convergence Engine

Data State-First convergence engine using Ray + Lance. Supports multiple workflows with cross-dataset lookups.

## Workflows

- **linkedin-person-enrichment** — Ingests LinkedIn profiles, derives `cleaned_name`, `cleaned_skills`, `profile_embedding`
- **github-user-enrichment** — Ingests GitHub users, derives `cleaned_name`, `linkedin_match` (cross-dataset lookup into LinkedIn)

Workflows converge in dependency order: LinkedIn first (no dependencies), then GitHub (depends on LinkedIn for name matching).

## Local Demo (no server)

```bash
python demo.py
```

Runs the full multi-workflow lifecycle: ingest LinkedIn profiles, converge, ingest GitHub users with name overlap, converge with cross-dataset lookup, show matched profiles.

## Docker Demo

```bash
# Terminal 1: Start the server
docker compose up --build

# Terminal 2: Run full lifecycle
python test_load.py
```

## Local Server (no Docker)

```bash
# Terminal 1: Start Ray Serve
python launch_serve.py

# Terminal 2: Run full lifecycle
python test_load.py
```

## Individual Commands (server must be running)

```bash
# List workflows and dependency order
python test_load.py workflows

# LinkedIn workflow
python test_load.py ingest 500                  # Ingest 500 profiles (protobuf)
python test_load.py converge                    # Converge derived columns
python test_load.py status                      # Dataset stats
python test_load.py sample 5                    # Sample rows

# GitHub workflow
python test_load.py ingest 300 github           # Ingest 300 github users
python test_load.py converge github             # Converge (includes linkedin_match lookup)
python test_load.py status github               # Dataset stats
python test_load.py sample 5 github             # Sample rows (shows linkedin_match)

# Multi-workflow
python test_load.py converge-all                # Converge all in dependency order

# Upsert (new + updates)
python test_load.py upsert 200 50               # 200 new + 50 updates (linkedin)

# Reset
python test_load.py reset                       # Delete linkedin dataset
python test_load.py reset github                # Delete github dataset
```

## HTTP API

All endpoints accept a `workflow` query parameter (defaults to `linkedin-person-enrichment`).

| Method | Path | Description |
|--------|------|-------------|
| GET | `/workflows` | List available workflows and dependency order |
| POST | `/ingest?workflow=<name>&count=500` | Ingest random profiles |
| POST | `/upsert?workflow=<name>&count=100&updates=50` | Upsert new + updated records |
| POST | `/converge?workflow=<name>` | Run convergence for one workflow |
| POST | `/converge-all` | Converge all workflows in dependency order |
| GET | `/status?workflow=<name>` | Dataset stats (rows, fragments, columns) |
| GET | `/sample?workflow=<name>&n=5` | Sample rows |
| POST | `/reset?workflow=<name>` | Delete dataset |
