# robot_memory

A persistent, spatially-aware 3D knowledge graph and path mapping system for autonomous robots, backed by PostgreSQL with PostGIS and pgvector.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Project Structure](#project-structure)
4. [Setup](#setup)
   - [1. Clone / place files](#1-clone--place-files)
   - [2. Configure .env](#2-configure-env)
   - [3. Install Python dependencies](#3-install-python-dependencies)
   - [4. Set up PostgreSQL](#4-set-up-postgresql)
   - [5. Apply schema](#5-apply-schema)
6. [Running the demos](#running-the-demos)
7. [API Quick Reference](#api-quick-reference)
8. [Coordinate System](#coordinate-system)
9. [Runtime Memory Contract](#runtime-memory-contract)
10. [Schema Reference](#schema-reference)
11. [Tunable Parameters](#tunable-parameters)

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                       robot_memory library                       │
│                                                                  │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────────┐  │
│  │  graph.py   │   │  pathmap.py  │   │    temporal.py       │  │
│  │             │   │              │   │                      │  │
│  │ EntityNodes │   │ RuntimePath  │   │ TemporalSession      │  │
│  │ InfoNodes   │   │ Map (RAM)    │   │ interaction log      │  │
│  │ RelEdges    │   │              │   │ path log             │  │
│  └──────┬──────┘   │ flush() ↓   │   └──────────────────────┘  │
│         │          └──────┬───────┘                             │
│         └─────────────────┴──────────────────────┐             │
│                                                   ▼             │
│                              ┌────────────────────────────┐     │
│                              │         think.py           │     │
│                              │                            │     │
│                              │  think()         local 3D  │     │
│                              │  think_similar() vec(128)  │     │
│                              │  think_local_info()        │     │
│                              │  think_path()   Dijkstra   │     │
│                              │  think_nearest() 3D k-NN   │     │
│                              │  deep_think()   full graph │     │
│                              └────────────┬───────────────┘     │
└───────────────────────────────────────────┼─────────────────────┘
                                            ▼
                          ┌─────────────────────────────┐
                          │         PostgreSQL           │
                          │  PostGIS  ·  pgvector        │
                          │  3D spatial indexes          │
                          │  ST_3DDWithin / ST_3DDistance│
                          └─────────────────────────────┘
```

### Key design decisions

| Decision | Rationale |
|---|---|
| `runtime_embedding vector(128)` on entity rows | Small enough to keep all entities in RAM; enables in-process cosine filtering without touching info_nodes |
| `full_data` + `embedding vector(1536)` on info_nodes only | Heavy data never loaded unless explicitly requested via `fetch_info_full()` or `deep_think()` |
| `record_position()` → RuntimePathMap + temporal_path_log only | Zero latency writes during movement; path_nodes/path_edges untouched until `flush_path_to_db()` |
| `ST_3DDWithin` / `ST_3DDistance` throughout | True 3D Euclidean queries, not 2D footprint — correct for multi-floor buildings, drones, robot arms |
| `floor_level INT` alongside z metric | Semantic floor number for per-floor queries without z-range arithmetic |
| `anchor_entity_to_map()` is manual | Robot controls when a detection becomes a persistent spatial fact |

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | ≥ 3.11 | asyncio, dataclasses |
| PostgreSQL | ≥ 14 | with PostGIS and pgvector extensions |
| PostGIS | ≥ 3.3 | `CREATE EXTENSION postgis` |
| pgvector | ≥ 0.6.0 | `CREATE EXTENSION vector` |
| pgRouting | optional | built-in Dijkstra used by default |

### Installing PostgreSQL extensions (Ubuntu / Debian)

```bash
# PostGIS
sudo apt-get install postgresql-15-postgis-3

# pgvector
sudo apt-get install postgresql-15-pgvector
# or build from source:
# git clone https://github.com/pgvector/pgvector && cd pgvector && make && sudo make install
```

### macOS (Homebrew)

```bash
brew install postgresql@15 postgis
brew install pgvector
```

---

## Project Structure

```
.                          ← project root
├── .env                   ← all configuration (copy from .env.example, never commit)
├── requirements.txt       ← Python dependencies
├── README.md              ← this file
└── robot_memory/          ← Python package
    ├── __init__.py        ← public API surface
    ├── db.py              ← asyncpg connection pool, loads .env
    ├── graph.py           ← entity / info node / edge CRUD + 3D spatial queries
    ├── pathmap.py         ← RuntimePathMap + flush + anchoring + Dijkstra
    ├── temporal.py        ← TemporalSession (interaction log + path log)
    ├── think.py           ← think() / deep_think() / think_path() etc.
    ├── schema.sql         ← full PostgreSQL schema (run once to create tables)
    ├── demo.py            ← end-to-end multi-floor warehouse demo
    └── demo_retrieval.py  ← four retrieval scenario demos
```

---

## Setup

### 1. Clone / place files

```bash
# If using git
git clone <your-repo-url>
cd <repo-root>

# Or just place the files so the structure matches the tree above
```

### 2. Configure .env

Copy `.env` to the project root (it should already be there) and edit it:

```bash
# .env is already at the project root
# Open it and set at minimum ROBOT_DB_DSN:
nano .env
```

The minimum required change is the database DSN:

```dotenv
ROBOT_DB_DSN=postgresql://YOUR_USER:YOUR_PASSWORD@localhost:5432/robot_memory
```

See [Tunable Parameters](#tunable-parameters) for the full list.

> **.env is loaded automatically** by `robot_memory/db.py` at import time via `python-dotenv`.
> You never need to `source .env` or `export` variables manually.

### 3. Install Python dependencies

```bash
# Create a virtual environment (strongly recommended)
python3 -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# Install dependencies from project root
pip install -r requirements.txt
```

### 4. Set up PostgreSQL

```bash
# Create the database and user
psql -U postgres <<SQL
CREATE USER robot WITH PASSWORD 'robot';
CREATE DATABASE robot_memory OWNER robot;
GRANT ALL PRIVILEGES ON DATABASE robot_memory TO robot;
SQL

# Connect to the database and enable extensions
psql -U postgres -d robot_memory <<SQL
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "postgis";
SQL
```

Verify extensions loaded:

```bash
psql -U robot -d robot_memory -c "\dx"
# Should show: postgis, uuid-ossp, vector
```

### 5. Apply schema

```bash
psql -U robot -d robot_memory -f robot_memory/schema.sql
```

Verify tables were created:

```bash
psql -U robot -d robot_memory -c "\dt"
# Should list: entity_nodes, info_nodes, relationship_edges,
#              path_nodes, path_edges, temporal_path_log,
#              entity_path_anchors, temporal_interactions
```

---

## Running the demos

All demos are run from the **project root** (the directory containing `.env` and `requirements.txt`).

### Full warehouse demo (multi-floor, all features)

```bash
python -m robot_memory.demo
```

What it covers:
- Seeding 10 entities across 2 floors with 3D positions, orientations, bounding boxes, and `runtime_embedding`
- Walking a robot path that includes a stairwell climb
- Passive path recording (RuntimePathMap + temporal_path_log only — no DB writes)
- Manual flush to committed `path_nodes` / `path_edges`
- Manual entity anchoring
- `think()` with floor filter
- `think_nearest()` with 3D distances
- `think_path()` cross-floor with ascent/descent metrics
- `deep_think()` with floor breakdown

### Retrieval scenarios demo

```bash
python -m robot_memory.demo_retrieval
```

Four focused demos:
1. **Cross-floor path search** — runtime graph first, committed DB graph as fallback
2. **3D k-nearest + similarity** — spatial nearest, per-floor filter, in-process cosine filter, DB-side vector search
3. **Local window full-info** — `think_local_info()` pulling `full_data` from `info_nodes` for entities in a 3D sphere
4. **Deep search** — `deep_think()` with keyword search, entity type filter, and full-graph vector search

### Running a single scenario from Python

```python
import asyncio
import robot_memory as rm

async def main():
    # .env is loaded automatically by db.py
    await rm.init_pool()
    session = await rm.TemporalSession.start()

    # Register an entity (3D position, orientation, bounding box)
    eid = await rm.upsert_entity(
        name="Red Crate",
        summary="Heavy red plastic crate",
        entity_type="object",
        x=5.0, y=3.0, z=0.0,
        floor_level=0,
        facing_deg=90.0,
        bbox_dx=0.4, bbox_dy=0.6, bbox_dz=0.5,
        runtime_embedding=[0.1] * 128,   # your encoder output here
    )

    # Record movement (runtime only — no DB path writes)
    for x, y, z in [(0,0,0), (1,0,0), (2,0,0), (3,1,0), (4,2,0)]:
        await rm.record_position(x, y, z, floor_level=0,
                                 session_id=session.session_id)

    # Query local knowledge (3D sphere, floor 0)
    ctx = await rm.think(robot_x=4.0, robot_y=2.0, robot_z=0.0,
                         floor_level=0, radius_m=5.0)
    print(ctx.summary())

    # Flush path to DB (manual call)
    from robot_memory.pathmap import flush_path_to_db
    n, e = await flush_path_to_db(session_id=session.session_id)
    print(f"Flushed {n} nodes, {e} edges")

    # Anchor entity to map
    await rm.anchor_entity_to_map(eid)

    await rm.close_pool()

asyncio.run(main())
```

---

## API Quick Reference

### Lifecycle

```python
await rm.init_pool()       # call once at startup
await rm.close_pool()      # call at shutdown
session = await rm.TemporalSession.start()   # one session per robot run
```

### Entity graph

```python
# Create / update entity
eid = await rm.upsert_entity(
    name, summary, entity_type, x, y, z,
    floor_level=0, facing_deg=None, pitch_deg=None,
    bbox_dx=None, bbox_dy=None, bbox_dz=None,
    tags=[], runtime_embedding=[...],   # vec(128)
)

# Update location only
await rm.update_entity_location(eid, x, y, z, floor_level=0)

# Attach detailed info (full_data and 1536-dim embedding stored but not loaded at runtime)
await rm.add_info_node(eid, crucial_words=[...], full_data="...", embedding=[...])

# Fetch
entity  = await rm.get_entity(eid)
entity  = await rm.get_entity_by_name("Red Crate")

# Relationships
await rm.upsert_edge(eid_a, eid_b, rel_type="near", rel_name="...", weight=1.0)
edges   = await rm.get_edges_for_entity(eid)
```

### Path map

```python
# Passive recording (runtime only, zero DB path writes)
rt_id = await rm.record_position(x, y, z,
    floor_level=0, heading_deg=45.0, pitch_deg=0.0,
    session_id=session.session_id)

# MANUAL: commit runtime map to DB
from robot_memory.pathmap import flush_path_to_db
nodes_written, edges_written = await flush_path_to_db(session_id=session.session_id)

# MANUAL: link entity to map (requires flush first)
await rm.anchor_entity_to_map(eid)

# Inspect runtime map
rt = rm.get_runtime_map()
print(rt.node_count, rt.edge_count)
print(rt.nodes_on_floor(1))   # nodes on floor 1
```

### Think functions

```python
# Localised 3D query (runtime-safe — includes runtime_embedding, no blobs)
ctx = await rm.think(robot_x, robot_y, robot_z,
                     floor_level=0, radius_m=10.0)
# ctx.entities          — EntityNode list (with runtime_embedding)
# ctx.info_metas        — {entity_id: [InfoNodeMeta]}
# ctx.edges             — {entity_id: [RelationshipEdge]}
# ctx.local_path_nodes  — committed PathNode list
# ctx.filter_by_similarity(query_emb, min_similarity=0.6)  — in-process cosine filter

# Full info for local window (loads full_data from info_nodes)
lctx = await rm.think_local_info(robot_x, robot_y, robot_z,
                                  floor_level=0, radius_m=10.0)

# DB-side 3D cosine search (runtime_embedding vec(128))
results = await rm.think_similar(
    query_embedding=[...],        # vec(128)
    top_k=10,
    radius_m=15.0,                # optional 3D spatial bound
    center_x=x, center_y=y, center_z=z,
    floor_level=0,                # optional floor filter
    min_similarity=0.5,
)  # → List[(EntityNode, similarity_score)]

# Pathfinding (runtime graph first, DB fallback)
path = await rm.think_path(eid_a, eid_b)
# path.waypoints         — List[PathNode]
# path.total_dist_3d_m   — true 3D distance
# path.total_dist_2d_m   — horizontal-only distance
# path.total_ascent_m    — cumulative upward change
# path.total_descent_m   — cumulative downward change
# path.floor_transitions()  → [(from_floor, to_floor), ...]

# K-nearest entities (3D Euclidean, optional floor filter)
nearest = await rm.think_nearest(eid, k=5, radius_m=20.0, floor_level=None)
# → List[(EntityNode, distance_3d_m)]

# Full graph (expensive — use for planning / memory consolidation)
dctx = await rm.deep_think(include_full_embeddings=False)
# dctx.all_entities, dctx.full_info, dctx.all_edges
# dctx.search_info("keyword")      — keyword search across full_data
# dctx.entities_by_type("place")
# dctx.entities_on_floor(1)

# Focused single-entity query
info = await rm.think_about(eid)
# {"entity": EntityNode, "info_metas": [...], "edges": [...]}
```

### Temporal session

```python
# Log entity interaction
await session.log(entity_id=eid, x=x, y=y, z=z,
                  floor_level=0, notes="picked up")

# Get most recent path log id (for linking interactions to path positions)
log_id = await session.latest_path_log_id()

# Query session records
records     = await session.get_records()
path_log    = await session.get_path_log()
log_status  = await session.path_log_status()
# {"total": N, "flushed": N, "pending": N}

# Print summary
await session.dump_summary()
```

---

## Coordinate System

```
        North (y+)
           ↑
           │
           │  z+ (Up)
           │  ↑
           │  │
           └──┼──────────── East (x+)
           origin (0,0,0)

floor_level 0 = ground floor  (z ≈ 0.0 … 2.9 m)
floor_level 1 = first floor   (z ≈ 3.0 … 5.9 m)
floor_level N = ground + N × ROBOT_FLOOR_HEIGHT_M
```

All distances are in **metres**. All angles are in **degrees**.

| Field | Unit | Convention |
|---|---|---|
| `x` | metres | East from world origin |
| `y` | metres | North from world origin |
| `z` | metres | Up from ground (z=0) |
| `floor_level` | integer | 0 = ground floor |
| `facing_deg` | degrees | Yaw: 0 = North, clockwise positive |
| `pitch_deg` | degrees | Nose-up: 0 = horizontal, +90 = straight up |
| `bbox_dx/dy/dz` | metres | Half-extents from object centre |

---

## Runtime Memory Contract

| Column / Field | In runtime memory | Loaded by |
|---|:---:|---|
| entity name, summary, weight, location, floor_level | ✅ | `think()`, `get_entity()` |
| entity facing_deg, pitch_deg, bbox_* | ✅ | `think()`, `get_entity()` |
| entity tags, entity_type | ✅ | `think()`, `get_entity()` |
| `runtime_embedding vector(128)` | ✅ | `think()`, `get_entity()` |
| info_node `crucial_words`, `weight` | ✅ | `think()` (meta only) |
| info_node `full_data` | ❌ | `think_local_info()`, `fetch_info_full()` |
| info_node `embedding vector(1536)` | ❌ | `deep_think(include_full_embeddings=True)` |
| entity image / video / audio pointers | ❌ | `fetch_info_full()` only |
| path_nodes / path_edges (committed) | ❌ | loaded at pathfinding time |
| RuntimePathMap (in-process) | ✅ RAM | updated by `record_position()` |
| temporal_path_log | write-only ✅ | flushed by `flush_path_to_db()` |

---

## Schema Reference

### Core tables

| Table | Purpose |
|---|---|
| `entity_nodes` | Knowledge graph nodes — entities the robot knows about |
| `info_nodes` | Detailed knowledge per entity (heavy — not in runtime RAM) |
| `relationship_edges` | Typed edges between entity pairs |
| `path_nodes` | Committed 3D waypoints (written by `flush_path_to_db()`) |
| `path_edges` | Traversals between waypoints (3D + 2D distances + ΔZ) |
| `temporal_path_log` | Write-ahead position log (every `record_position()` call) |
| `entity_path_anchors` | Manual links: entity ↔ committed path_node |
| `temporal_interactions` | Per-session entity encounter log |

### Helper views

| View | Purpose |
|---|---|
| `vw_entity_runtime` | All runtime-safe entity columns including `runtime_embedding` |
| `vw_path_nodes_3d` | path_nodes with extracted x/y/z coordinates |

### 3D spatial functions used

| PostgreSQL function | Purpose |
|---|---|
| `ST_3DDWithin(geom_a, geom_b, radius)` | True 3D sphere query |
| `ST_3DDistance(geom_a, geom_b)` | True 3D Euclidean distance |
| `ST_MakePoint(x, y, z)` | Construct PointZ |
| `ST_SetSRID(geom, 0)` | Assign local Cartesian SRID 0 |
| `ST_X / ST_Y / ST_Z` | Extract coordinates |
| `embedding <=> query` | pgvector cosine distance (for vector search) |

---

## Tunable Parameters

All parameters live in `.env` at the project root. `db.py` loads them automatically via `python-dotenv`.

| Variable | Default | Description |
|---|---|---|
| `ROBOT_DB_DSN` | `postgresql://robot:robot@localhost:5432/robot_memory` | asyncpg connection string |
| `ROBOT_DB_POOL_MIN` | `2` | Minimum DB connection pool size |
| `ROBOT_DB_POOL_MAX` | `10` | Maximum DB connection pool size |
| `ROBOT_THINK_RADIUS_M` | `10.0` | Default 3D sphere radius for `think()` in metres |
| `ROBOT_K_NEAREST` | `5` | Default k for `think_nearest()` |
| `ROBOT_MERGE_RADIUS_3D_M` | `0.5` | Path-node merge threshold in metres (3D Euclidean) |
| `ROBOT_FLOOR_HEIGHT_M` | `3.0` | Floor height for auto floor_level estimation from z |
| `ROBOT_RUNTIME_EMBEDDING_DIM` | `128` | Dimensionality of `runtime_embedding` on entity rows |
| `ROBOT_FULL_EMBEDDING_DIM` | `1536` | Dimensionality of `embedding` on info_nodes |
| `ROBOT_LOG_LEVEL` | `INFO` | Python logging level: `DEBUG \| INFO \| WARNING \| ERROR` |

---

## Common Issues

**`ST_3DDWithin` not found**
PostGIS is not installed or the extension was not created. Run:
```sql
CREATE EXTENSION postgis;
```

**`vector` type not found**
pgvector is not installed. See [Prerequisites](#prerequisites).

**`No committed path_nodes. Call flush_path_to_db() first.`**
`anchor_entity_to_map()` requires at least one `flush_path_to_db()` call. Run a flush before anchoring.

**Embeddings not matching**
Ensure `runtime_embedding` is always a unit-normalised vector of exactly `ROBOT_RUNTIME_EMBEDDING_DIM` dimensions (default 128). The pgvector `<=>` operator computes cosine distance; non-normalised vectors give incorrect similarity scores.

**`python-dotenv` not installed**
`db.py` catches the `ImportError` and falls back to shell environment variables. Either install it (`pip install python-dotenv`) or export the variables manually:
```bash
export ROBOT_DB_DSN="postgresql://..."
```