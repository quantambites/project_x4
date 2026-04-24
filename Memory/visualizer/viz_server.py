"""
viz_server.py
─────────────
FastAPI backend for the robot_memory world visualizer.
Runs alongside your robot_memory project and reads live data from PostgreSQL.

Usage:
    pip install fastapi uvicorn asyncpg python-dotenv
    python viz_server.py
    # Then open http://localhost:8765 in your browser

The server reads the same .env as the rest of robot_memory (ROBOT_DB_DSN, etc).
CORS is open so the HTML file can also be opened directly from disk.
"""

from __future__ import annotations
import os
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

import asyncpg
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# ── Load .env ─────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    _env = Path(__file__).resolve().parent / ".env"
    if not _env.exists():
        _env = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=_env, override=False)
except ImportError:
    pass

DSN      = os.getenv("ROBOT_DB_DSN", "postgresql://postgres:1234@localhost:5432/memory")
LOG_LEVEL = os.getenv("ROBOT_LOG_LEVEL", "INFO").upper()
PORT      = int(os.getenv("VIZ_PORT", "8765"))

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
log = logging.getLogger("viz_server")

app   = FastAPI(title="robot_memory visualizer", version="2.0")
_pool: Optional[asyncpg.Pool] = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── Pool lifecycle ─────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    global _pool
    log.info("Connecting to %s", DSN.split("@")[-1])
    try:
        _pool = await asyncpg.create_pool(DSN, min_size=1, max_size=5)
        log.info("DB pool ready")
    except Exception as e:
        log.error("DB connection failed: %s", e)
        _pool = None

@app.on_event("shutdown")
async def shutdown():
    if _pool:
        await _pool.close()

async def pool() -> asyncpg.Pool:
    if _pool is None:
        raise HTTPException(503, "Database not connected")
    return _pool

# ── Helper: safely serialize asyncpg Record to dict ──────────────────────────
def rec(row) -> Dict[str, Any]:
    d = dict(row)
    for k, v in d.items():
        if isinstance(v, datetime):
            d[k] = v.isoformat()
        elif hasattr(v, '__iter__') and not isinstance(v, (str, bytes, list)):
            d[k] = list(v)
    return d

# ─────────────────────────────────────────────────────────────────────────────
# API endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    """Check DB connectivity and return table row counts."""
    db = await pool()
    counts = {}
    tables = [
        "entity_nodes", "info_nodes", "relationship_edges",
        "path_nodes", "path_edges", "temporal_path_log",
        "temporal_interactions", "raw_temporal_nodes", "consolidation_log",
    ]
    async with db.acquire() as conn:
        for t in tables:
            try:
                row = await conn.fetchrow(f"SELECT COUNT(*) AS n FROM {t}")
                counts[t] = row["n"]
            except Exception:
                counts[t] = -1
    return {"status": "ok", "dsn_host": DSN.split("@")[-1], "counts": counts}


@app.get("/api/entities")
async def get_entities():
    """All entity nodes with 3D position, type, tags, top crucial words."""
    db = await pool()
    async with db.acquire() as conn:
        rows = await conn.fetch("""
            SELECT
                e.node_id::text,
                e.name,
                e.summary,
                e.weight,
                ST_X(e.location::geometry) AS x,
                ST_Y(e.location::geometry) AS y,
                ST_Z(e.location::geometry) AS z,
                e.floor_level,
                e.facing_deg,
                e.pitch_deg,
                e.bbox_dx, e.bbox_dy, e.bbox_dz,
                e.entity_type,
                e.tags,
                e.location_ts,
                e.created_at,
                e.updated_at,
                e.image_ptrs,
                e.video_ptr,
                e.audio_ptr,
                ARRAY(
                    SELECT unnest(i.crucial_words)
                    FROM info_nodes i
                    WHERE i.entity_id = e.node_id
                    ORDER BY i.weight DESC
                    LIMIT 1
                ) AS top_words,
                (SELECT COUNT(*) FROM info_nodes i WHERE i.entity_id = e.node_id) AS info_count
            FROM entity_nodes e
            ORDER BY e.weight DESC, e.created_at DESC
        """)
    return [rec(r) for r in rows]


@app.get("/api/edges")
async def get_edges():
    """All relationship edges."""
    db = await pool()
    async with db.acquire() as conn:
        rows = await conn.fetch("""
            SELECT
                edge_id::text,
                summary,
                rel_type,
                rel_name,
                node_id_1::text,
                node_id_2::text,
                weight,
                directed,
                created_at
            FROM relationship_edges
            ORDER BY weight DESC
        """)
    return [rec(r) for r in rows]


@app.get("/api/path_nodes")
async def get_path_nodes(floor: Optional[int] = None):
    """Committed path nodes (global map), optionally filtered by floor."""
    db = await pool()
    async with db.acquire() as conn:
        if floor is not None:
            rows = await conn.fetch("""
                SELECT
                    path_node_id::text,
                    ST_X(position::geometry) AS x,
                    ST_Y(position::geometry) AS y,
                    ST_Z(position::geometry) AS z,
                    floor_level,
                    heading_deg,
                    pitch_deg,
                    visited_at,
                    visit_count,
                    tags
                FROM path_nodes
                WHERE floor_level = $1
                ORDER BY visited_at ASC
            """, floor)
        else:
            rows = await conn.fetch("""
                SELECT
                    path_node_id::text,
                    ST_X(position::geometry) AS x,
                    ST_Y(position::geometry) AS y,
                    ST_Z(position::geometry) AS z,
                    floor_level,
                    heading_deg,
                    pitch_deg,
                    visited_at,
                    visit_count,
                    tags
                FROM path_nodes
                ORDER BY visited_at ASC
            """)
    return [rec(r) for r in rows]


@app.get("/api/path_edges")
async def get_path_edges():
    """Committed path edges with 3D distances."""
    db = await pool()
    async with db.acquire() as conn:
        rows = await conn.fetch("""
            SELECT
                path_edge_id::text,
                from_node_id::text,
                to_node_id::text,
                distance_3d_m,
                distance_2d_m,
                delta_z_m,
                traversal_cost,
                traversal_count
            FROM path_edges
        """)
    return [rec(r) for r in rows]


@app.get("/api/temporal_path")
async def get_temporal_path(session_id: Optional[str] = None, limit: int = 500):
    """
    Runtime path log (unflushed positions).
    If session_id given, returns only that session. Otherwise latest session.
    """
    db = await pool()
    async with db.acquire() as conn:
        if session_id:
            rows = await conn.fetch("""
                SELECT
                    log_id::text,
                    session_id::text,
                    x, y, z,
                    floor_level,
                    heading_deg,
                    pitch_deg,
                    tags,
                    recorded_at,
                    flushed
                FROM temporal_path_log
                WHERE session_id = $1::uuid
                ORDER BY recorded_at ASC
                LIMIT $2
            """, session_id, limit)
        else:
            # Latest session
            rows = await conn.fetch("""
                SELECT
                    tpl.log_id::text,
                    tpl.session_id::text,
                    tpl.x, tpl.y, tpl.z,
                    tpl.floor_level,
                    tpl.heading_deg,
                    tpl.pitch_deg,
                    tpl.tags,
                    tpl.recorded_at,
                    tpl.flushed
                FROM temporal_path_log tpl
                INNER JOIN (
                    SELECT session_id FROM temporal_path_log
                    ORDER BY recorded_at DESC LIMIT 1
                ) latest ON tpl.session_id = latest.session_id
                ORDER BY tpl.recorded_at ASC
                LIMIT $1
            """, limit)
    return [rec(r) for r in rows]


@app.get("/api/raw_nodes")
async def get_raw_nodes(
    processed: Optional[bool] = None,
    limit: int = 200,
    session_id: Optional[str] = None,
):
    """Raw temporal nodes — unprocessed sensor data."""
    db = await pool()
    clauses = []
    params  = [limit]

    if processed is not None:
        params.append(processed)
        clauses.append(f"processed = ${len(params)}")
    if session_id:
        params.append(session_id)
        clauses.append(f"session_id = ${len(params)}::uuid")

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    async with db.acquire() as conn:
        rows = await conn.fetch(f"""
            SELECT
                raw_id::text,
                session_id::text,
                data_type,
                raw_text,
                raw_json::text,
                x, y, z,
                floor_level,
                heading_deg,
                captured_at,
                related_entity_id::text,
                processed,
                consolidation_id::text
            FROM raw_temporal_nodes
            {where}
            ORDER BY captured_at DESC
            LIMIT $1
        """, *params)
    return [rec(r) for r in rows]


@app.get("/api/temporal_interactions")
async def get_temporal_interactions(limit: int = 100):
    """Recent temporal interaction log entries."""
    db = await pool()
    async with db.acquire() as conn:
        rows = await conn.fetch("""
            SELECT
                interaction_id::text,
                session_id::text,
                entity_id::text,
                entity_name,
                ST_X(location_snap::geometry) AS x,
                ST_Y(location_snap::geometry) AS y,
                ST_Z(location_snap::geometry) AS z,
                floor_level,
                interaction_ts,
                notes
            FROM temporal_interactions
            ORDER BY interaction_ts DESC
            LIMIT $1
        """, limit)
    return [rec(r) for r in rows]


@app.get("/api/info_nodes")
async def get_info_nodes(entity_id: Optional[str] = None):
    """Info nodes — crucial_words + full_data (no embedding to keep response small)."""
    db = await pool()
    async with db.acquire() as conn:
        if entity_id:
            rows = await conn.fetch("""
                SELECT
                    node_id::text,
                    entity_id::text,
                    full_data,
                    weight,
                    crucial_words,
                    created_at,
                    image_ptr,
                    video_ptr,
                    audio_ptr
                FROM info_nodes
                WHERE entity_id = $1::uuid
                ORDER BY weight DESC
            """, entity_id)
        else:
            rows = await conn.fetch("""
                SELECT
                    node_id::text,
                    entity_id::text,
                    full_data,
                    weight,
                    crucial_words,
                    created_at,
                    image_ptr,
                    video_ptr,
                    audio_ptr
                FROM info_nodes
                ORDER BY weight DESC
                LIMIT 500
            """)
    return [rec(r) for r in rows]


@app.get("/api/anchors")
async def get_anchors():
    """Entity–path anchor links."""
    db = await pool()
    async with db.acquire() as conn:
        rows = await conn.fetch("""
            SELECT
                anchor_id::text,
                entity_id::text,
                path_node_id::text,
                anchored_at,
                confidence
            FROM entity_path_anchors
            ORDER BY anchored_at DESC
        """)
    return [rec(r) for r in rows]


@app.get("/api/consolidation_log")
async def get_consolidation_log(limit: int = 20):
    """Recent LLM consolidation run history."""
    db = await pool()
    async with db.acquire() as conn:
        rows = await conn.fetch("""
            SELECT
                consolidation_id::text,
                session_id::text,
                started_at,
                finished_at,
                raw_nodes_processed,
                entities_created,
                entities_updated,
                edges_created,
                info_nodes_created,
                llm_model,
                llm_calls,
                status,
                error_msg,
                summary_text
            FROM consolidation_log
            ORDER BY started_at DESC
            LIMIT $1
        """, limit)
    return [rec(r) for r in rows]


@app.get("/api/sessions")
async def get_sessions():
    """Distinct sessions from temporal_path_log, newest first."""
    db = await pool()
    async with db.acquire() as conn:
        rows = await conn.fetch("""
            SELECT
                session_id::text,
                MIN(recorded_at) AS started,
                MAX(recorded_at) AS last_seen,
                COUNT(*)         AS position_count,
                COUNT(*) FILTER (WHERE flushed) AS flushed_count
            FROM temporal_path_log
            GROUP BY session_id
            ORDER BY last_seen DESC
            LIMIT 20
        """)
    return [rec(r) for r in rows]


@app.get("/api/world_bounds")
async def get_world_bounds():
    """
    Compute the actual bounding box of all known positions in the DB
    (entity_nodes + path_nodes + temporal_path_log).
    Used by the frontend to auto-scale the map canvas.
    """
    db = await pool()
    async with db.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT
                MIN(x) AS min_x, MAX(x) AS max_x,
                MIN(y) AS min_y, MAX(y) AS max_y,
                MIN(z) AS min_z, MAX(z) AS max_z,
                MIN(floor_level) AS min_floor,
                MAX(floor_level) AS max_floor
            FROM (
                SELECT
                    ST_X(location::geometry) AS x,
                    ST_Y(location::geometry) AS y,
                    ST_Z(location::geometry) AS z,
                    floor_level
                FROM entity_nodes
                WHERE location IS NOT NULL
                UNION ALL
                SELECT
                    ST_X(position::geometry) AS x,
                    ST_Y(position::geometry) AS y,
                    ST_Z(position::geometry) AS z,
                    floor_level
                FROM path_nodes
                UNION ALL
                SELECT x, y, z, floor_level
                FROM temporal_path_log
            ) all_pts
        """)
    if row and row["min_x"] is not None:
        return rec(row)
    return {"min_x": -1, "max_x": 12, "min_y": -1, "max_y": 12,
            "min_z": 0, "max_z": 5, "min_floor": 0, "max_floor": 1}


@app.get("/api/snapshot")
async def snapshot():
    """
    Single endpoint that returns everything the visualizer needs in one call.
    Reduces round-trips: entities + edges + path_nodes + raw_nodes (last 100)
    + world_bounds + health counts.
    """
    db = await pool()
    async with db.acquire() as conn:
        entities = await conn.fetch("""
            SELECT
                e.node_id::text,
                e.name,
                e.summary,
                e.weight,
                ST_X(e.location::geometry) AS x,
                ST_Y(e.location::geometry) AS y,
                ST_Z(e.location::geometry) AS z,
                e.floor_level,
                e.facing_deg,
                e.pitch_deg,
                e.bbox_dx, e.bbox_dy, e.bbox_dz,
                e.entity_type,
                e.tags,
                e.location_ts,
                e.image_ptrs,
                e.video_ptr,
                e.audio_ptr,
                ARRAY(
                    SELECT unnest(i.crucial_words)
                    FROM info_nodes i WHERE i.entity_id = e.node_id
                    ORDER BY i.weight DESC LIMIT 1
                ) AS top_words,
                (SELECT COUNT(*) FROM info_nodes i WHERE i.entity_id = e.node_id)::int AS info_count
            FROM entity_nodes e
            ORDER BY e.weight DESC
        """)

        edges = await conn.fetch("""
            SELECT edge_id::text, rel_type, rel_name,
                   node_id_1::text, node_id_2::text, weight, directed, summary
            FROM relationship_edges ORDER BY weight DESC
        """)

        path_nodes = await conn.fetch("""
            SELECT path_node_id::text,
                   ST_X(position::geometry) AS x,
                   ST_Y(position::geometry) AS y,
                   ST_Z(position::geometry) AS z,
                   floor_level, heading_deg, visit_count
            FROM path_nodes ORDER BY visited_at ASC
        """)

        # Latest session's runtime path (unflushed)
        runtime_path = await conn.fetch("""
            SELECT tpl.x, tpl.y, tpl.z, tpl.floor_level, tpl.heading_deg, tpl.recorded_at
            FROM temporal_path_log tpl
            INNER JOIN (
                SELECT session_id FROM temporal_path_log
                ORDER BY recorded_at DESC LIMIT 1
            ) s ON tpl.session_id = s.session_id
            ORDER BY tpl.recorded_at ASC
            LIMIT 500
        """)

        raw_nodes = await conn.fetch("""
            SELECT raw_id::text, data_type, raw_text, raw_json::text,
                   x, y, z, floor_level, heading_deg, captured_at, processed,
                   related_entity_id::text, session_id::text
            FROM raw_temporal_nodes
            ORDER BY captured_at DESC LIMIT 150
        """)

        interactions = await conn.fetch("""
            SELECT entity_name,
                   ST_X(location_snap::geometry) AS x,
                   ST_Y(location_snap::geometry) AS y,
                   ST_Z(location_snap::geometry) AS z,
                   floor_level, interaction_ts, notes
            FROM temporal_interactions
            ORDER BY interaction_ts DESC LIMIT 50
        """)

        bounds = await conn.fetchrow("""
            SELECT MIN(x)-1 AS min_x, MAX(x)+1 AS max_x,
                   MIN(y)-1 AS min_y, MAX(y)+1 AS max_y,
                   MIN(floor_level) AS min_floor,
                   MAX(floor_level) AS max_floor
            FROM (
                SELECT ST_X(location::geometry) AS x,
                       ST_Y(location::geometry) AS y,
                       floor_level
                FROM entity_nodes WHERE location IS NOT NULL
                UNION ALL
                SELECT ST_X(position::geometry), ST_Y(position::geometry), floor_level
                FROM path_nodes
                UNION ALL
                SELECT x, y, floor_level FROM temporal_path_log
            ) p
        """)

        consol = await conn.fetch("""
            SELECT consolidation_id::text, started_at, status,
                   entities_created, edges_created, info_nodes_created,
                   llm_model, llm_calls, summary_text
            FROM consolidation_log ORDER BY started_at DESC LIMIT 5
        """)

        counts = {}
        for t in ["entity_nodes","relationship_edges","path_nodes","raw_temporal_nodes",
                  "temporal_path_log","temporal_interactions","consolidation_log"]:
            try:
                r = await conn.fetchrow(f"SELECT COUNT(*)::int AS n FROM {t}")
                counts[t] = r["n"]
            except Exception:
                counts[t] = 0

    b = dict(bounds) if bounds and bounds["min_x"] is not None else \
        {"min_x":-1,"max_x":12,"min_y":-1,"max_y":12,"min_floor":0,"max_floor":1}
    for k,v in b.items():
        if isinstance(v, float) and v != v:  # NaN guard
            b[k] = 0

    return {
        "entities":     [rec(r) for r in entities],
        "edges":        [rec(r) for r in edges],
        "path_nodes":   [rec(r) for r in path_nodes],
        "runtime_path": [rec(r) for r in runtime_path],
        "raw_nodes":    [rec(r) for r in raw_nodes],
        "interactions": [rec(r) for r in interactions],
        "bounds":       b,
        "counts":       counts,
        "consolidation_log": [rec(r) for r in consol],
    }


# ── Serve the HTML visualizer at root ─────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = Path(__file__).parent / "viz_ui.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<p>viz_ui.html not found next to viz_server.py</p>", status_code=404)


if __name__ == "__main__":
    uvicorn.run("viz_server:app", host="0.0.0.0", port=PORT, reload=True)