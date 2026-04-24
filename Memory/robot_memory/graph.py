"""
robot_memory/graph.py
─────────────────────
Entity node + info node + relationship edge management.

Performance
───────────
think() uses batch_load_info_metas() and batch_load_edges() which each
execute a single ANY($ids) query instead of N queries.  This eliminates
the N+1 pattern and is the single biggest latency win for the think() call.

All spatial queries are TRUE 3D:
  • ST_3DDWithin  — spherical radius in 3D space
  • ST_3DDistance — Euclidean distance across (x,y,z)

Coordinate system
─────────────────
  x  = metres East  from world origin
  y  = metres North from world origin
  z  = metres Up    from ground floor (z=0.0)
  floor_level = integer floor number (0 = ground)
  facing_deg  = yaw  (0 = North, clockwise)
  pitch_deg   = nose-up tilt (-90…+90)
  bbox_dx/dy/dz = half-extents of physical bounding box in metres
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import os
from .db import get_pool

THINK_RADIUS_M = float(os.getenv("ROBOT_THINK_RADIUS_M", "10.0"))
K_NEAREST      = int(os.getenv("ROBOT_K_NEAREST",        "5"))


# ─────────────────────────────────────────────────────────────────────────────
# DTOs
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EntityNode:
    node_id:     str
    name:        str
    summary:     Optional[str]
    weight:      float
    x:           Optional[float]
    y:           Optional[float]
    z:           Optional[float]
    floor_level: int
    location_ts: Optional[datetime]
    facing_deg:  Optional[float]
    pitch_deg:   Optional[float]
    bbox_dx:     Optional[float]
    bbox_dy:     Optional[float]
    bbox_dz:     Optional[float]
    entity_type: Optional[str]
    tags:        List[str]                       = field(default_factory=list)
    top_words:   List[str]                       = field(default_factory=list)
    runtime_embedding: Optional[List[float]]     = field(default=None, repr=False)


@dataclass
class InfoNodeMeta:
    node_id:       str
    entity_id:     str
    weight:        float
    crucial_words: List[str] = field(default_factory=list)


@dataclass
class RelationshipEdge:
    edge_id:   str
    summary:   Optional[str]
    rel_type:  str
    rel_name:  Optional[str]
    node_id_1: str
    node_id_2: str
    weight:    float
    directed:  bool


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _euclidean_3d(ax, ay, az, bx, by, bz) -> float:
    return math.sqrt((ax-bx)**2 + (ay-by)**2 + (az-bz)**2)


def _parse_embedding(raw) -> Optional[List[float]]:
    if raw is None:
        return None
    if isinstance(raw, str):
        return [float(v) for v in raw.strip("[]").split(",") if v.strip()]
    return list(raw)


_ENTITY_SELECT = """
    SELECT
        e.node_id::text,
        e.name,
        e.summary,
        e.weight,
        ST_X(e.location::geometry)  AS x,
        ST_Y(e.location::geometry)  AS y,
        ST_Z(e.location::geometry)  AS z,
        e.floor_level,
        e.location_ts,
        e.facing_deg,
        e.pitch_deg,
        e.bbox_dx,
        e.bbox_dy,
        e.bbox_dz,
        e.entity_type,
        e.tags,
        e.runtime_embedding::text,
        ARRAY(
            SELECT unnest(crucial_words)
            FROM info_nodes i
            WHERE i.entity_id = e.node_id
            ORDER BY i.weight DESC
            LIMIT 1
        ) AS top_words
    FROM entity_nodes e
"""


def _row_to_entity(row) -> EntityNode:
    return EntityNode(
        node_id=row["node_id"],
        name=row["name"],
        summary=row["summary"],
        weight=row["weight"],
        x=row["x"], y=row["y"], z=row["z"],
        floor_level=row["floor_level"] or 0,
        location_ts=row["location_ts"],
        facing_deg=row["facing_deg"],
        pitch_deg=row["pitch_deg"],
        bbox_dx=row["bbox_dx"],
        bbox_dy=row["bbox_dy"],
        bbox_dz=row["bbox_dz"],
        entity_type=row["entity_type"],
        tags=list(row["tags"] or []),
        top_words=list(row["top_words"] or []),
        runtime_embedding=_parse_embedding(row["runtime_embedding"]),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entity CRUD
# ─────────────────────────────────────────────────────────────────────────────

async def upsert_entity(
    name:        str,
    summary:     str        = "",
    weight:      float      = 1.0,
    x:           float      = 0.0,
    y:           float      = 0.0,
    z:           float      = 0.0,
    floor_level: int        = 0,
    entity_type: str        = "object",
    tags:        List[str]  = None,
    facing_deg:  float      = None,
    pitch_deg:   float      = None,
    bbox_dx:     float      = None,
    bbox_dy:     float      = None,
    bbox_dz:     float      = None,
    *,
    runtime_embedding: List[float] = None,
    image_ptrs:  List[str]  = None,
    video_ptr:   str        = None,
    audio_ptr:   str        = None,
) -> str:
    """Insert or update an entity node. Returns node_id string."""
    pool = await get_pool()
    tags       = tags or []
    image_ptrs = image_ptrs or []
    emb_str    = f"[{','.join(map(str, runtime_embedding))}]" if runtime_embedding else None

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO entity_nodes
                (name, summary, weight,
                 location, location_ts, floor_level,
                 facing_deg, pitch_deg,
                 bbox_dx, bbox_dy, bbox_dz,
                 entity_type, tags, runtime_embedding,
                 image_ptrs, video_ptr, audio_ptr)
            VALUES
                ($1, $2, $3,
                 ST_SetSRID(ST_MakePoint($4, $5, $6), 0), NOW(), $7,
                 $8, $9,
                 $10, $11, $12,
                 $13, $14, $15::vector,
                 $16, $17, $18)
            ON CONFLICT DO NOTHING
            RETURNING node_id::text
            """,
            name, summary, weight,
            x, y, z, floor_level,
            facing_deg, pitch_deg,
            bbox_dx, bbox_dy, bbox_dz,
            entity_type, tags, emb_str,
            image_ptrs, video_ptr, audio_ptr,
        )
        if row:
            return row["node_id"]
        # Conflict — fetch existing
        row = await conn.fetchrow(
            "SELECT node_id::text FROM entity_nodes WHERE name=$1", name
        )
        return row["node_id"]


async def update_entity_location(
    node_id:     str,
    x:           float,
    y:           float,
    z:           float = 0.0,
    floor_level: int   = None,
) -> None:
    """Passively update an entity's last-seen 3D position."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        if floor_level is not None:
            await conn.execute(
                """
                UPDATE entity_nodes
                SET location    = ST_SetSRID(ST_MakePoint($1,$2,$3),0),
                    location_ts = NOW(),
                    floor_level = $4,
                    updated_at  = NOW()
                WHERE node_id = $5::uuid
                """,
                x, y, z, floor_level, node_id,
            )
        else:
            await conn.execute(
                """
                UPDATE entity_nodes
                SET location    = ST_SetSRID(ST_MakePoint($1,$2,$3),0),
                    location_ts = NOW(),
                    updated_at  = NOW()
                WHERE node_id = $4::uuid
                """,
                x, y, z, node_id,
            )


async def get_entity(node_id: str) -> Optional[EntityNode]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            _ENTITY_SELECT + "WHERE e.node_id = $1::uuid", node_id,
        )
        return _row_to_entity(row) if row else None


async def get_entity_by_name(name: str) -> Optional[EntityNode]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            _ENTITY_SELECT + "WHERE e.name = $1", name,
        )
        return _row_to_entity(row) if row else None


# ─────────────────────────────────────────────────────────────────────────────
# Info-node CRUD
# ─────────────────────────────────────────────────────────────────────────────

async def add_info_node(
    entity_id:     str,
    crucial_words: List[str],
    weight:        float = 1.0,
    *,
    full_data:     str         = None,
    embedding:     List[float] = None,
    image_ptr:     str         = None,
    video_ptr:     str         = None,
    audio_ptr:     str         = None,
) -> str:
    pool = await get_pool()
    async with pool.acquire() as conn:
        emb_str = f"[{','.join(map(str, embedding))}]" if embedding else None
        row = await conn.fetchrow(
            """
            INSERT INTO info_nodes
                (entity_id, full_data, embedding, weight,
                 crucial_words, image_ptr, video_ptr, audio_ptr)
            VALUES ($1::uuid, $2, $3::vector, $4, $5, $6, $7, $8)
            RETURNING node_id::text
            """,
            entity_id, full_data, emb_str, weight,
            crucial_words, image_ptr, video_ptr, audio_ptr,
        )
        return row["node_id"]


async def get_info_meta(entity_id: str) -> List[InfoNodeMeta]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT node_id::text, entity_id::text, weight, crucial_words
            FROM info_nodes WHERE entity_id=$1::uuid ORDER BY weight DESC
            """,
            entity_id,
        )
        return [InfoNodeMeta(r["node_id"], r["entity_id"], r["weight"],
                             list(r["crucial_words"] or [])) for r in rows]


async def fetch_info_full(info_node_id: str) -> Optional[Dict[str, Any]]:
    """Fetch heavy info-node data. Use only in deep_think() or explicit calls."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT node_id::text, entity_id::text, full_data, embedding::text,
                   weight, crucial_words, image_ptr, video_ptr, audio_ptr
            FROM info_nodes WHERE node_id=$1::uuid
            """,
            info_node_id,
        )
        return dict(row) if row else None


# ─────────────────────────────────────────────────────────────────────────────
# BATCH loaders  (used by think() — eliminates N+1 queries)
# ─────────────────────────────────────────────────────────────────────────────

async def batch_load_info_metas(
    entity_ids: List[str],
) -> Dict[str, List[InfoNodeMeta]]:
    """
    Load info-node metadata for a list of entity IDs in ONE query.
    Returns {entity_id: [InfoNodeMeta, ...]}
    Replaces N individual get_info_meta() calls in think().
    """
    if not entity_ids:
        return {}
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT node_id::text, entity_id::text, weight, crucial_words
            FROM info_nodes
            WHERE entity_id = ANY($1::uuid[])
            ORDER BY entity_id, weight DESC
            """,
            entity_ids,
        )
    result: Dict[str, List[InfoNodeMeta]] = {eid: [] for eid in entity_ids}
    for r in rows:
        eid = r["entity_id"]
        if eid in result:
            result[eid].append(
                InfoNodeMeta(r["node_id"], eid, r["weight"],
                             list(r["crucial_words"] or []))
            )
    return result


async def batch_load_edges(
    entity_ids: List[str],
) -> Dict[str, List[RelationshipEdge]]:
    """
    Load relationship edges for a list of entity IDs in ONE query.
    Returns {entity_id: [RelationshipEdge, ...]} — edge appears in both
    endpoint buckets so callers can look up by either side.
    Replaces N individual get_edges_for_entity() calls in think().
    """
    if not entity_ids:
        return {}
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT edge_id::text, summary, rel_type, rel_name,
                   node_id_1::text, node_id_2::text, weight, directed
            FROM relationship_edges
            WHERE node_id_1 = ANY($1::uuid[]) OR node_id_2 = ANY($1::uuid[])
            ORDER BY weight DESC
            """,
            entity_ids,
        )
    result: Dict[str, List[RelationshipEdge]] = {eid: [] for eid in entity_ids}
    seen: set = set()
    for r in rows:
        edge = RelationshipEdge(
            r["edge_id"], r["summary"], r["rel_type"], r["rel_name"],
            r["node_id_1"], r["node_id_2"], r["weight"], r["directed"],
        )
        # Avoid duplicate edge objects; add to both endpoint buckets
        if edge.edge_id not in seen:
            seen.add(edge.edge_id)
        for nid in (edge.node_id_1, edge.node_id_2):
            if nid in result:
                result[nid].append(edge)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Relationship edges
# ─────────────────────────────────────────────────────────────────────────────

async def upsert_edge(
    node_id_1: str,
    node_id_2: str,
    rel_type:  str,
    rel_name:  str   = "",
    summary:   str   = "",
    weight:    float = 1.0,
    directed:  bool  = False,
) -> str:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO relationship_edges
                (summary, rel_type, rel_name, node_id_1, node_id_2, weight, directed)
            VALUES ($1, $2, $3, $4::uuid, $5::uuid, $6, $7)
            ON CONFLICT (node_id_1, node_id_2, rel_type) DO UPDATE SET
                weight     = relationship_edges.weight + 0.1,
                summary    = EXCLUDED.summary,
                updated_at = NOW()
            RETURNING edge_id::text
            """,
            summary, rel_type, rel_name, node_id_1, node_id_2, weight, directed,
        )
        return row["edge_id"]


async def get_edges_for_entity(node_id: str) -> List[RelationshipEdge]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT edge_id::text, summary, rel_type, rel_name,
                   node_id_1::text, node_id_2::text, weight, directed
            FROM relationship_edges
            WHERE node_id_1=$1::uuid OR node_id_2=$1::uuid
            ORDER BY weight DESC
            """,
            node_id,
        )
        return [RelationshipEdge(r["edge_id"], r["summary"], r["rel_type"], r["rel_name"],
                                 r["node_id_1"], r["node_id_2"], r["weight"], r["directed"])
                for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# 3D Spatial queries
# ─────────────────────────────────────────────────────────────────────────────

async def entities_in_radius(
    x:        float,
    y:        float,
    z:        float = 0.0,
    radius_m: float = THINK_RADIUS_M,
    floor_level: int = None,
) -> List[EntityNode]:
    """
    Return entities within a 3D sphere of radius_m around (x,y,z).
    Uses ST_3DDWithin — true 3D Euclidean distance, not 2D footprint.
    Optional floor_level filter: restricts to a single floor.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        if floor_level is not None:
            rows = await conn.fetch(
                _ENTITY_SELECT + """
                WHERE ST_3DDWithin(
                    e.location,
                    ST_SetSRID(ST_MakePoint($1,$2,$3),0),
                    $4
                )
                AND e.floor_level = $5
                ORDER BY e.weight DESC
                """,
                x, y, z, radius_m, floor_level,
            )
        else:
            rows = await conn.fetch(
                _ENTITY_SELECT + """
                WHERE ST_3DDWithin(
                    e.location,
                    ST_SetSRID(ST_MakePoint($1,$2,$3),0),
                    $4
                )
                ORDER BY e.weight DESC
                """,
                x, y, z, radius_m,
            )
        return [_row_to_entity(r) for r in rows]


async def k_nearest_entities(
    node_id: str,
    k: int = K_NEAREST,
    radius_m: float = 100.0,
    floor_level: int = None,
) -> List[Tuple[EntityNode, float]]:
    """K nearest entities to a given entity, by 3D Euclidean distance."""
    pool = await get_pool()
    floor_filter = "AND e.floor_level = $4" if floor_level is not None else ""
    floor_param  = [floor_level] if floor_level is not None else []

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            WITH origin AS (
                SELECT location FROM entity_nodes WHERE node_id = $1::uuid
            )
            SELECT
                e.node_id::text,
                e.name,
                e.summary,
                e.weight,
                ST_X(e.location::geometry) AS x,
                ST_Y(e.location::geometry) AS y,
                ST_Z(e.location::geometry) AS z,
                e.floor_level,
                e.location_ts,
                e.facing_deg,
                e.pitch_deg,
                e.bbox_dx,
                e.bbox_dy,
                e.bbox_dz,
                e.entity_type,
                e.tags,
                e.runtime_embedding::text,
                ARRAY(
                    SELECT unnest(crucial_words)
                    FROM info_nodes i
                    WHERE i.entity_id = e.node_id
                    ORDER BY i.weight DESC
                    LIMIT 1
                ) AS top_words,
                ST_3DDistance(e.location, o.location) AS dist_3d_m
            FROM entity_nodes e, origin o
            WHERE e.node_id <> $1::uuid
              AND ST_3DDWithin(e.location, o.location, $3)
              {floor_filter}
            ORDER BY dist_3d_m ASC
            LIMIT $2
            """,
            node_id, k, radius_m, *floor_param,
        )
        return [(_row_to_entity(r), r["dist_3d_m"]) for r in rows]


async def entities_by_runtime_similarity(
    query_embedding: List[float],
    top_k: int = 10,
    radius_m: float = None,
    center_x: float = None,
    center_y: float = None,
    center_z: float = None,
    floor_level: int = None,
    min_similarity: float = 0.0,
) -> List[Tuple[EntityNode, float]]:
    """
    Cosine similarity search against runtime_embedding (vector(128)).
    Optional 3D spatial filter + optional floor filter.
    Returns (entity, similarity).
    """
    pool = await get_pool()
    emb_str = f"[{','.join(map(str, query_embedding))}]"

    params = [emb_str, top_k, min_similarity]
    spatial_clause = ""
    floor_clause   = ""

    if radius_m is not None and center_x is not None:
        params += [center_x, center_y, center_z or 0.0, radius_m]
        n = len(params)
        spatial_clause = f"""
        AND ST_3DDWithin(
            e.location,
            ST_SetSRID(ST_MakePoint(${n-3}, ${n-2}, ${n-1}), 0),
            ${n}
        )
        """

    if floor_level is not None:
        params.append(floor_level)
        floor_clause = f"AND e.floor_level = ${len(params)}"

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT
                e.node_id::text,
                e.name,
                e.summary,
                e.weight,
                ST_X(e.location::geometry) AS x,
                ST_Y(e.location::geometry) AS y,
                ST_Z(e.location::geometry) AS z,
                e.floor_level,
                e.location_ts,
                e.facing_deg,
                e.pitch_deg,
                e.bbox_dx,
                e.bbox_dy,
                e.bbox_dz,
                e.entity_type,
                e.tags,
                e.runtime_embedding::text,
                ARRAY(
                    SELECT unnest(crucial_words)
                    FROM info_nodes i
                    WHERE i.entity_id = e.node_id
                    ORDER BY i.weight DESC
                    LIMIT 1
                ) AS top_words,
                1 - (e.runtime_embedding <=> $1::vector) AS similarity
            FROM entity_nodes e
            WHERE e.runtime_embedding IS NOT NULL
              AND 1 - (e.runtime_embedding <=> $1::vector) >= $3
              {spatial_clause}
              {floor_clause}
            ORDER BY e.runtime_embedding <=> $1::vector
            LIMIT $2
            """,
            *params,
        )
        return [(_row_to_entity(r), r["similarity"]) for r in rows]