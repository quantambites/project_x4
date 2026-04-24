"""
robot_memory/think.py
─────────────────────
All spatial queries are 3D.

Performance optimisations vs v1
────────────────────────────────
think() now uses batch_load_info_metas() and batch_load_edges() which each
issue ONE query for ALL entities in the local window, replacing the previous
N+1 pattern.  For a window with 10 entities this reduces DB round-trips from
~21 to ~3.

think()             — localised 3D sphere query (batch-loaded edges + info).
think_similar()     — 3D-bounded cosine search on runtime_embedding.
think_local_info()  — full info_nodes for entities in 3D local window.
deep_think()        — entire graph, all heavy data.
think_path()        — entity-to-entity 3D pathfinding.
think_nearest()     — k-nearest entities, true 3D distance.
think_about()       — single-entity focused query.
"""

from __future__ import annotations
import os
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .db import get_pool
from .graph import (
    EntityNode, InfoNodeMeta, RelationshipEdge,
    entities_in_radius, get_edges_for_entity, get_info_meta, fetch_info_full,
    k_nearest_entities, entities_by_runtime_similarity,
    batch_load_info_metas, batch_load_edges,
    THINK_RADIUS_M,
)
from .pathmap import (
    PathNode, PathEdge, PathResult,
    get_local_map, find_path_between_entities,
)

_FLOOR_HEIGHT_M = float(os.getenv("ROBOT_FLOOR_HEIGHT_M", "3.0"))


# ─────────────────────────────────────────────────────────────────────────────
# ThoughtContext
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ThoughtContext:
    robot_x:      float
    robot_y:      float
    robot_z:      float
    floor_level:  int
    radius_m:     float
    entities:             List[EntityNode]                  = field(default_factory=list)
    info_metas:           Dict[str, List[InfoNodeMeta]]     = field(default_factory=dict)
    edges:                Dict[str, List[RelationshipEdge]] = field(default_factory=dict)
    local_path_nodes:     List[PathNode]                    = field(default_factory=list)
    local_path_edges:     List[PathEdge]                    = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"ThoughtContext  robot=({self.robot_x:.1f}, {self.robot_y:.1f}, {self.robot_z:.1f})"
            f"  floor={self.floor_level}  radius={self.radius_m}m",
            f"  entities  : {len(self.entities)}",
            f"  path nodes: {len(self.local_path_nodes)}"
            f"  path edges: {len(self.local_path_edges)}",
        ]
        for e in self.entities:
            d3 = math.sqrt(
                (e.x - self.robot_x)**2 + (e.y - self.robot_y)**2 + (e.z - self.robot_z)**2
            ) if e.x is not None else float("nan")
            kw   = ", ".join(e.top_words[:4]) if e.top_words else "—"
            emb  = f"vec({len(e.runtime_embedding)})" if e.runtime_embedding else "no-emb"
            bbox = f"  bbox=({e.bbox_dx:.1f},{e.bbox_dy:.1f},{e.bbox_dz:.1f})" if e.bbox_dx else ""
            orient = f"  hdg={e.facing_deg:.0f}°" if e.facing_deg is not None else ""
            lines.append(
                f"    [{e.entity_type or '?':8}] {e.name:<22}"
                f"  d3={d3:5.1f}m  fl={e.floor_level}  w={e.weight:.2f}"
                f"  {emb}{orient}{bbox}  [{kw}]"
            )
        return "\n".join(lines)

    def filter_by_similarity(
        self,
        query_embedding: List[float],
        min_similarity:  float = 0.6,
    ) -> List[Tuple[EntityNode, float]]:
        """In-process cosine filter on already-loaded runtime_embeddings."""
        def cosine(a, b):
            dot = sum(x*y for x, y in zip(a, b))
            na  = math.sqrt(sum(x*x for x in a))
            nb  = math.sqrt(sum(x*x for x in b))
            return dot / (na * nb + 1e-9)
        results = [
            (e, cosine(query_embedding, e.runtime_embedding))
            for e in self.entities if e.runtime_embedding
        ]
        return sorted(
            [(e, s) for e, s in results if s >= min_similarity],
            key=lambda t: t[1], reverse=True,
        )

    def entities_on_floor(self, floor_level: int) -> List[EntityNode]:
        return [e for e in self.entities if e.floor_level == floor_level]


# ─────────────────────────────────────────────────────────────────────────────
# think()  — optimised local 3D query
# ─────────────────────────────────────────────────────────────────────────────

async def think(
    robot_x:     float,
    robot_y:     float,
    robot_z:     float = 0.0,
    floor_level: int   = None,
    radius_m:    float = THINK_RADIUS_M,
    load_edges:  bool  = True,
    load_info:   bool  = True,
) -> ThoughtContext:
    """
    Localised 3D knowledge query — OPTIMISED.

    DB round-trips vs v1:
      v1: 1 (spatial) + N (info metas) + N (edges) + 1 (path map) = 2N+2
      v2: 1 (spatial) + 1 (batch info) + 1 (batch edges) + 1 (path map) = 4

    Set floor_level to restrict results to a single floor.
    runtime_embedding always included; use ctx.filter_by_similarity() in-process.
    """
    resolved_floor = (
        floor_level if floor_level is not None
        else int(round(robot_z / _FLOOR_HEIGHT_M))
    )
    ctx = ThoughtContext(
        robot_x=robot_x, robot_y=robot_y, robot_z=robot_z,
        floor_level=resolved_floor, radius_m=radius_m,
    )

    # 1. Spatial entity fetch (1 query)
    ctx.entities = await entities_in_radius(
        robot_x, robot_y, robot_z, radius_m, floor_level=floor_level,
    )

    if ctx.entities:
        entity_ids = [e.node_id for e in ctx.entities]

        # 2. Batch info-meta fetch (1 query for ALL entities)
        if load_info:
            ctx.info_metas = await batch_load_info_metas(entity_ids)

        # 3. Batch edge fetch (1 query for ALL entities)
        if load_edges:
            ctx.edges = await batch_load_edges(entity_ids)

    # 4. Local path map (1 query)
    ctx.local_path_nodes, ctx.local_path_edges = await get_local_map(
        robot_x, robot_y, robot_z,
        radius_m=radius_m * 1.5,
        floor_level=floor_level,
    )

    return ctx


# ─────────────────────────────────────────────────────────────────────────────
# think_similar — DB-side cosine search
# ─────────────────────────────────────────────────────────────────────────────

async def think_similar(
    query_embedding: List[float],
    top_k:           int   = 10,
    radius_m:        float = None,
    center_x:        float = None,
    center_y:        float = None,
    center_z:        float = None,
    floor_level:     int   = None,
    min_similarity:  float = 0.5,
) -> List[Tuple[EntityNode, float]]:
    """DB-side cosine search on runtime_embedding (vector(128))."""
    return await entities_by_runtime_similarity(
        query_embedding=query_embedding,
        top_k=top_k,
        radius_m=radius_m,
        center_x=center_x,
        center_y=center_y,
        center_z=center_z,
        floor_level=floor_level,
        min_similarity=min_similarity,
    )


# ─────────────────────────────────────────────────────────────────────────────
# think_local_info — full info_nodes for spatial window
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LocalInfoContext:
    robot_x:     float
    robot_y:     float
    robot_z:     float
    floor_level: int
    radius_m:    float
    data: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"LocalInfoContext @ ({self.robot_x:.1f},{self.robot_y:.1f},{self.robot_z:.1f})"
            f"  fl={self.floor_level}  radius={self.radius_m}m  —  {len(self.data)} entities"
        ]
        for eid, d in self.data.items():
            e     = d["entity"]
            infos = d["full_info"]
            lines.append(f"  {e.name}  fl={e.floor_level}  ({len(infos)} info nodes)")
            for fi in infos:
                snippet = (fi.get("full_data") or "")[:80].replace("\n", " ")
                lines.append(f"    · {snippet}…")
        return "\n".join(lines)


async def think_local_info(
    robot_x:     float,
    robot_y:     float,
    robot_z:     float = 0.0,
    floor_level: int   = None,
    radius_m:    float = THINK_RADIUS_M,
) -> LocalInfoContext:
    """Full info_node content for all entities in the 3D spatial window."""
    resolved_floor = (
        floor_level if floor_level is not None
        else int(round(robot_z / _FLOOR_HEIGHT_M))
    )
    ctx = LocalInfoContext(
        robot_x=robot_x, robot_y=robot_y, robot_z=robot_z,
        floor_level=resolved_floor, radius_m=radius_m,
    )
    entities = await entities_in_radius(
        robot_x, robot_y, robot_z, radius_m, floor_level=floor_level,
    )
    for ent in entities:
        metas      = await get_info_meta(ent.node_id)
        full_infos = [fi for m in metas if (fi := await fetch_info_full(m.node_id))]
        edges      = await get_edges_for_entity(ent.node_id)
        ctx.data[ent.node_id] = {"entity": ent, "full_info": full_infos, "edges": edges}
    return ctx


# ─────────────────────────────────────────────────────────────────────────────
# deep_think — full graph
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DeepThoughtContext:
    all_entities: List[EntityNode]
    full_info:    List[Dict[str, Any]]
    all_edges:    List[RelationshipEdge]
    entity_count: int = 0
    info_count:   int = 0
    edge_count:   int = 0

    def __post_init__(self):
        self.entity_count = len(self.all_entities)
        self.info_count   = len(self.full_info)
        self.edge_count   = len(self.all_edges)

    def summary(self) -> str:
        floors = sorted(set(e.floor_level for e in self.all_entities))
        return (
            f"DeepThoughtContext: {self.entity_count} entities  "
            f"{self.info_count} info nodes  {self.edge_count} edges  "
            f"floors={floors}"
        )

    def search_info(self, keyword: str) -> List[Dict[str, Any]]:
        kw = keyword.lower()
        return [fi for fi in self.full_info
                if fi.get("full_data") and kw in fi["full_data"].lower()]

    def entities_by_type(self, entity_type: str) -> List[EntityNode]:
        return [e for e in self.all_entities if e.entity_type == entity_type]

    def entities_on_floor(self, floor_level: int) -> List[EntityNode]:
        return [e for e in self.all_entities if e.floor_level == floor_level]


async def deep_think(include_full_embeddings: bool = False) -> DeepThoughtContext:
    """Full knowledge graph including full_data and (optionally) full embeddings."""
    pool = await get_pool()
    from .graph import _ENTITY_SELECT, _row_to_entity, RelationshipEdge
    async with pool.acquire() as conn:
        ent_rows     = await conn.fetch(_ENTITY_SELECT + "ORDER BY e.weight DESC")
        all_entities = [_row_to_entity(r) for r in ent_rows]

        emb_col   = "embedding::text" if include_full_embeddings else "'<omitted>' AS embedding"
        info_rows = await conn.fetch(
            f"SELECT node_id::text, entity_id::text, full_data, {emb_col}, weight, "
            f"crucial_words, image_ptr, video_ptr, audio_ptr FROM info_nodes ORDER BY weight DESC"
        )
        full_info = [dict(r) for r in info_rows]

        edge_rows = await conn.fetch(
            "SELECT edge_id::text, summary, rel_type, rel_name, "
            "node_id_1::text, node_id_2::text, weight, directed "
            "FROM relationship_edges ORDER BY weight DESC"
        )
        all_edges = [
            RelationshipEdge(r["edge_id"], r["summary"], r["rel_type"], r["rel_name"],
                             r["node_id_1"], r["node_id_2"], r["weight"], r["directed"])
            for r in edge_rows
        ]
    return DeepThoughtContext(all_entities=all_entities, full_info=full_info, all_edges=all_edges)


# ─────────────────────────────────────────────────────────────────────────────
# Focused helpers
# ─────────────────────────────────────────────────────────────────────────────

async def think_path(
    entity_id_a:    str,
    entity_id_b:    str,
    prefer_runtime: bool = True,
) -> Optional[PathResult]:
    """3D path between two entities."""
    return await find_path_between_entities(entity_id_a, entity_id_b, prefer_runtime)


async def think_nearest(
    entity_id:   str,
    k:           int   = 5,
    radius_m:    float = 50.0,
    floor_level: int   = None,
) -> List[Tuple[EntityNode, float]]:
    """K nearest entities, true 3D distance."""
    return await k_nearest_entities(entity_id, k=k, radius_m=radius_m, floor_level=floor_level)


async def think_about(entity_id: str) -> Optional[Dict[str, Any]]:
    """Single-entity: metadata + info metas + edges. Runtime-safe."""
    from .graph import get_entity
    entity = await get_entity(entity_id)
    if not entity:
        return None
    return {
        "entity":     entity,
        "info_metas": await get_info_meta(entity_id),
        "edges":      await get_edges_for_entity(entity_id),
    }