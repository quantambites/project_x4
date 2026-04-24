"""
robot_memory/pathmap.py
───────────────────────
Full 3D path mapping + entity anchoring + pathfinding.

Coordinate system
─────────────────
  x           = metres East  from world origin
  y           = metres North from world origin
  z           = metres Up    from ground floor (z=0)
  floor_level = integer floor number (0 = ground)
  heading_deg = yaw  (0 = North, clockwise)
  pitch_deg   = nose-up tilt — useful for ramps/stairs/drones

Path edges carry both 3D and 2D distances:
  distance_3d_m — true Euclidean (used for cost / pathfinding)
  distance_2d_m — horizontal-only (useful for occupancy map display)
  delta_z_m     — signed vertical change (+ = ascending)

Runtime policy
──────────────
  record_position() → RuntimePathMap (in-process) + temporal_path_log only.
  flush_path_to_db() MANUAL → commits to path_nodes + path_edges.
  anchor_entity_to_map() MANUAL → links entity to committed path_node.
  find_path_between_entities() → tries runtime map first, falls back to DB.
"""

from __future__ import annotations
import uuid
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import os
from .db import get_pool

# Tunables — read from env (set in .env or shell)
MERGE_RADIUS_3D_M = float(os.getenv("ROBOT_MERGE_RADIUS_3D_M", "0.5"))
_FLOOR_HEIGHT_M   = float(os.getenv("ROBOT_FLOOR_HEIGHT_M",    "3.0"))


# ─────────────────────────────────────────────────────────────────────────────
# DTOs
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PathNode:
    path_node_id: str
    x:            float
    y:            float
    z:            float
    floor_level:  int
    heading_deg:  Optional[float]
    pitch_deg:    Optional[float]
    visited_at:   datetime
    visit_count:  int
    tags:         List[str] = field(default_factory=list)

    @property
    def xyz(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass
class PathEdge:
    path_edge_id:  str
    from_node_id:  str
    to_node_id:    str
    distance_3d_m: float
    distance_2d_m: float
    delta_z_m:     float
    traversal_cost: float
    traversal_count: int


@dataclass
class PathResult:
    waypoints:    List[PathNode]
    total_dist_3d_m: float
    total_dist_2d_m: float
    total_ascent_m:  float    # cumulative upward z change
    total_descent_m: float    # cumulative downward z change (positive value)
    total_cost:      float

    def floor_transitions(self) -> List[Tuple[int, int]]:
        """Return (from_floor, to_floor) pairs where floor changes along path."""
        transitions = []
        for i in range(len(self.waypoints) - 1):
            a = self.waypoints[i].floor_level
            b = self.waypoints[i+1].floor_level
            if a != b:
                transitions.append((a, b))
        return transitions


# ─────────────────────────────────────────────────────────────────────────────
# RuntimePathMap — pure in-process 3D graph
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _RTNode:
    rt_id:       str
    x:           float
    y:           float
    z:           float
    floor_level: int
    heading_deg: Optional[float]
    pitch_deg:   Optional[float]
    visit_count: int = 1
    tags:        List[str] = field(default_factory=list)


@dataclass
class _RTEdge:
    from_id:      str
    to_id:        str
    dist_3d:      float
    dist_2d:      float
    delta_z:      float
    cost:         float
    count:        int = 1


class RuntimePathMap:
    """Pure in-process 3D path graph. No DB access."""

    def __init__(self):
        self._nodes:   Dict[str, _RTNode]               = {}
        self._edges:   Dict[Tuple[str, str], _RTEdge]   = {}
        self._last_id: Optional[str]                    = None

    def record(
        self,
        x:           float,
        y:           float,
        z:           float = 0.0,
        floor_level: int   = 0,
        heading_deg: float = None,
        pitch_deg:   float = None,
        tags:        List[str] = None,
    ) -> str:
        tags = tags or []
        existing_id = self._nearest_within_3d(x, y, z, MERGE_RADIUS_3D_M)
        if existing_id:
            self._nodes[existing_id].visit_count += 1
            current_id = existing_id
        else:
            current_id = str(uuid.uuid4())
            self._nodes[current_id] = _RTNode(
                rt_id=current_id, x=x, y=y, z=z,
                floor_level=floor_level,
                heading_deg=heading_deg,
                pitch_deg=pitch_deg,
                tags=tags,
            )
        if self._last_id and self._last_id != current_id:
            prev = self._nodes[self._last_id]
            d3   = _euclidean_3d(prev.x, prev.y, prev.z, x, y, z)
            d2   = _euclidean_2d(prev.x, prev.y, x, y)
            dz   = z - prev.z
            key  = (self._last_id, current_id)
            rkey = (current_id, self._last_id)
            if key in self._edges:
                self._edges[key].count += 1
            else:
                self._edges[key]  = _RTEdge(self._last_id, current_id, d3, d2, dz, d3)
                self._edges[rkey] = _RTEdge(current_id, self._last_id, d3, d2, -dz, d3)
        self._last_id = current_id
        return current_id

    def _nearest_within_3d(self, x, y, z, r) -> Optional[str]:
        best_id, best_d = None, r
        for nid, n in self._nodes.items():
            d = _euclidean_3d(n.x, n.y, n.z, x, y, z)
            if d <= best_d:
                best_d, best_id = d, nid
        return best_id

    def dijkstra(self, start_id: str, end_id: str) -> Optional[PathResult]:
        import heapq
        if start_id not in self._nodes or end_id not in self._nodes:
            return None
        adj: Dict[str, List[Tuple[str, float]]] = {n: [] for n in self._nodes}
        for (fr, to), e in self._edges.items():
            adj[fr].append((to, e.cost))
        dist = {n: math.inf for n in self._nodes}
        prev: Dict[str, Optional[str]] = {n: None for n in self._nodes}
        dist[start_id] = 0.0
        heap = [(0.0, start_id)]
        while heap:
            cost, u = heapq.heappop(heap)
            if cost > dist[u]: continue
            if u == end_id: break
            for v, w in adj.get(u, []):
                alt = dist[u] + w
                if alt < dist[v]:
                    dist[v] = alt; prev[v] = u
                    heapq.heappush(heap, (alt, v))
        if math.isinf(dist.get(end_id, math.inf)):
            return None
        path_ids: List[str] = []
        cur: Optional[str] = end_id
        while cur:
            path_ids.append(cur); cur = prev.get(cur)
        path_ids.reverse()
        waypoints = [_rt_to_path_node(self._nodes[p]) for p in path_ids if p in self._nodes]
        return _build_path_result(waypoints, dist[end_id])

    @property
    def node_count(self): return len(self._nodes)
    @property
    def edge_count(self): return len(self._edges) // 2
    @property
    def last_id(self): return self._last_id

    def nodes_on_floor(self, floor_level: int) -> List[_RTNode]:
        return [n for n in self._nodes.values() if n.floor_level == floor_level]


def _rt_to_path_node(n: _RTNode) -> PathNode:
    return PathNode(
        path_node_id=n.rt_id, x=n.x, y=n.y, z=n.z,
        floor_level=n.floor_level,
        heading_deg=n.heading_deg, pitch_deg=n.pitch_deg,
        visited_at=datetime.now(), visit_count=n.visit_count, tags=n.tags,
    )


def _build_path_result(waypoints: List[PathNode], total_cost: float) -> PathResult:
    d3 = d2 = asc = desc = 0.0
    for i in range(len(waypoints) - 1):
        a, b = waypoints[i], waypoints[i+1]
        d3  += _euclidean_3d(a.x, a.y, a.z, b.x, b.y, b.z)
        d2  += _euclidean_2d(a.x, a.y, b.x, b.y)
        dz   = b.z - a.z
        if dz > 0: asc  += dz
        else:      desc -= dz
    return PathResult(waypoints=waypoints,
                      total_dist_3d_m=d3, total_dist_2d_m=d2,
                      total_ascent_m=asc, total_descent_m=desc,
                      total_cost=total_cost)


# Module-level singleton
_runtime_map = RuntimePathMap()

def get_runtime_map() -> RuntimePathMap:
    return _runtime_map


# ─────────────────────────────────────────────────────────────────────────────
# record_position
# ─────────────────────────────────────────────────────────────────────────────

async def record_position(
    x:           float,
    y:           float,
    z:           float      = 0.0,
    floor_level: int        = 0,
    heading_deg: float      = None,
    pitch_deg:   float      = None,
    tags:        List[str]  = None,
    *,
    session_id:  str        = None,
) -> str:
    """
    Record the robot's current 3D position.
    1. Updates RuntimePathMap in-process (zero DB write, instant).
    2. Appends to temporal_path_log (cheap INSERT).
    path_nodes/path_edges are NOT touched until flush_path_to_db().
    Returns runtime node id.
    """
    rt_id = _runtime_map.record(x, y, z, floor_level, heading_deg, pitch_deg, tags or [])
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO temporal_path_log
                (session_id, x, y, z, floor_level, heading_deg, pitch_deg, tags)
            VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8)
            """,
            session_id, x, y, z, floor_level, heading_deg, pitch_deg, tags or [],
        )
    return rt_id


# ─────────────────────────────────────────────────────────────────────────────
# flush_path_to_db — MANUAL
# ─────────────────────────────────────────────────────────────────────────────

async def flush_path_to_db(session_id: str = None) -> Tuple[int, int]:
    """
    Commit unflushed temporal_path_log rows to path_nodes + path_edges.
    Uses 3D merge radius and stores distance_3d_m, distance_2d_m, delta_z_m.
    Returns (nodes_upserted, edges_written).
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        log_rows = await conn.fetch(
            """
            SELECT log_id::text, x, y, z, floor_level,
                   heading_deg, pitch_deg, tags, recorded_at
            FROM temporal_path_log
            WHERE ($1::uuid IS NULL OR session_id = $1::uuid)
              AND flushed = FALSE
            ORDER BY recorded_at ASC
            """,
            session_id,
        )
    if not log_rows:
        return 0, 0

    node_db_ids: set   = set()
    edges_written: int = 0
    pos_to_node: Dict[str, str] = {}
    last_db_id: Optional[str]   = None
    last_xyz: Optional[Tuple[float,float,float]] = None

    async with pool.acquire() as conn:
        async with conn.transaction():
            for row in log_rows:
                x, y, z = row["x"], row["y"], row["z"]
                fl = row["floor_level"] or 0

                # 3D nearest within merge radius
                ex = await conn.fetchrow(
                    """
                    SELECT path_node_id::text
                    FROM path_nodes
                    WHERE ST_3DDWithin(
                        position,
                        ST_SetSRID(ST_MakePoint($1,$2,$3),0),
                        $4
                    )
                    ORDER BY ST_3DDistance(position, ST_SetSRID(ST_MakePoint($1,$2,$3),0))
                    LIMIT 1
                    """,
                    x, y, z, MERGE_RADIUS_3D_M,
                )
                if ex:
                    curr_db = ex["path_node_id"]
                    await conn.execute(
                        "UPDATE path_nodes SET visit_count=visit_count+1, visited_at=NOW() WHERE path_node_id=$1::uuid",
                        curr_db,
                    )
                else:
                    nr = await conn.fetchrow(
                        """
                        INSERT INTO path_nodes
                            (position, floor_level, heading_deg, pitch_deg, tags)
                        VALUES
                            (ST_SetSRID(ST_MakePoint($1,$2,$3),0), $4, $5, $6, $7)
                        RETURNING path_node_id::text
                        """,
                        x, y, z, fl,
                        row["heading_deg"], row["pitch_deg"],
                        list(row["tags"] or []),
                    )
                    curr_db = nr["path_node_id"]

                node_db_ids.add(curr_db)
                pos_to_node[row["log_id"]] = curr_db

                if last_db_id and last_db_id != curr_db and last_xyz:
                    px, py, pz = last_xyz
                    d3  = _euclidean_3d(px, py, pz, x, y, z)
                    d2  = _euclidean_2d(px, py, x, y)
                    dz  = z - pz
                    for fr, to, zdelta in [(last_db_id, curr_db, dz), (curr_db, last_db_id, -dz)]:
                        await conn.execute(
                            """
                            INSERT INTO path_edges
                                (from_node_id, to_node_id,
                                 distance_3d_m, distance_2d_m, delta_z_m,
                                 traversal_cost)
                            VALUES ($1::uuid, $2::uuid, $3, $4, $5, $3)
                            ON CONFLICT DO NOTHING
                            """,
                            fr, to, d3, d2, zdelta,
                        )
                    edges_written += 1

                last_db_id = curr_db
                last_xyz   = (x, y, z)

            log_ids = [r["log_id"] for r in log_rows]
            await conn.execute(
                "UPDATE temporal_path_log SET flushed=TRUE WHERE log_id=ANY($1::uuid[])",
                log_ids,
            )

    return len(node_db_ids), edges_written


# ─────────────────────────────────────────────────────────────────────────────
# Entity anchoring
# ─────────────────────────────────────────────────────────────────────────────

async def anchor_entity_to_map(
    entity_id:    str,
    path_node_id: str   = None,
    confidence:   float = 1.0,
) -> str:
    """
    Manually associate an entity with the nearest committed path_node (3D).
    Requires flush_path_to_db() to have run.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        target = path_node_id
        if not target:
            row = await conn.fetchrow(
                """
                SELECT pn.path_node_id::text
                FROM path_nodes pn, entity_nodes e
                WHERE e.node_id = $1::uuid
                ORDER BY ST_3DDistance(pn.position, e.location) ASC
                LIMIT 1
                """,
                entity_id,
            )
            if not row:
                raise ValueError("No committed path_nodes. Call flush_path_to_db() first.")
            target = row["path_node_id"]
        row = await conn.fetchrow(
            """
            INSERT INTO entity_path_anchors (entity_id, path_node_id, confidence)
            VALUES ($1::uuid, $2::uuid, $3)
            ON CONFLICT (entity_id, path_node_id) DO UPDATE
            SET anchored_at=NOW(), confidence=EXCLUDED.confidence
            RETURNING anchor_id::text
            """,
            entity_id, target, confidence,
        )
        return row["anchor_id"]


async def get_entity_path_node(entity_id: str) -> Optional[PathNode]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT pn.path_node_id::text,
                   ST_X(pn.position::geometry) AS x,
                   ST_Y(pn.position::geometry) AS y,
                   ST_Z(pn.position::geometry) AS z,
                   pn.floor_level,
                   pn.heading_deg, pn.pitch_deg,
                   pn.visited_at, pn.visit_count, pn.tags
            FROM entity_path_anchors epa
            JOIN path_nodes pn ON pn.path_node_id = epa.path_node_id
            WHERE epa.entity_id = $1::uuid
            ORDER BY epa.anchored_at DESC LIMIT 1
            """,
            entity_id,
        )
        return _row_to_path_node(row) if row else None


def _row_to_path_node(row) -> PathNode:
    return PathNode(
        path_node_id=row["path_node_id"],
        x=row["x"], y=row["y"], z=row["z"],
        floor_level=row["floor_level"] or 0,
        heading_deg=row["heading_deg"],
        pitch_deg=row.get("pitch_deg"),
        visited_at=row["visited_at"],
        visit_count=row["visit_count"],
        tags=list(row["tags"] or []),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pathfinding
# ─────────────────────────────────────────────────────────────────────────────

async def find_path_between_entities(
    entity_id_a:    str,
    entity_id_b:    str,
    prefer_runtime: bool = True,
) -> Optional[PathResult]:
    if prefer_runtime:
        result = await _runtime_path_for_entities(entity_id_a, entity_id_b)
        if result:
            return result
    node_a = await get_entity_path_node(entity_id_a)
    node_b = await get_entity_path_node(entity_id_b)
    if not node_a or not node_b:
        return None
    if node_a.path_node_id == node_b.path_node_id:
        return PathResult(waypoints=[node_a],
                          total_dist_3d_m=0.0, total_dist_2d_m=0.0,
                          total_ascent_m=0.0, total_descent_m=0.0,
                          total_cost=0.0)
    return await dijkstra_path(node_a.path_node_id, node_b.path_node_id)


async def _runtime_path_for_entities(eid_a: str, eid_b: str) -> Optional[PathResult]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT node_id::text,
                   ST_X(location::geometry) AS x,
                   ST_Y(location::geometry) AS y,
                   ST_Z(location::geometry) AS z
            FROM entity_nodes WHERE node_id=ANY($1::uuid[])
            """,
            [eid_a, eid_b],
        )
    pm = {r["node_id"]: (r["x"], r["y"], r["z"]) for r in rows}
    if eid_a not in pm or eid_b not in pm:
        return None
    rt = _runtime_map
    s = rt._nearest_within_3d(*pm[eid_a], MERGE_RADIUS_3D_M * 10)
    e = rt._nearest_within_3d(*pm[eid_b], MERGE_RADIUS_3D_M * 10)
    if not s or not e:
        return None
    return rt.dijkstra(s, e)


async def dijkstra_path(start_id: str, end_id: str) -> Optional[PathResult]:
    """Dijkstra on committed path_nodes/path_edges using distance_3d_m as cost."""
    import heapq
    pool = await get_pool()
    async with pool.acquire() as conn:
        edges = await conn.fetch(
            """
            SELECT from_node_id::text, to_node_id::text,
                   distance_3d_m, distance_2d_m, delta_z_m, traversal_cost
            FROM path_edges
            """
        )
        nodes = await conn.fetch(
            """
            SELECT path_node_id::text,
                   ST_X(position::geometry) AS x,
                   ST_Y(position::geometry) AS y,
                   ST_Z(position::geometry) AS z,
                   floor_level, heading_deg, pitch_deg,
                   visited_at, visit_count, tags
            FROM path_nodes
            """
        )

    node_map = {r["path_node_id"]: _row_to_path_node(r) for r in nodes}
    adj: Dict[str, List[Tuple[str, float]]] = {n: [] for n in node_map}
    for e in edges:
        adj[e["from_node_id"]].append((e["to_node_id"], e["traversal_cost"]))

    dist = {n: math.inf for n in adj}
    prev: Dict[str, Optional[str]] = {n: None for n in adj}
    dist[start_id] = 0.0
    heap = [(0.0, start_id)]
    while heap:
        cost, u = heapq.heappop(heap)
        if cost > dist[u]: continue
        if u == end_id: break
        for v, w in adj.get(u, []):
            alt = dist[u] + w
            if alt < dist[v]:
                dist[v] = alt; prev[v] = u
                heapq.heappush(heap, (alt, v))
    if math.isinf(dist.get(end_id, math.inf)):
        return None

    path_ids: List[str] = []
    cur: Optional[str] = end_id
    while cur:
        path_ids.append(cur); cur = prev.get(cur)
    path_ids.reverse()
    waypoints = [node_map[p] for p in path_ids if p in node_map]
    return _build_path_result(waypoints, dist[end_id])


async def get_local_map(
    x:           float,
    y:           float,
    z:           float = 0.0,
    radius_m:    float = 15.0,
    floor_level: int   = None,
) -> Tuple[List[PathNode], List[PathEdge]]:
    """
    Return committed path nodes and edges within a 3D sphere.
    Optional floor_level filter to restrict to one floor.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        floor_clause = "AND floor_level = $5" if floor_level is not None else ""
        floor_param  = [floor_level] if floor_level is not None else []

        node_rows = await conn.fetch(
            f"""
            SELECT path_node_id::text,
                   ST_X(position::geometry) AS x,
                   ST_Y(position::geometry) AS y,
                   ST_Z(position::geometry) AS z,
                   floor_level, heading_deg, pitch_deg,
                   visited_at, visit_count, tags
            FROM path_nodes
            WHERE ST_3DDWithin(position, ST_SetSRID(ST_MakePoint($1,$2,$3),0), $4)
            {floor_clause}
            """,
            x, y, z, radius_m, *floor_param,
        )
        node_ids = [r["path_node_id"] for r in node_rows]
        if not node_ids:
            return [], []
        edge_rows = await conn.fetch(
            """
            SELECT path_edge_id::text, from_node_id::text, to_node_id::text,
                   distance_3d_m, distance_2d_m, delta_z_m,
                   traversal_cost, traversal_count
            FROM path_edges
            WHERE from_node_id=ANY($1::uuid[]) AND to_node_id=ANY($1::uuid[])
            """,
            node_ids,
        )
    nodes = [_row_to_path_node(r) for r in node_rows]
    edges = [
        PathEdge(
            path_edge_id=r["path_edge_id"],
            from_node_id=r["from_node_id"],
            to_node_id=r["to_node_id"],
            distance_3d_m=r["distance_3d_m"] or 0.0,
            distance_2d_m=r["distance_2d_m"] or 0.0,
            delta_z_m=r["delta_z_m"] or 0.0,
            traversal_cost=r["traversal_cost"],
            traversal_count=r["traversal_count"],
        )
        for r in edge_rows
    ]
    return nodes, edges


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _euclidean_3d(ax, ay, az, bx, by, bz) -> float:
    return math.sqrt((ax-bx)**2 + (ay-by)**2 + (az-bz)**2)

def _euclidean_2d(ax, ay, bx, by) -> float:
    return math.sqrt((ax-bx)**2 + (ay-by)**2)