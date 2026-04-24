"""
robot_memory/temporal.py
────────────────────────
Temporal interaction log + temporal path log + raw temporal node management.

Key changes in v2
─────────────────
  • add_raw_node()        — store unprocessed sensor data (text, audio transcript,
                            video frame metadata, sensor readings, etc.)
  • get_raw_nodes()       — fetch unprocessed raw nodes for a session
  • mark_raw_processed()  — mark raw nodes as processed (called by consolidator)
  • delete_raw_nodes()    — HARD DELETE processed raw nodes after consolidation
  • delete_interactions() — HARD DELETE temporal_interactions after consolidation
  • flush_path_to_db() flushes the path log but temporal_interactions and
    raw_temporal_nodes are cleaned up by consolidator.flush_and_consolidate()

All positions are 3D: (x, y, z, floor_level).
"""

from __future__ import annotations
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from .db import get_pool


@dataclass
class TemporalRecord:
    interaction_id: str
    session_id:     str
    entity_id:      Optional[str]
    entity_name:    str
    x:              Optional[float]
    y:              Optional[float]
    z:              Optional[float]
    floor_level:    int
    path_log_ref:   Optional[str]
    interaction_ts: datetime
    notes:          Optional[str]


@dataclass
class PathLogRecord:
    log_id:      str
    session_id:  Optional[str]
    x:           float
    y:           float
    z:           float
    floor_level: int
    heading_deg: Optional[float]
    pitch_deg:   Optional[float]
    tags:        List[str]
    recorded_at: datetime
    flushed:     bool


@dataclass
class RawTemporalNode:
    raw_id:             str
    session_id:         str
    data_type:          str
    raw_text:           Optional[str]
    raw_json:           Optional[Dict[str, Any]]
    x:                  Optional[float]
    y:                  Optional[float]
    z:                  Optional[float]
    floor_level:        int
    heading_deg:        Optional[float]
    captured_at:        datetime
    related_entity_id:  Optional[str]
    processed:          bool
    consolidation_id:   Optional[str]


class TemporalSession:
    def __init__(self, session_id: str):
        self.session_id = session_id

    @classmethod
    async def start(cls) -> "TemporalSession":
        sid = str(uuid.uuid4())
        print(f"[temporal] Session started: {sid}")
        return cls(session_id=sid)

    # ── Entity interaction log ────────────────────────────────────────────────

    async def log(
        self,
        entity_id:    Optional[str],
        x:            float,
        y:            float,
        z:            float  = 0.0,
        floor_level:  int    = 0,
        entity_name:  str    = "",
        path_log_ref: str    = None,
        notes:        str    = "",
    ) -> str:
        pool = await get_pool()
        async with pool.acquire() as conn:
            if entity_id and not entity_name:
                row = await conn.fetchrow(
                    "SELECT name FROM entity_nodes WHERE node_id=$1::uuid", entity_id
                )
                entity_name = row["name"] if row else "unknown"
            row = await conn.fetchrow(
                """
                INSERT INTO temporal_interactions
                    (session_id, entity_id, entity_name,
                     location_snap, floor_level, path_log_ref, notes)
                VALUES
                    ($1::uuid, $2::uuid, $3,
                     ST_SetSRID(ST_MakePoint($4,$5,$6),0), $7, $8::uuid, $9)
                RETURNING interaction_id::text
                """,
                self.session_id, entity_id, entity_name,
                x, y, z, floor_level, path_log_ref, notes,
            )
            return row["interaction_id"]

    # ── Raw sensor data ───────────────────────────────────────────────────────

    async def add_raw_node(
        self,
        data_type:          str,
        raw_text:           str   = None,
        raw_json:           dict  = None,
        x:                  float = None,
        y:                  float = None,
        z:                  float = 0.0,
        floor_level:        int   = 0,
        heading_deg:        float = None,
        related_entity_id:  str   = None,
    ) -> str:
        """
        Store unprocessed sensor data for later LLM consolidation.

        data_type examples:
          'text_command'      — a command the robot received in text
          'audio_transcript'  — STT output from microphone
          'video_frame'       — description or metadata of a video frame
          'conversation'      — multi-turn dialogue chunk
          'sensor_reading'    — LIDAR, temperature, battery, etc. in raw_json
          'observation'       — free-form robot observation note
          'image_description' — VLM caption of a camera image

        raw_text: human-readable text payload (transcript, command, caption, etc.)
        raw_json: structured payload (sensor readings, frame metadata, bounding boxes)
        Returns raw_id string.
        """
        pool = await get_pool()
        import json as _json
        json_str = _json.dumps(raw_json) if raw_json else None

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO raw_temporal_nodes
                    (session_id, data_type, raw_text, raw_json,
                     x, y, z, floor_level, heading_deg,
                     related_entity_id)
                VALUES
                    ($1::uuid, $2, $3, $4::jsonb,
                     $5, $6, $7, $8, $9,
                     $10::uuid)
                RETURNING raw_id::text
                """,
                self.session_id, data_type, raw_text, json_str,
                x, y, z, floor_level, heading_deg,
                related_entity_id,
            )
            return row["raw_id"]

    async def get_raw_nodes(
        self,
        unprocessed_only: bool = True,
        limit:            int  = 500,
        data_types:       List[str] = None,
    ) -> List[RawTemporalNode]:
        """Fetch raw temporal nodes for this session."""
        pool = await get_pool()
        type_clause = ""
        params: list = [self.session_id, limit]
        if unprocessed_only:
            params.insert(1, False)  # processed = False
            type_clause_proc = "AND processed = $2"
            # renumber
            type_clause_proc = "AND processed = FALSE"
        else:
            type_clause_proc = ""
        type_clause_types = ""
        if data_types:
            params.append(data_types)
            type_clause_types = f"AND data_type = ANY(${len(params)})"

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT raw_id::text, session_id::text, data_type,
                       raw_text, raw_json, x, y, z, floor_level, heading_deg,
                       captured_at, related_entity_id::text,
                       processed, consolidation_id::text
                FROM raw_temporal_nodes
                WHERE session_id = $1::uuid
                  {type_clause_proc}
                  {type_clause_types}
                ORDER BY captured_at ASC
                LIMIT $2
                """,
                self.session_id, limit,
            )
        return [_row_to_raw_node(r) for r in rows]

    async def mark_raw_processed(
        self,
        raw_ids:         List[str],
        consolidation_id: str,
    ) -> None:
        """Mark raw nodes as processed (set processed=TRUE + consolidation_id)."""
        if not raw_ids:
            return
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE raw_temporal_nodes
                SET processed = TRUE, consolidation_id = $1::uuid
                WHERE raw_id = ANY($2::uuid[])
                """,
                consolidation_id, raw_ids,
            )

    async def delete_raw_nodes(
        self,
        consolidation_id: str = None,
        processed_only:   bool = True,
    ) -> int:
        """
        HARD DELETE raw temporal nodes after successful consolidation.
        If consolidation_id given, only deletes nodes from that run.
        Returns count of deleted rows.
        """
        pool = await get_pool()
        async with pool.acquire() as conn:
            if consolidation_id:
                result = await conn.execute(
                    "DELETE FROM raw_temporal_nodes WHERE consolidation_id=$1::uuid",
                    consolidation_id,
                )
            elif processed_only:
                result = await conn.execute(
                    "DELETE FROM raw_temporal_nodes WHERE session_id=$1::uuid AND processed=TRUE",
                    self.session_id,
                )
            else:
                result = await conn.execute(
                    "DELETE FROM raw_temporal_nodes WHERE session_id=$1::uuid",
                    self.session_id,
                )
        # asyncpg returns "DELETE N"
        try:
            return int(str(result).split()[-1])
        except Exception:
            return 0

    async def delete_interactions(self) -> int:
        """
        HARD DELETE temporal_interactions for this session after consolidation.
        The information they contained has been absorbed into the knowledge graph.
        Returns count of deleted rows.
        """
        pool = await get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM temporal_interactions WHERE session_id=$1::uuid",
                self.session_id,
            )
        try:
            return int(str(result).split()[-1])
        except Exception:
            return 0

    # ── Path log ──────────────────────────────────────────────────────────────

    async def latest_path_log_id(self) -> Optional[str]:
        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT log_id::text FROM temporal_path_log "
                "WHERE session_id=$1::uuid ORDER BY recorded_at DESC LIMIT 1",
                self.session_id,
            )
            return row["log_id"] if row else None

    async def get_records(self, limit: int = 200) -> List[TemporalRecord]:
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT interaction_id::text, session_id::text,
                       entity_id::text, entity_name,
                       ST_X(location_snap::geometry) AS x,
                       ST_Y(location_snap::geometry) AS y,
                       ST_Z(location_snap::geometry) AS z,
                       floor_level,
                       path_log_ref::text, interaction_ts, notes
                FROM temporal_interactions
                WHERE session_id=$1::uuid
                ORDER BY interaction_ts ASC LIMIT $2
                """,
                self.session_id, limit,
            )
            return [_row_to_record(r) for r in rows]

    async def get_path_log(self, limit: int = 1000, unflushed: bool = False) -> List[PathLogRecord]:
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT log_id::text, session_id::text, x, y, z, floor_level,
                       heading_deg, pitch_deg, tags, recorded_at, flushed
                FROM temporal_path_log
                WHERE session_id=$1::uuid
                  AND ($2 = FALSE OR flushed = FALSE)
                ORDER BY recorded_at ASC LIMIT $3
                """,
                self.session_id, unflushed, limit,
            )
            return [_row_to_path_log(r) for r in rows]

    async def path_log_status(self) -> dict:
        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT COUNT(*) AS total,
                       COUNT(*) FILTER (WHERE flushed)     AS flushed,
                       COUNT(*) FILTER (WHERE NOT flushed) AS pending
                FROM temporal_path_log WHERE session_id=$1::uuid
                """,
                self.session_id,
            )
            return dict(row)

    async def raw_node_status(self) -> dict:
        """Count of raw temporal nodes for this session by processed state."""
        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT COUNT(*) AS total,
                       COUNT(*) FILTER (WHERE processed)     AS processed,
                       COUNT(*) FILTER (WHERE NOT processed) AS pending
                FROM raw_temporal_nodes WHERE session_id=$1::uuid
                """,
                self.session_id,
            )
            return dict(row)

    async def dump_summary(self) -> None:
        records    = await self.get_records()
        log_status = await self.path_log_status()
        try:
            raw_status = await self.raw_node_status()
        except Exception:
            raw_status = {"total": "?", "pending": "?", "processed": "?"}

        print(f"\n{'━'*65}")
        print(f"  TEMPORAL SESSION  |  {self.session_id[:8]}…")
        print(f"{'━'*65}")
        print(f"  Path log   : {log_status['total']} positions  "
              f"({log_status['pending']} pending flush / {log_status['flushed']} committed)")
        print(f"  Raw nodes  : {raw_status['total']} total  "
              f"({raw_status['pending']} pending LLM / {raw_status['processed']} processed)")
        print(f"  Interactions: {len(records)}")
        print(f"{'─'*65}")
        for r in records:
            ts  = r.interaction_ts.strftime("%H:%M:%S")
            loc = f"({r.x:.1f},{r.y:.1f},{r.z:.1f}) fl={r.floor_level}"
            print(f"  {ts}  {r.entity_name:<22}  @ {loc}  — {r.notes or ''}")
        print(f"{'━'*65}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

async def sessions_for_entity(entity_id: str, limit: int = 10) -> List[str]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT DISTINCT session_id::text FROM temporal_interactions "
            "WHERE entity_id=$1::uuid ORDER BY 1 DESC LIMIT $2",
            entity_id, limit,
        )
        return [r["session_id"] for r in rows]


async def get_all_pending_raw_nodes(
    session_id: str = None,
    limit:      int = 1000,
) -> List[RawTemporalNode]:
    """Fetch ALL unprocessed raw nodes, optionally filtered by session."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        if session_id:
            rows = await conn.fetch(
                """
                SELECT raw_id::text, session_id::text, data_type,
                       raw_text, raw_json, x, y, z, floor_level, heading_deg,
                       captured_at, related_entity_id::text,
                       processed, consolidation_id::text
                FROM raw_temporal_nodes
                WHERE session_id=$1::uuid AND processed=FALSE
                ORDER BY captured_at ASC LIMIT $2
                """,
                session_id, limit,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT raw_id::text, session_id::text, data_type,
                       raw_text, raw_json, x, y, z, floor_level, heading_deg,
                       captured_at, related_entity_id::text,
                       processed, consolidation_id::text
                FROM raw_temporal_nodes
                WHERE processed=FALSE
                ORDER BY captured_at ASC LIMIT $2
                """,
                limit,
            )
    return [_row_to_raw_node(r) for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# Row parsers
# ─────────────────────────────────────────────────────────────────────────────

def _row_to_record(row) -> TemporalRecord:
    return TemporalRecord(
        interaction_id=row["interaction_id"],
        session_id=row["session_id"],
        entity_id=row["entity_id"],
        entity_name=row["entity_name"] or "",
        x=row["x"], y=row["y"], z=row["z"],
        floor_level=row["floor_level"] or 0,
        path_log_ref=row["path_log_ref"],
        interaction_ts=row["interaction_ts"],
        notes=row["notes"],
    )


def _row_to_path_log(row) -> PathLogRecord:
    return PathLogRecord(
        log_id=row["log_id"], session_id=row["session_id"],
        x=row["x"], y=row["y"], z=row["z"],
        floor_level=row["floor_level"] or 0,
        heading_deg=row["heading_deg"], pitch_deg=row.get("pitch_deg"),
        tags=list(row["tags"] or []),
        recorded_at=row["recorded_at"], flushed=row["flushed"],
    )


def _row_to_raw_node(row) -> RawTemporalNode:
    import json as _json
    raw_json = None
    if row["raw_json"]:
        try:
            raw_json = _json.loads(row["raw_json"]) if isinstance(row["raw_json"], str) else dict(row["raw_json"])
        except Exception:
            raw_json = None

    return RawTemporalNode(
        raw_id=row["raw_id"],
        session_id=row["session_id"],
        data_type=row["data_type"],
        raw_text=row["raw_text"],
        raw_json=raw_json,
        x=row["x"], y=row["y"], z=row["z"],
        floor_level=row["floor_level"] or 0,
        heading_deg=row["heading_deg"],
        captured_at=row["captured_at"],
        related_entity_id=row["related_entity_id"],
        processed=row["processed"],
        consolidation_id=row["consolidation_id"],
    )