"""
robot_memory/consolidator.py  (v4)
───────────────────────────────────
Episode-first consolidation pipeline:

  PHASE 1 — UNDERSTAND THE EPISODE
    Read ALL raw nodes for the session at once. Make a single LLM call that
    synthesises them into a structured episode summary: what happened, what
    objects/people were present, what relationships were observed. This is the
    robot "thinking about" what it just experienced — like writing a diary entry.

  PHASE 2 — RECONCILE WITH EXISTING MEMORY
    Load full state of every entity mentioned in the episode summary. Compare
    what is already known vs what is genuinely new. Only write the delta.

  PHASE 3 — WRITE MINIMAL ENTITY GRAPH
    Create / update entities. Write at most one new info_node per entity per
    session. Prune old info_nodes beyond the keep limit. Promote media.
    Write edges only where a real spatial/functional relationship was observed.

  PHASE 4 — CLEAN UP
    Hard-delete all raw_temporal_nodes (ALL types, ALL sessions' old ones).
    Hard-delete flushed temporal_path_log rows older than 24h.
    Delete temporal_interactions for this session.

Key rules enforced in the system prompt:
  * Never create a "Robot", "Self", "Me", "I", "Assistant", or "System" entity.
    The robot IS the memory system — it has no self-node.
  * Only create edges where a clear spatial or functional relationship exists
    (proximity, containment, dependency). Do NOT create edges just because two
    things were mentioned in the same sentence.
  * action=skip if nothing new was observed about an entity.
  * new_info must be a single factual sentence not already stored.
"""

from __future__ import annotations
import asyncio, json, logging, uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .db import get_pool, get_llm_client, get_llm_model, log
from .graph import (
    upsert_entity, add_info_node, upsert_edge,
    get_entity_by_name, get_info_meta, fetch_info_full,
    get_edges_for_entity, update_entity_location,
)
from .temporal import TemporalSession, RawTemporalNode, get_all_pending_raw_nodes
from .think import deep_think

import os
MAX_INFO_NODES_KEEP = int(os.getenv("ROBOT_MAX_INFO_NODES", "3"))
MAX_EDGE_CREATES    = int(os.getenv("ROBOT_MAX_EDGES_PER_SESSION", "6"))

# ---------------------------------------------------------------------------
# SELF-ENTITY BLACKLIST — names the LLM must never create as entity nodes
# ---------------------------------------------------------------------------
_SELF_NAMES = frozenset({
    "robot", "self", "me", "i", "assistant", "system", "agent",
    "the robot", "myself", "camera", "microphone", "the system",
    "memory system", "robot memory",
})

def _is_self_entity(name: str) -> bool:
    return name.strip().lower() in _SELF_NAMES


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 SYSTEM PROMPT  — understand the episode
# ─────────────────────────────────────────────────────────────────────────────

EPISODE_PROMPT = """
You are the memory core of an autonomous robot. You have just finished a
navigation session and need to understand what you experienced.

You will receive raw sensor data: audio transcripts, video frame captions,
and ambient conversation snippets captured during the session.

Your task: synthesise this into a structured episode report — what the robot
observed, what objects and people were present, and what spatial relationships
existed between them.

OUTPUT: a single valid JSON object. No markdown. No preamble.

Schema:
{
  "episode_summary": "<2-4 sentence narrative of what happened in this session>",
  "observations": [
    {
      "entity_name": "<specific name of the thing observed — NOT 'Robot', 'Self', 'Me', 'I', 'Camera', 'Microphone'>",
      "entity_type": "object|place|person|vehicle|sensor|concept",
      "description": "<what was observed about this entity — 1-2 sentences of fact>",
      "position_note": "<where it was relative to the robot: 'nearby', 'on desk', 'to the left', etc.>",
      "x": <float metres East from robot origin, or null>,
      "y": <float metres North from robot origin, or null>,
      "z": <float metres Up, or null>,
      "floor_level": <int, default 0>,
      "tags": ["tag1"],
      "weight": <0.5-2.0, higher = more prominent/important>,
      "has_image": <true if a camera image exists for this entity>,
      "has_audio": <true if audio was recorded related to this entity>
    }
  ],
  "relationships": [
    {
      "entity_1": "<name>",
      "entity_2": "<name>",
      "rel_type": "near|contains|on_top_of|blocks|connects|monitors|charges|services",
      "description": "<why this relationship exists — one sentence>",
      "confidence": "high|medium|low"
    }
  ]
}

STRICT RULES:
1. NEVER create observations for: Robot, Self, Me, I, Assistant, Camera, Microphone,
   System, Memory System, or any variation. The robot is not an entity in its own memory.
2. Only include entities that are REAL, PERSISTENT, PHYSICAL things in the environment.
   Do NOT include: sounds, movements, lighting conditions, abstract concepts.
3. Relationships: ONLY create one where the robot clearly observed a direct spatial
   or functional connection. Not just co-occurrence in the same recording. Maximum 4.
4. If nothing real was observed, return:
   {"episode_summary":"No notable entities observed.","observations":[],"relationships":[]}
5. entity_name must be a specific proper noun or descriptive name (e.g. "Blue Coffee Mug",
   "Oak Desk", "Person in Red Shirt") — not generic words like "object", "thing", "item".
"""


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 SYSTEM PROMPT  — reconcile with existing memory
# ─────────────────────────────────────────────────────────────────────────────

RECONCILE_PROMPT = """
You are the memory manager of an autonomous robot. You have just processed
a navigation session and produced an episode report. Now you must decide
what to write into long-term memory.

You will receive:
  A) The episode report (what was just observed)
  B) The robot's EXISTING KNOWLEDGE about each entity mentioned

Your task: for each observed entity, decide exactly what to write.

OUTPUT: a single valid JSON object. No markdown. No preamble.

Schema:
{
  "entities": [
    {
      "name": "<exact name from episode report>",
      "action": "create" | "update" | "skip",
      "summary": "<concise 1-sentence description — only if creating or changing summary>",
      "entity_type": "object|place|person|vehicle|sensor|concept",
      "tags": ["tag"],
      "x": <float|null>, "y": <float|null>, "z": <float|null>,
      "floor_level": <int>,
      "new_info": "<single NEW factual sentence not already in existing knowledge — null if nothing new>",
      "crucial_words": ["word1","word2"],
      "weight": <0.5-2.0>,
      "has_audio": <true|false>,
      "has_image": <true|false>
    }
  ],
  "edges": [
    {
      "entity_1": "<name>",
      "entity_2": "<name>",
      "rel_type": "near|contains|on_top_of|blocks|connects|monitors|charges|services",
      "rel_name": "<3-5 word label>",
      "directed": false,
      "confidence": "high|medium"
    }
  ]
}

RULES:
1. action="skip"   — everything about this entity is already accurately stored.
2. action="update" — entity exists; new_info adds one new fact not already known.
3. action="create" — entity is not in existing knowledge at all.
4. new_info        — must be ONE sentence of pure fact. Null if nothing new.
                     Do NOT repeat what is already in EXISTING KNOWLEDGE.
5. NEVER create entities named: Robot, Self, Me, I, Camera, Microphone, System, Assistant.
6. Only write edges with confidence=high or confidence=medium.
   Maximum 4 edges total. Skip low-confidence relationships entirely.
7. crucial_words: 2-4 retrieval keywords, lowercase.
8. If action=skip for all entities and no edges, return:
   {"entities":[],"edges":[]}
"""


# ─────────────────────────────────────────────────────────────────────────────
# DTO
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConsolidationResult:
    consolidation_id:    str
    session_id:          Optional[str]
    started_at:          datetime
    finished_at:         Optional[datetime]
    raw_nodes_processed: int
    entities_created:    int
    entities_updated:    int
    edges_created:       int
    info_nodes_created:  int
    llm_calls:           int
    status:              str
    error_msg:           Optional[str]
    summary_text:        str
    llm_model:           str


# ─────────────────────────────────────────────────────────────────────────────
# Format raw nodes for the episode LLM prompt
# ─────────────────────────────────────────────────────────────────────────────

def _format_episode_input(raw_nodes: List[RawTemporalNode]) -> str:
    """
    Produce a clean chronological transcript of the session's sensor data.
    Strips all b64 blobs — the LLM only needs text.
    Only includes audio_transcript, video_frame, conversation nodes.
    """
    lines = []
    for r in raw_nodes:
        ts  = r.captured_at.strftime("%H:%M:%S")
        pos = f"({r.x:.1f},{r.y:.1f},{r.z:.1f})" if r.x is not None else "(?,?,?)"

        rj = dict(r.raw_json) if r.raw_json else {}
        has_img = bool(rj.get("jpeg_b64"))
        has_aud = bool(rj.get("wav_b64"))
        # Strip blobs
        rj_clean = {k: v for k, v in rj.items()
                    if k not in ("jpeg_b64", "wav_b64") and v is not None}

        media = ""
        if has_img: media += " [camera_image_attached]"
        if has_aud: media += " [audio_recording_attached]"

        text = (r.raw_text or "").strip()
        if not text and not rj_clean:
            continue   # skip empty nodes

        line = f"[{ts}] {r.data_type.upper()}{media} @ {pos} fl={r.floor_level or 0}"
        if text:
            line += f"\n  {text}"
        if rj_clean:
            line += f"\n  meta: {json.dumps(rj_clean)[:80]}"
        lines.append(line)
    return "\n\n".join(lines) if lines else "(no sensor data)"


# ─────────────────────────────────────────────────────────────────────────────
# Load full entity state for reconciliation context
# ─────────────────────────────────────────────────────────────────────────────

async def _load_entity_full_state(name: str) -> str:
    ent = await get_entity_by_name(name)
    if not ent:
        return f"[{name}]: NOT IN MEMORY\n"
    lines = [
        f"[{name}]  type={ent.entity_type}  "
        f"pos=({ent.x},{ent.y},{ent.z})  fl={ent.floor_level}  w={ent.weight:.2f}",
        f"  summary: {ent.summary or '—'}",
        f"  tags: {', '.join(ent.tags) if ent.tags else '—'}",
    ]
    metas = await get_info_meta(ent.node_id)
    for m in metas[:MAX_INFO_NODES_KEEP]:
        fi = await fetch_info_full(m.node_id)
        if fi and fi.get("full_data"):
            kw = ", ".join((fi.get("crucial_words") or [])[:4])
            lines.append(f"  known: [{kw}] {fi['full_data'][:180]}")
    edges = await get_edges_for_entity(ent.node_id)
    for ed in edges[:3]:
        oid = ed.node_id_2 if ed.node_id_1 == ent.node_id else ed.node_id_1
        lines.append(f"  edge: {ed.rel_type} -> {oid[:8]}…")
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# LLM call helper — parse JSON safely
# ─────────────────────────────────────────────────────────────────────────────

async def _llm_json(system: str, user: str, client, model: str,
                    max_tokens: int = 2000) -> dict:
    resp = await client.chat.completions.create(
        model=model, max_tokens=max_tokens, temperature=0.1,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    raw = resp.choices[0].message.content.strip()
    # Strip markdown fences
    if "```" in raw:
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else parts[0]
        if raw.lower().startswith("json"):
            raw = raw[4:]
    raw = raw.strip().strip("`")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        log.warning("LLM JSON parse failed: %s\n%.300s", e, raw)
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Dedup: is new_info already stored?
# ─────────────────────────────────────────────────────────────────────────────

async def _is_duplicate_info(entity_id: str, new_text: str) -> bool:
    if not new_text or len(new_text.strip()) < 8:
        return True
    new_words = set(new_text.lower().split())
    if not new_words:
        return True
    metas = await get_info_meta(entity_id)
    for m in metas:
        fi = await fetch_info_full(m.node_id)
        if fi and fi.get("full_data"):
            existing_words = set(fi["full_data"].lower().split())
            overlap = len(new_words & existing_words) / max(len(new_words), 1)
            if overlap > 0.65:
                return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Prune excess info_nodes
# ─────────────────────────────────────────────────────────────────────────────

async def _prune_info_nodes(entity_id: str) -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT node_id::text FROM info_nodes "
            "WHERE entity_id=$1::uuid ORDER BY weight DESC OFFSET $2",
            entity_id, MAX_INFO_NODES_KEEP,
        )
        if rows:
            await conn.execute(
                "DELETE FROM info_nodes WHERE node_id=ANY($1::uuid[])",
                [r["node_id"] for r in rows],
            )


# ─────────────────────────────────────────────────────────────────────────────
# Promote media (full data URI into entity_nodes)
# ─────────────────────────────────────────────────────────────────────────────

async def _promote_media(entity_id: str, raw_nodes: List[RawTemporalNode]) -> None:
    """
    Copy jpeg_b64 → image_ptrs (as data URI) and wav_b64 → audio_ptr.
    Must run BEFORE raw nodes are deleted.
    """
    pool = await get_pool()
    best_img: Optional[str] = None
    best_aud: Optional[str] = None

    for r in raw_nodes:
        rj = r.raw_json or {}
        if not best_img and rj.get("jpeg_b64"):
            best_img = f"data:image/jpeg;base64,{rj['jpeg_b64']}"
        if not best_aud and rj.get("wav_b64"):
            best_aud = f"data:audio/wav;base64,{rj['wav_b64']}"
        if best_img and best_aud:
            break

    if not best_img and not best_aud:
        return

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT image_ptrs, audio_ptr FROM entity_nodes WHERE node_id=$1::uuid",
            entity_id,
        )
        if not row:
            return
        existing_ptrs = list(row["image_ptrs"] or [])
        existing_aud  = row["audio_ptr"]

        updates: List[str] = []
        params:  List[Any] = [entity_id]

        if best_img:
            # Deduplicate by fingerprint (first 60 chars)
            fp = best_img[:60]
            if not any(p[:60] == fp for p in existing_ptrs):
                params.append(best_img)
                updates.append(
                    f"image_ptrs = array_append(COALESCE(image_ptrs,ARRAY[]::text[]), ${len(params)})"
                )

        if best_aud and not existing_aud:
            params.append(best_aud)
            updates.append(f"audio_ptr = ${len(params)}")

        if updates:
            await conn.execute(
                f"UPDATE entity_nodes SET {', '.join(updates)}, updated_at=NOW() "
                f"WHERE node_id=$1::uuid",
                *params,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Write reconciled knowledge to entity graph
# ─────────────────────────────────────────────────────────────────────────────

async def _write_reconciled(
    reconciled:  dict,
    raw_batch:   List[RawTemporalNode],
    stats:       Dict[str, int],
) -> None:
    entity_name_to_id: Dict[str, str] = {}
    media_nodes = [r for r in raw_batch
                   if r.raw_json and (r.raw_json.get("jpeg_b64") or r.raw_json.get("wav_b64"))]
    edges_written = 0

    for ent_data in reconciled.get("entities", []):
        name   = (ent_data.get("name") or "").strip()
        action = (ent_data.get("action") or "skip").lower()

        if not name or action == "skip" or _is_self_entity(name):
            continue

        x  = ent_data.get("x")  or 0.0
        y  = ent_data.get("y")  or 0.0
        z  = ent_data.get("z")  or 0.0
        fl = int(ent_data.get("floor_level") or 0)
        tags   = ent_data.get("tags") or []
        weight = float(ent_data.get("weight") or 1.0)
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",")]

        existing = await get_entity_by_name(name)

        if action == "update" and existing:
            eid = existing.node_id
            if x != 0.0 or y != 0.0:
                await update_entity_location(eid, x, y, z, fl)
            stats["entities_updated"] = stats.get("entities_updated", 0) + 1

        elif action in ("create", "update"):   # create even if action=update but entity missing
            eid = await upsert_entity(
                name=name,
                summary=ent_data.get("summary") or "",
                entity_type=ent_data.get("entity_type") or "object",
                x=x, y=y, z=z, floor_level=fl,
                tags=tags, weight=weight,
            )
            stats["entities_created"] = stats.get("entities_created", 0) + 1
        else:
            continue

        entity_name_to_id[name] = eid

        # Write info_node only if new_info is genuinely new
        new_info = (ent_data.get("new_info") or "").strip()
        crucial  = ent_data.get("crucial_words") or []
        if isinstance(crucial, str):
            crucial = [w.strip() for w in crucial.split(",")]

        if new_info and not await _is_duplicate_info(eid, new_info):
            await add_info_node(
                entity_id=eid,
                crucial_words=crucial,
                full_data=new_info,
                weight=weight,
            )
            stats["info_nodes_created"] = stats.get("info_nodes_created", 0) + 1
            await _prune_info_nodes(eid)

        # Promote media
        if (ent_data.get("has_audio") or ent_data.get("has_image")) and media_nodes:
            near = [r for r in media_nodes
                    if r.x is not None and abs((r.x or 0) - x) < 8
                    and abs((r.y or 0) - y) < 8] or media_nodes[:1]
            await _promote_media(eid, near)

    # Write edges (max MAX_EDGE_CREATES, confidence=high|medium only)
    for ed in reconciled.get("edges", []):
        if edges_written >= MAX_EDGE_CREATES:
            break
        n1 = (ed.get("entity_1") or "").strip()
        n2 = (ed.get("entity_2") or "").strip()
        conf = (ed.get("confidence") or "low").lower()
        if not n1 or not n2 or n1 == n2: continue
        if conf == "low": continue
        if _is_self_entity(n1) or _is_self_entity(n2): continue

        eid1 = entity_name_to_id.get(n1) or (
            e.node_id if (e := await get_entity_by_name(n1)) else None)
        eid2 = entity_name_to_id.get(n2) or (
            e.node_id if (e := await get_entity_by_name(n2)) else None)

        if not eid1 or not eid2:
            continue

        await upsert_edge(
            node_id_1=eid1, node_id_2=eid2,
            rel_type=ed.get("rel_type") or "near",
            rel_name=ed.get("rel_name") or "",
            directed=bool(ed.get("directed", False)),
        )
        stats["edges_created"] = stats.get("edges_created", 0) + 1
        edges_written += 1


# ─────────────────────────────────────────────────────────────────────────────
# Global context (compact — just names, positions, summary)
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_global_context(max_entities: int = 50) -> str:
    dctx = await deep_think(include_full_embeddings=False)
    if not dctx.all_entities:
        return "EXISTING MEMORY: empty.\n"
    lines = ["EXISTING MEMORY OVERVIEW:"]
    for e in dctx.all_entities[:max_entities]:
        if _is_self_entity(e.name):
            continue
        pos = f"({e.x:.1f},{e.y:.1f},{e.z:.1f})" if e.x is not None else "(?,?,?)"
        kw  = ", ".join(e.top_words[:3]) if e.top_words else ""
        lines.append(
            f"  [{e.entity_type or '?'}] {e.name} @ {pos} fl={e.floor_level} — {e.summary or '—'} [{kw}]"
        )
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# Cleanup helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _delete_raw_nodes(raw_ids: List[str]) -> int:
    """Hard-delete specific raw nodes by ID."""
    if not raw_ids:
        return 0
    pool = await get_pool()
    async with pool.acquire() as conn:
        res = await conn.execute(
            "DELETE FROM raw_temporal_nodes WHERE raw_id=ANY($1::uuid[])", raw_ids
        )
    try:
        return int(str(res).split()[-1])
    except Exception:
        return len(raw_ids)


async def _delete_all_session_raw_nodes(session_id: str) -> int:
    """Delete ALL raw nodes for this session (including observation-only)."""
    if not session_id:
        return 0
    pool = await get_pool()
    async with pool.acquire() as conn:
        res = await conn.execute(
            "DELETE FROM raw_temporal_nodes WHERE session_id=$1::uuid", session_id
        )
    try:
        return int(str(res).split()[-1])
    except Exception:
        return 0


async def _cleanup_path_log(session_id: str, older_than_hours: int = 24) -> int:
    """
    Delete flushed temporal_path_log rows older than N hours to prevent unbounded growth.
    Keeps recent unflushed rows intact.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        res = await conn.execute(
            """
            DELETE FROM temporal_path_log
            WHERE session_id = $1::uuid
              AND flushed = TRUE
              AND recorded_at < NOW() - INTERVAL '1 hour' * $2
            """,
            session_id, older_than_hours,
        )
    try:
        return int(str(res).split()[-1])
    except Exception:
        return 0


async def _delete_orphan_self_entities() -> int:
    """Remove any existing self-entity nodes (Robot, Self, etc.) from the graph."""
    pool = await get_pool()
    deleted = 0
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT node_id::text, name FROM entity_nodes"
        )
        for row in rows:
            if _is_self_entity(row["name"]):
                await conn.execute(
                    "DELETE FROM entity_nodes WHERE node_id=$1::uuid", row["node_id"]
                )
                deleted += 1
                log.info("Deleted self-entity: %s", row["name"])
    return deleted


# ─────────────────────────────────────────────────────────────────────────────
# Consolidation log helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _create_log(sid: Optional[str], model: str) -> str:
    cid = str(uuid.uuid4())
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO consolidation_log (consolidation_id,session_id,llm_model,status) "
            "VALUES ($1::uuid,$2::uuid,$3,'running')",
            cid, sid, model,
        )
    return cid


async def _finish_log(cid: str, stats: dict, calls: int,
                      status: str, error: str = None, summary: str = "") -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE consolidation_log SET
                finished_at=NOW(), raw_nodes_processed=$2,
                entities_created=$3, entities_updated=$4,
                edges_created=$5, info_nodes_created=$6,
                llm_calls=$7, status=$8, error_msg=$9, summary_text=$10
            WHERE consolidation_id=$1::uuid
            """,
            cid,
            stats.get("raw_nodes_processed", 0),
            stats.get("entities_created", 0),
            stats.get("entities_updated", 0),
            stats.get("edges_created", 0),
            stats.get("info_nodes_created", 0),
            calls, status, error, (summary or "")[:1000],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

async def flush_and_consolidate(
    session_id: str             = None,
    session:    TemporalSession = None,
    batch_size: int             = 0,    # ignored — episode-first uses all nodes
    verbose:    bool            = True,
) -> ConsolidationResult:
    """
    Episode-first consolidation:
      Phase 1: Understand the full episode (1 LLM call for all nodes)
      Phase 2: Reconcile with existing memory (1 LLM call)
      Phase 3: Write minimal entity graph delta
      Phase 4: Clean up all raw/path/interaction data
    """
    _sid  = session_id or (session.session_id if session else None)
    model = get_llm_model()
    cid   = await _create_log(_sid, model)
    t0    = datetime.utcnow()
    stats = {k: 0 for k in ("raw_nodes_processed", "entities_created",
                              "entities_updated", "edges_created", "info_nodes_created")}
    calls = 0
    err   = None

    try:
        client = get_llm_client()

        # ── Step 0: Remove any stale self-entities from previous sessions ──────
        removed = await _delete_orphan_self_entities()
        if removed and verbose:
            print(f"[consolidator] Removed {removed} self-entity node(s) from graph.")

        # ── Step 1: Fetch ALL useful raw nodes for this session ───────────────
        raw_all = await get_all_pending_raw_nodes(session_id=_sid, limit=5000)

        # Filter to only sensor content — observations are position data, not LLM-worthy
        useful = [
            r for r in raw_all
            if r.data_type in ("audio_transcript", "video_frame", "conversation",
                               "image_description", "text_command")
            and (r.raw_text or r.raw_json)
        ]

        if not useful:
            if verbose:
                print(f"[consolidator] No sensor content to consolidate "
                      f"({len(raw_all)} observation-only nodes).")
            # Still clean up all raw nodes for this session
            deleted = await _delete_all_session_raw_nodes(_sid)
            if _sid:
                await _cleanup_path_log(_sid)
            if session:
                await session.delete_interactions()
            if verbose and deleted:
                print(f"[consolidator] Cleaned up {deleted} raw nodes.")
            await _finish_log(cid, stats, 0, "done", summary="No sensor content.")
            return ConsolidationResult(
                consolidation_id=cid, session_id=_sid,
                started_at=t0, finished_at=datetime.utcnow(),
                raw_nodes_processed=0, entities_created=0, entities_updated=0,
                edges_created=0, info_nodes_created=0, llm_calls=0,
                status="done", error_msg=None,
                summary_text="No sensor content to consolidate.", llm_model=model,
            )

        if verbose:
            print(f"[consolidator] {len(useful)} sensor nodes to process "
                  f"(skipped {len(raw_all) - len(useful)} observation-only)  "
                  f"model={model}")

        # ── Phase 1: Understand the episode ──────────────────────────────────
        if verbose: print("[consolidator] Phase 1 — understanding episode…")
        episode_input = _format_episode_input(useful)
        global_ctx    = await fetch_global_context()

        episode_result = await _llm_json(
            EPISODE_PROMPT,
            f"EXISTING MEMORY:\n{global_ctx}\n\n"
            f"SESSION SENSOR DATA:\n{episode_input}\n\n"
            f"Produce the episode report. Output ONLY valid JSON.",
            client, model, max_tokens=2000,
        )
        calls += 1

        if verbose:
            summary_text = episode_result.get("episode_summary", "")
            obs_count    = len(episode_result.get("observations", []))
            rel_count    = len(episode_result.get("relationships", []))
            print(f"  Episode: {summary_text[:100]}")
            print(f"  Observations: {obs_count}  Relationships: {rel_count}")

        observations = episode_result.get("observations", [])
        if not observations:
            if verbose: print("[consolidator] No observations extracted — cleaning up.")
            await _delete_all_session_raw_nodes(_sid)
            if _sid: await _cleanup_path_log(_sid)
            if session: await session.delete_interactions()
            await _finish_log(cid, stats, calls, "done",
                              summary=episode_result.get("episode_summary", ""))
            return ConsolidationResult(
                consolidation_id=cid, session_id=_sid,
                started_at=t0, finished_at=datetime.utcnow(),
                raw_nodes_processed=len(useful),
                entities_created=0, entities_updated=0,
                edges_created=0, info_nodes_created=0,
                llm_calls=calls, status="done", error_msg=None,
                summary_text=episode_result.get("episode_summary", ""),
                llm_model=model,
            )

        # ── Phase 2: Reconcile with existing memory ──────────────────────────
        if verbose: print("[consolidator] Phase 2 — reconciling with existing memory…")

        # Load full state for every entity mentioned in episode
        entity_names = [
            o["entity_name"] for o in observations
            if o.get("entity_name") and not _is_self_entity(o.get("entity_name", ""))
        ]
        existing_block = ""
        for nm in entity_names[:12]:
            existing_block += await _load_entity_full_state(nm)
        if not existing_block:
            existing_block = "(No matching entities in current memory)\n"

        episode_json = json.dumps(episode_result, indent=2)[:3000]

        reconciled = await _llm_json(
            RECONCILE_PROMPT,
            f"EXISTING KNOWLEDGE:\n{existing_block}\n\n"
            f"EPISODE REPORT:\n{episode_json}\n\n"
            f"Decide what to write into long-term memory. Output ONLY valid JSON.",
            client, model, max_tokens=2000,
        )
        calls += 1

        if verbose:
            ent_count  = len([e for e in reconciled.get("entities", [])
                               if e.get("action") != "skip"])
            edge_count = len(reconciled.get("edges", []))
            print(f"  Writing: {ent_count} entities  {edge_count} edges")

        # ── Phase 3: Write to entity graph ────────────────────────────────────
        if verbose: print("[consolidator] Phase 3 — writing to entity graph…")
        await _write_reconciled(reconciled, useful, stats)
        stats["raw_nodes_processed"] = len(useful)

        # ── Phase 4: Clean up all temporal data ───────────────────────────────
        if verbose: print("[consolidator] Phase 4 — cleaning up…")

        # Delete ALL raw nodes for this session (observations + sensor data)
        deleted_raw = await _delete_all_session_raw_nodes(_sid)
        if verbose: print(f"  Deleted {deleted_raw} raw nodes (all types).")

        # Clean up flushed path_log rows
        if _sid:
            deleted_path = await _cleanup_path_log(_sid, older_than_hours=1)
            if verbose and deleted_path:
                print(f"  Cleaned {deleted_path} old path_log rows.")

        # Delete temporal_interactions
        if session:
            n = await session.delete_interactions()
            if verbose and n: print(f"  Deleted {n} interaction records.")

        # ── Log ───────────────────────────────────────────────────────────────
        ep_summary = episode_result.get("episode_summary", "")
        await _finish_log(cid, stats, calls, "done", None, ep_summary)

        if verbose:
            print(f"[consolidator] Done — {stats['raw_nodes_processed']} sensor nodes → "
                  f"+{stats['entities_created']} entities  "
                  f"+{stats['entities_updated']} updated  "
                  f"+{stats['edges_created']} edges  "
                  f"+{stats['info_nodes_created']} info  "
                  f"({calls} LLM calls)")

        return ConsolidationResult(
            consolidation_id=cid, session_id=_sid,
            started_at=t0, finished_at=datetime.utcnow(),
            raw_nodes_processed=stats["raw_nodes_processed"],
            entities_created=stats["entities_created"],
            entities_updated=stats["entities_updated"],
            edges_created=stats["edges_created"],
            info_nodes_created=stats["info_nodes_created"],
            llm_calls=calls, status="done", error_msg=None,
            summary_text=ep_summary[:500], llm_model=model,
        )

    except Exception as exc:
        log.exception("Consolidation failed: %s", exc)
        await _finish_log(cid, stats, calls, "failed", str(exc))
        return ConsolidationResult(
            consolidation_id=cid, session_id=_sid,
            started_at=t0, finished_at=datetime.utcnow(),
            raw_nodes_processed=stats.get("raw_nodes_processed", 0),
            entities_created=stats.get("entities_created", 0),
            entities_updated=stats.get("entities_updated", 0),
            edges_created=stats.get("edges_created", 0),
            info_nodes_created=stats.get("info_nodes_created", 0),
            llm_calls=calls, status="failed",
            error_msg=str(exc), summary_text="", llm_model=model,
        )


# ─────────────────────────────────────────────────────────────────────────────
# History query
# ─────────────────────────────────────────────────────────────────────────────

async def get_consolidation_history(
    session_id: str = None, limit: int = 20,
) -> List[Dict[str, Any]]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        if session_id:
            rows = await conn.fetch(
                """
                SELECT consolidation_id::text, session_id::text,
                       started_at, finished_at, raw_nodes_processed,
                       entities_created, entities_updated, edges_created,
                       info_nodes_created, llm_model, llm_calls,
                       status, error_msg, summary_text
                FROM consolidation_log
                WHERE session_id=$1::uuid
                ORDER BY started_at DESC LIMIT $2
                """, session_id, limit,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT consolidation_id::text, session_id::text,
                       started_at, finished_at, raw_nodes_processed,
                       entities_created, entities_updated, edges_created,
                       info_nodes_created, llm_model, llm_calls,
                       status, error_msg, summary_text
                FROM consolidation_log
                ORDER BY started_at DESC LIMIT $1
                """, limit,
            )
    return [dict(r) for r in rows]