"""
robot_memory/consolidator.py
─────────────────────────────
LLM-powered consolidation of raw temporal nodes into the persistent knowledge graph.

How it works
────────────
1.  fetch_global_context()   — Pull the current knowledge graph (entities + edges + info)
                               into a compact text summary using recursive LLM calls if
                               the graph is too large to fit in one prompt.

2.  flush_and_consolidate()  — Main entry point:
      a. Fetch all unprocessed raw_temporal_nodes for the session
      b. Fetch temporal_interactions for context
      c. Build a global graph context string (recursive if large)
      d. Call LLM to interpret raw nodes → structured JSON (entities, edges, info)
      e. Write entities / info_nodes / relationship_edges to DB
      f. HARD DELETE raw_temporal_nodes and temporal_interactions
      g. Write to consolidation_log

LLM call structure
──────────────────
  • Provider: Groq (llama-3.3-70b-versatile) or Together AI (Llama-3) or Ollama
  • API: OpenAI-compatible chat completions via openai SDK
  • Structured output: LLM instructed to return ONLY valid JSON
  • Recursive context: If entity count > CONTEXT_CHUNK_SIZE, graph is
    summarised in chunks and then synthesised into a single context string

Raw data types handled
──────────────────────
  text_command, audio_transcript, conversation, observation,
  video_frame, image_description, sensor_reading

Usage
─────
  from robot_memory.consolidator import flush_and_consolidate
  log = await flush_and_consolidate(session_id="...", session=session_obj)
"""

from __future__ import annotations
import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .db import get_pool, get_llm_client, get_llm_model, log
from .graph import (
    upsert_entity, add_info_node, upsert_edge,
    get_entity_by_name,
)
from .temporal import (
    TemporalSession, RawTemporalNode,
    get_all_pending_raw_nodes,
)
from .think import deep_think

# ── Tunables ──────────────────────────────────────────────────────────────────
CONTEXT_CHUNK_SIZE = int(__import__("os").getenv("ROBOT_CONSOLIDATE_CHUNK", "30"))
MAX_RAW_PER_CALL   = int(__import__("os").getenv("ROBOT_CONSOLIDATE_BATCH", "20"))
MAX_RECURSIVE_DEPTH = 3

SYSTEM_PROMPT = """
You are a spatial knowledge graph builder for an autonomous robot.
Your job is to read raw sensor data (audio transcripts, text commands, video frame
descriptions, observations, sensor readings) that the robot captured, along with
the existing knowledge graph context, and extract structured knowledge.

You MUST output ONLY a valid JSON object — no preamble, no explanation, no markdown fences.

Output schema:
{
  "entities": [
    {
      "name": "<unique entity name>",
      "summary": "<1-2 sentence description>",
      "entity_type": "<object|place|person|vehicle|sensor|concept>",
      "tags": ["tag1", "tag2"],
      "x": <float metres East or null>,
      "y": <float metres North or null>,
      "z": <float metres Up or null>,
      "floor_level": <int or 0>,
      "facing_deg": <float yaw 0=North CW or null>,
      "crucial_words": ["word1", "word2"],
      "full_data": "<detailed description including all known facts>",
      "weight": <float 0.5-2.0, higher = more important>
    }
  ],
  "edges": [
    {
      "entity_1": "<entity name>",
      "entity_2": "<entity name>",
      "rel_type": "<near|contains|blocks|monitors|connects|services|charges|related_to>",
      "rel_name": "<short description>",
      "directed": false
    }
  ],
  "summary": "<2-3 sentence narrative of what was consolidated>"
}

Rules:
- Only create entities that are real, persistent things in the environment.
- Reuse existing entity names exactly when the thing is already known.
- Positions (x,y,z) must match the robot's spatial context at time of observation.
- If position is unknown, set x/y/z to null.
- crucial_words should be 2-6 keywords that would help retrieve this entity later.
- If no new entities or edges can be extracted, return {"entities": [], "edges": [], "summary": "No new knowledge extracted."}.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConsolidationResult:
    consolidation_id:   str
    session_id:         Optional[str]
    started_at:         datetime
    finished_at:        Optional[datetime]
    raw_nodes_processed: int
    entities_created:   int
    entities_updated:   int
    edges_created:      int
    info_nodes_created: int
    llm_calls:          int
    status:             str
    error_msg:          Optional[str]
    summary_text:       str
    llm_model:          str


# ─────────────────────────────────────────────────────────────────────────────
# Global context builder
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_global_context(max_entities: int = 200) -> str:
    """
    Build a compact text summary of the current knowledge graph.
    Used as context for the LLM consolidation prompt.

    If the graph has more than CONTEXT_CHUNK_SIZE entities, the graph is
    summarised recursively:
      1. Split entities into chunks of CONTEXT_CHUNK_SIZE
      2. LLM summarises each chunk into a dense paragraph
      3. LLM synthesises chunk summaries into a single global summary

    Returns a string of ≤ ~4000 chars suitable for inclusion in a prompt.
    """
    dctx = await deep_think(include_full_embeddings=False)

    if not dctx.all_entities:
        return "The knowledge graph is currently empty."

    # Build compact entity lines
    entity_lines = []
    for e in dctx.all_entities[:max_entities]:
        pos = f"({e.x:.1f},{e.y:.1f},{e.z:.1f})" if e.x is not None else "(?,?,?)"
        kw  = ",".join(e.top_words[:4]) if e.top_words else ""
        entity_lines.append(
            f"  [{e.entity_type or '?'}] {e.name} @ {pos} fl={e.floor_level}"
            f"  '{e.summary or ''}' [{kw}]"
        )

    edge_lines = []
    for ed in dctx.all_edges[:200]:
        n1 = next((e.name for e in dctx.all_entities if e.node_id == ed.node_id_1), ed.node_id_1[:8])
        n2 = next((e.name for e in dctx.all_entities if e.node_id == ed.node_id_2), ed.node_id_2[:8])
        edge_lines.append(f"  {n1} --[{ed.rel_type}]--> {n2}")

    entity_block = "\n".join(entity_lines)
    edge_block   = "\n".join(edge_lines[:80])  # cap at 80 edges for prompt size

    context = (
        f"EXISTING KNOWLEDGE GRAPH  ({dctx.entity_count} entities, {dctx.edge_count} edges):\n\n"
        f"ENTITIES:\n{entity_block}\n\n"
        f"EDGES:\n{edge_block}\n"
    )

    # If graph is small enough, return directly
    if len(dctx.all_entities) <= CONTEXT_CHUNK_SIZE:
        return context

    # Graph too large — use recursive LLM summarisation
    return await _recursive_summarise_context(entity_lines, edge_lines)


async def _recursive_summarise_context(
    entity_lines: List[str],
    edge_lines:   List[str],
    depth:        int = 0,
) -> str:
    """
    Recursively summarise large graphs into a compact context string.
    Each level halves the entity list via LLM summarisation.
    """
    if depth >= MAX_RECURSIVE_DEPTH or len(entity_lines) <= CONTEXT_CHUNK_SIZE:
        return (
            f"KNOWLEDGE GRAPH SUMMARY:\nEntities:\n" + "\n".join(entity_lines[:CONTEXT_CHUNK_SIZE])
            + f"\nEdges:\n" + "\n".join(edge_lines[:40])
        )

    client = get_llm_client()
    model  = get_llm_model()

    # Split into chunks
    chunk_size = CONTEXT_CHUNK_SIZE
    chunks = [entity_lines[i:i+chunk_size] for i in range(0, len(entity_lines), chunk_size)]
    chunk_summaries = []

    for chunk in chunks:
        chunk_text = "\n".join(chunk)
        try:
            resp = await client.chat.completions.create(
                model=model,
                max_tokens=400,
                temperature=0.1,
                messages=[
                    {"role": "system", "content":
                     "You are summarising a robot's spatial knowledge graph. "
                     "Produce a dense 2-4 sentence paragraph capturing the most "
                     "important entities, their locations, and relationships. "
                     "Output ONLY the paragraph, no other text."},
                    {"role": "user", "content":
                     f"Summarise these entities:\n{chunk_text}\n\nEdges:\n" + "\n".join(edge_lines[:20])},
                ],
            )
            summary = resp.choices[0].message.content.strip()
            chunk_summaries.append(summary)
        except Exception as exc:
            log.warning("Chunk summary LLM call failed: %s", exc)
            chunk_summaries.append(chunk_text[:300])

    if len(chunk_summaries) == 1:
        return f"KNOWLEDGE GRAPH CONTEXT:\n{chunk_summaries[0]}"

    # Synthesise chunk summaries
    combined = "\n\n".join(f"Area {i+1}:\n{s}" for i, s in enumerate(chunk_summaries))
    try:
        resp = await client.chat.completions.create(
            model=model,
            max_tokens=600,
            temperature=0.1,
            messages=[
                {"role": "system", "content":
                 "Synthesise multiple area summaries of a robot's environment into "
                 "a single coherent global context paragraph. Be dense and factual. "
                 "Output ONLY the paragraph."},
                {"role": "user", "content": combined},
            ],
        )
        return f"KNOWLEDGE GRAPH CONTEXT:\n{resp.choices[0].message.content.strip()}"
    except Exception as exc:
        log.warning("Context synthesis failed: %s", exc)
        return f"KNOWLEDGE GRAPH CONTEXT:\n{combined[:1000]}"


# ─────────────────────────────────────────────────────────────────────────────
# Core LLM extraction call
# ─────────────────────────────────────────────────────────────────────────────

def _format_raw_nodes(raw_nodes: List[RawTemporalNode]) -> str:
    """Format raw nodes into a readable string for the LLM prompt."""
    lines = []
    for r in raw_nodes:
        ts  = r.captured_at.strftime("%H:%M:%S")
        pos = f"({r.x:.1f},{r.y:.1f},{r.z:.1f})" if r.x is not None else "(?,?,?)"
        hdg = f" hdg={r.heading_deg:.0f}°" if r.heading_deg is not None else ""
        lines.append(
            f"[{ts}] [{r.data_type}] @ fl={r.floor_level} {pos}{hdg}\n"
            f"  TEXT: {(r.raw_text or '').strip()}\n"
            + (f"  JSON: {json.dumps(r.raw_json, default=str)[:200]}\n" if r.raw_json else "")
        )
    return "\n".join(lines)


async def _llm_extract_knowledge(
    raw_text_block: str,
    global_context: str,
    client,
    model: str,
) -> Tuple[dict, str]:
    """
    Single LLM call: extract structured knowledge from raw sensor data.
    Returns (parsed_json_dict, raw_response_text).
    """
    user_prompt = (
        f"{global_context}\n\n"
        f"RAW SENSOR DATA TO PROCESS:\n{raw_text_block}\n\n"
        f"Extract all entities, relationships, and knowledge from the raw data above. "
        f"Place each entity at its observed spatial position. "
        f"Link it to existing entities where appropriate. "
        f"Output ONLY valid JSON."
    )

    response = await client.chat.completions.create(
        model=model,
        max_tokens=2000,
        temperature=0.1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
    )

    raw_response = response.choices[0].message.content.strip()

    # Strip any accidental markdown fences
    clean = raw_response
    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.lower().startswith("json"):
            clean = clean[4:]
    clean = clean.strip().strip("`")

    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError as exc:
        log.warning("LLM returned invalid JSON: %s\n---\n%s", exc, raw_response[:500])
        parsed = {"entities": [], "edges": [], "summary": "JSON parse error."}

    return parsed, raw_response


# ─────────────────────────────────────────────────────────────────────────────
# DB write helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _write_extraction(
    extracted: dict,
    stats:     Dict[str, int],
) -> None:
    """Write LLM-extracted entities/edges/info into the knowledge graph."""
    entity_name_to_id: Dict[str, str] = {}

    for ent_data in extracted.get("entities", []):
        name = (ent_data.get("name") or "").strip()
        if not name:
            continue

        # Try to reuse existing entity
        existing = await get_entity_by_name(name)

        x    = ent_data.get("x") or 0.0
        y    = ent_data.get("y") or 0.0
        z    = ent_data.get("z") or 0.0
        fl   = int(ent_data.get("floor_level") or 0)
        tags = ent_data.get("tags") or []
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",")]

        if existing:
            # Update location if new position is non-zero
            if x != 0.0 or y != 0.0:
                from .graph import update_entity_location
                await update_entity_location(existing.node_id, x, y, z, fl)
            eid = existing.node_id
            stats["entities_updated"] = stats.get("entities_updated", 0) + 1
        else:
            eid = await upsert_entity(
                name=name,
                summary=ent_data.get("summary") or "",
                entity_type=ent_data.get("entity_type") or "object",
                x=x, y=y, z=z, floor_level=fl,
                facing_deg=ent_data.get("facing_deg"),
                tags=tags,
                weight=float(ent_data.get("weight") or 1.0),
            )
            stats["entities_created"] = stats.get("entities_created", 0) + 1

        entity_name_to_id[name] = eid

        # Write info node
        crucial_words = ent_data.get("crucial_words") or []
        if isinstance(crucial_words, str):
            crucial_words = [w.strip() for w in crucial_words.split(",")]
        full_data = ent_data.get("full_data") or ent_data.get("summary") or ""
        if full_data or crucial_words:
            await add_info_node(
                entity_id=eid,
                crucial_words=crucial_words,
                full_data=full_data,
                weight=float(ent_data.get("weight") or 1.0),
            )
            stats["info_nodes_created"] = stats.get("info_nodes_created", 0) + 1

    # Write edges
    for edge_data in extracted.get("edges", []):
        n1_name = (edge_data.get("entity_1") or "").strip()
        n2_name = (edge_data.get("entity_2") or "").strip()
        if not n1_name or not n2_name or n1_name == n2_name:
            continue

        # Resolve IDs — check local cache first, then DB
        eid1 = entity_name_to_id.get(n1_name)
        if not eid1:
            e1 = await get_entity_by_name(n1_name)
            if e1:
                eid1 = e1.node_id

        eid2 = entity_name_to_id.get(n2_name)
        if not eid2:
            e2 = await get_entity_by_name(n2_name)
            if e2:
                eid2 = e2.node_id

        if not eid1 or not eid2:
            log.debug("Edge skip — entity not found: '%s' or '%s'", n1_name, n2_name)
            continue

        await upsert_edge(
            node_id_1=eid1,
            node_id_2=eid2,
            rel_type=edge_data.get("rel_type") or "related_to",
            rel_name=edge_data.get("rel_name") or "",
            directed=bool(edge_data.get("directed", False)),
        )
        stats["edges_created"] = stats.get("edges_created", 0) + 1


# ─────────────────────────────────────────────────────────────────────────────
# Consolidation log helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _create_consolidation_log(session_id: Optional[str], model: str) -> str:
    pool = await get_pool()
    cid  = str(uuid.uuid4())
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO consolidation_log
                (consolidation_id, session_id, llm_model, status)
            VALUES ($1::uuid, $2::uuid, $3, 'running')
            """,
            cid, session_id, model,
        )
    return cid


async def _finish_consolidation_log(
    consolidation_id: str,
    stats:            Dict[str, int],
    llm_calls:        int,
    status:           str,
    error_msg:        str   = None,
    summary_text:     str   = "",
) -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE consolidation_log SET
                finished_at         = NOW(),
                raw_nodes_processed = $2,
                entities_created    = $3,
                entities_updated    = $4,
                edges_created       = $5,
                info_nodes_created  = $6,
                llm_calls           = $7,
                status              = $8,
                error_msg           = $9,
                summary_text        = $10
            WHERE consolidation_id = $1::uuid
            """,
            consolidation_id,
            stats.get("raw_nodes_processed", 0),
            stats.get("entities_created", 0),
            stats.get("entities_updated", 0),
            stats.get("edges_created", 0),
            stats.get("info_nodes_created", 0),
            llm_calls,
            status,
            error_msg,
            summary_text,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

async def flush_and_consolidate(
    session_id:   str             = None,
    session:      TemporalSession = None,
    batch_size:   int             = MAX_RAW_PER_CALL,
    verbose:      bool            = True,
) -> ConsolidationResult:
    """
    Main consolidation entry point.

    Steps:
      1.  Fetch unprocessed raw_temporal_nodes
      2.  Fetch temporal_interactions for extra context
      3.  Build global graph context (recursive LLM summary if large)
      4.  Batch-process raw nodes through LLM (batch_size per call)
      5.  Write extracted entities/edges/info to DB
      6.  Mark raw nodes as processed
      7.  HARD DELETE raw nodes + temporal_interactions
      8.  Update consolidation_log

    Returns ConsolidationResult with counts and status.
    """
    _sid  = session_id or (session.session_id if session else None)
    model = get_llm_model()
    cid   = await _create_consolidation_log(_sid, model)

    started_at = datetime.utcnow()
    stats: Dict[str, int] = {
        "raw_nodes_processed": 0,
        "entities_created":    0,
        "entities_updated":    0,
        "edges_created":       0,
        "info_nodes_created":  0,
    }
    llm_calls   = 0
    error_msg   = None
    summary_texts: List[str] = []

    try:
        client = get_llm_client()

        # ── 1. Fetch raw nodes ───────────────────────────────────────────────
        raw_nodes = await get_all_pending_raw_nodes(session_id=_sid, limit=2000)
        if not raw_nodes:
            if verbose:
                print("[consolidator] No unprocessed raw nodes found.")
            await _finish_consolidation_log(cid, stats, 0, "done",
                                            summary_text="Nothing to consolidate.")
            return ConsolidationResult(
                consolidation_id=cid, session_id=_sid,
                started_at=started_at, finished_at=datetime.utcnow(),
                raw_nodes_processed=0, entities_created=0, entities_updated=0,
                edges_created=0, info_nodes_created=0,
                llm_calls=0, status="done",
                error_msg=None, summary_text="Nothing to consolidate.",
                llm_model=model,
            )

        if verbose:
            print(f"[consolidator] {len(raw_nodes)} raw nodes to process "
                  f"(batch={batch_size})  model={model}")

        # ── 2. Fetch temporal interactions for extra context ─────────────────
        interaction_context = ""
        if session:
            try:
                records = await session.get_records(limit=100)
                if records:
                    lines = [
                        f"  [{r.interaction_ts.strftime('%H:%M:%S')}] "
                        f"{r.entity_name} @ ({r.x:.1f},{r.y:.1f},{r.z:.1f}) "
                        f"fl={r.floor_level}: {r.notes or ''}"
                        for r in records
                    ]
                    interaction_context = (
                        "RECENT ROBOT INTERACTIONS (temporal log):\n"
                        + "\n".join(lines) + "\n\n"
                    )
            except Exception as e:
                log.warning("Could not load interactions: %s", e)

        # ── 3. Build global graph context ────────────────────────────────────
        if verbose:
            print("[consolidator] Building global graph context...")
        global_context = await fetch_global_context()
        if interaction_context:
            global_context = interaction_context + global_context

        # ── 4. Batch LLM extraction ──────────────────────────────────────────
        batches   = [raw_nodes[i:i+batch_size] for i in range(0, len(raw_nodes), batch_size)]
        all_raw_ids: List[str] = []

        for batch_idx, batch in enumerate(batches):
            if verbose:
                print(f"[consolidator] Batch {batch_idx+1}/{len(batches)} "
                      f"({len(batch)} nodes)...")

            raw_text_block = _format_raw_nodes(batch)
            try:
                extracted, _ = await _llm_extract_knowledge(
                    raw_text_block, global_context, client, model
                )
                llm_calls += 1
            except Exception as exc:
                log.error("LLM call failed for batch %d: %s", batch_idx, exc)
                error_msg = str(exc)
                continue

            summary_texts.append(extracted.get("summary", ""))

            # ── 5. Write extracted knowledge ─────────────────────────────────
            await _write_extraction(extracted, stats)
            stats["raw_nodes_processed"] += len(batch)
            all_raw_ids.extend([r.raw_id for r in batch])

            # Update global context after each batch so next batch
            # sees newly-created entities (lightweight refresh)
            if batch_idx < len(batches) - 1:
                global_context = await fetch_global_context()
                if interaction_context:
                    global_context = interaction_context + global_context

            if verbose:
                print(f"  → entities+={stats['entities_created']}  "
                      f"edges+={stats['edges_created']}  "
                      f"info+={stats['info_nodes_created']}")

        # ── 6. Mark raw nodes as processed ───────────────────────────────────
        if all_raw_ids and _sid:
            _sess = session or TemporalSession(session_id=_sid)
            await _sess.mark_raw_processed(all_raw_ids, cid)

        # ── 7. HARD DELETE temporal data after successful consolidation ───────
        deleted_raw = 0
        deleted_interactions = 0
        if all_raw_ids:
            deleted_raw = await _delete_raw_by_ids(all_raw_ids)
            if verbose:
                print(f"[consolidator] Deleted {deleted_raw} raw nodes from DB.")

        if session and stats["raw_nodes_processed"] > 0:
            deleted_interactions = await session.delete_interactions()
            if verbose:
                print(f"[consolidator] Deleted {deleted_interactions} interaction records.")

        # ── 8. Finalise log ───────────────────────────────────────────────────
        combined_summary = " | ".join(s for s in summary_texts if s)
        await _finish_consolidation_log(
            cid, stats, llm_calls,
            status="done" if not error_msg else "partial",
            error_msg=error_msg,
            summary_text=combined_summary[:1000],
        )

        if verbose:
            print(f"[consolidator] Done — {stats['raw_nodes_processed']} raw nodes → "
                  f"{stats['entities_created']} new entities, "
                  f"{stats['edges_created']} new edges, "
                  f"{stats['info_nodes_created']} info nodes  ({llm_calls} LLM calls)")

        return ConsolidationResult(
            consolidation_id=cid,
            session_id=_sid,
            started_at=started_at,
            finished_at=datetime.utcnow(),
            raw_nodes_processed=stats["raw_nodes_processed"],
            entities_created=stats["entities_created"],
            entities_updated=stats["entities_updated"],
            edges_created=stats["edges_created"],
            info_nodes_created=stats["info_nodes_created"],
            llm_calls=llm_calls,
            status="done" if not error_msg else "partial",
            error_msg=error_msg,
            summary_text=combined_summary[:500],
            llm_model=model,
        )

    except Exception as exc:
        log.exception("Consolidation failed: %s", exc)
        await _finish_consolidation_log(cid, stats, llm_calls, "failed", str(exc))
        return ConsolidationResult(
            consolidation_id=cid, session_id=_sid,
            started_at=started_at, finished_at=datetime.utcnow(),
            raw_nodes_processed=stats.get("raw_nodes_processed", 0),
            entities_created=stats.get("entities_created", 0),
            entities_updated=stats.get("entities_updated", 0),
            edges_created=stats.get("edges_created", 0),
            info_nodes_created=stats.get("info_nodes_created", 0),
            llm_calls=llm_calls, status="failed",
            error_msg=str(exc), summary_text="",
            llm_model=model,
        )


async def _delete_raw_by_ids(raw_ids: List[str]) -> int:
    """Hard-delete raw_temporal_nodes by ID list."""
    if not raw_ids:
        return 0
    pool = await get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM raw_temporal_nodes WHERE raw_id = ANY($1::uuid[])",
            raw_ids,
        )
    try:
        return int(str(result).split()[-1])
    except Exception:
        return len(raw_ids)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: get consolidation history
# ─────────────────────────────────────────────────────────────────────────────

async def get_consolidation_history(
    session_id: str = None,
    limit:      int = 20,
) -> List[Dict[str, Any]]:
    """Fetch recent consolidation log entries."""
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
                """,
                session_id, limit,
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
                """,
                limit,
            )
    return [dict(r) for r in rows]