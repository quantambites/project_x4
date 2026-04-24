"""
robot_memory
────────────
Spatial knowledge graph + path map memory system for autonomous robots.

Public API
──────────
  from robot_memory import (
      # DB lifecycle
      init_pool, close_pool,

      # Entity graph
      upsert_entity, update_entity_location,
      get_entity, get_entity_by_name,
      add_info_node, get_info_meta, fetch_info_full,
      upsert_edge, get_edges_for_entity,

      # Path map  (passive)
      record_position, get_local_map,
      # Path map  (manual anchoring)
      anchor_entity_to_map, get_entity_path_node,
      # Pathfinding
      find_path_between_entities, dijkstra_path,

      # Temporal session
      TemporalSession,

      # Think functions
      think, deep_think,
      think_path, think_nearest, think_about,
      think_similar, think_local_info,

      # LLM Consolidation  (raw sensor data → knowledge graph)
      flush_and_consolidate,
      fetch_global_context,
      get_consolidation_history,
  )

v2 additions
────────────
  think()           — now uses batch queries (2 extra queries total vs N+1)
  add_raw_node()    — store unprocessed sensor data (audio, video, commands)
  flush_and_consolidate() — LLM parses raw nodes → entity/info/edge rows,
                            then HARD DELETES temporal data
  new tables        — raw_temporal_nodes, consolidation_log
"""

from .db import init_pool, close_pool

from .graph import (
    upsert_entity,
    update_entity_location,
    get_entity,
    get_entity_by_name,
    add_info_node,
    get_info_meta,
    fetch_info_full,
    upsert_edge,
    get_edges_for_entity,
    entities_in_radius,
    k_nearest_entities,
    batch_load_info_metas,
    batch_load_edges,
    EntityNode,
    InfoNodeMeta,
    RelationshipEdge,
)

from .pathmap import (
    record_position,
    anchor_entity_to_map,
    get_entity_path_node,
    find_path_between_entities,
    dijkstra_path,
    get_local_map,
    get_runtime_map,
    PathNode,
    PathEdge,
    PathResult,
)

from .temporal import (
    TemporalSession,
    TemporalRecord,
    RawTemporalNode,
    get_all_pending_raw_nodes,
)

from .think import (
    think,
    deep_think,
    think_path,
    think_nearest,
    think_about,
    think_similar,
    think_local_info,
    ThoughtContext,
    DeepThoughtContext,
    LocalInfoContext,
)

from .consolidator import (
    flush_and_consolidate,
    fetch_global_context,
    get_consolidation_history,
    ConsolidationResult,
)


__all__ = [
    # DB lifecycle
    "init_pool", "close_pool",

    # Entity graph
    "upsert_entity", "update_entity_location",
    "get_entity", "get_entity_by_name",
    "add_info_node", "get_info_meta", "fetch_info_full",
    "upsert_edge", "get_edges_for_entity",
    "entities_in_radius", "k_nearest_entities",
    "batch_load_info_metas", "batch_load_edges",

    # Path map
    "record_position", "anchor_entity_to_map", "get_entity_path_node",
    "find_path_between_entities", "dijkstra_path", "get_local_map",
    "get_runtime_map",

    # Temporal
    "TemporalSession", "TemporalRecord",
    "RawTemporalNode", "get_all_pending_raw_nodes",

    # Think
    "think", "deep_think", "think_path", "think_nearest",
    "think_about", "think_similar", "think_local_info",
    "ThoughtContext", "DeepThoughtContext", "LocalInfoContext",

    # Consolidation
    "flush_and_consolidate", "fetch_global_context",
    "get_consolidation_history", "ConsolidationResult",

    # DTOs
    "EntityNode", "InfoNodeMeta", "RelationshipEdge",
    "PathNode", "PathEdge", "PathResult",
]