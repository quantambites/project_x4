-- ============================================================
--  ROBOT SPATIAL KNOWLEDGE GRAPH + PATH MAP  —  SCHEMA  (3D)
--  v2 — raw_temporal_nodes, consolidation_log, perf indexes
-- ============================================================
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "postgis";

-- ============================================================
-- COORDINATE SYSTEM
-- x = metres East, y = metres North, z = metres Up (z=0 ground)
-- floor_level = integer floor number (0=ground)
-- heading_deg = yaw (0=North CW), pitch_deg = nose-up angle
-- ============================================================


-- ============================================================
-- 1. ENTITY NODES
-- ============================================================
CREATE TABLE IF NOT EXISTS entity_nodes (
    node_id             UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name                TEXT NOT NULL,
    summary             TEXT,
    weight              FLOAT NOT NULL DEFAULT 1.0,
    location            GEOMETRY(PointZ, 0),
    location_ts         TIMESTAMPTZ,
    floor_level         INT  DEFAULT 0,
    facing_deg          FLOAT,
    pitch_deg           FLOAT,
    bbox_dx             FLOAT,
    bbox_dy             FLOAT,
    bbox_dz             FLOAT,
    entity_type         TEXT,
    tags                TEXT[],
    runtime_embedding   vector(128),
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW(),
    image_ptrs          TEXT[],
    video_ptr           TEXT,
    audio_ptr           TEXT
);

-- 3D spatial index (critical for think() ST_3DDWithin)
CREATE INDEX IF NOT EXISTS idx_entity_location_3d  ON entity_nodes USING GIST(location);
CREATE INDEX IF NOT EXISTS idx_entity_floor        ON entity_nodes(floor_level);
CREATE INDEX IF NOT EXISTS idx_entity_name_text    ON entity_nodes USING GIN(to_tsvector('english', name));
CREATE INDEX IF NOT EXISTS idx_entity_weight       ON entity_nodes(weight DESC);
CREATE INDEX IF NOT EXISTS idx_entity_type         ON entity_nodes(entity_type);
-- Covering index for think(): avoids heap fetch on floor-filtered spatial hits
CREATE INDEX IF NOT EXISTS idx_entity_floor_w      ON entity_nodes(floor_level, weight DESC)
    INCLUDE (node_id, name, summary, entity_type, tags);
-- IVFFlat for fast runtime cosine search (vec 128)
CREATE INDEX IF NOT EXISTS idx_entity_rt_embedding ON entity_nodes
    USING ivfflat(runtime_embedding vector_cosine_ops) WITH (lists = 50);


-- ============================================================
-- 2. INFORMATION NODES  (heavy — not in runtime memory)
-- ============================================================
CREATE TABLE IF NOT EXISTS info_nodes (
    node_id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_id       UUID REFERENCES entity_nodes(node_id) ON DELETE CASCADE,
    full_data       TEXT,
    embedding       vector(1536),
    weight          FLOAT NOT NULL DEFAULT 1.0,
    crucial_words   TEXT[],
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    image_ptr       TEXT,
    video_ptr       TEXT,
    audio_ptr       TEXT
);

CREATE INDEX IF NOT EXISTS idx_info_entity        ON info_nodes(entity_id);
CREATE INDEX IF NOT EXISTS idx_info_entity_weight ON info_nodes(entity_id, weight DESC);
CREATE INDEX IF NOT EXISTS idx_info_embedding     ON info_nodes
    USING ivfflat(embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_info_words         ON info_nodes USING GIN(crucial_words);
CREATE INDEX IF NOT EXISTS idx_info_weight        ON info_nodes(weight DESC);


-- ============================================================
-- 3. RELATIONSHIP EDGES
-- ============================================================
CREATE TABLE IF NOT EXISTS relationship_edges (
    edge_id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    summary         TEXT,
    rel_type        TEXT NOT NULL,
    rel_name        TEXT,
    node_id_1       UUID NOT NULL REFERENCES entity_nodes(node_id) ON DELETE CASCADE,
    node_id_2       UUID NOT NULL REFERENCES entity_nodes(node_id) ON DELETE CASCADE,
    weight          FLOAT NOT NULL DEFAULT 1.0,
    directed        BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT no_self_loop CHECK (node_id_1 <> node_id_2)
);

CREATE INDEX IF NOT EXISTS idx_edge_n1      ON relationship_edges(node_id_1);
CREATE INDEX IF NOT EXISTS idx_edge_n2      ON relationship_edges(node_id_2);
CREATE INDEX IF NOT EXISTS idx_edge_type    ON relationship_edges(rel_type);
CREATE INDEX IF NOT EXISTS idx_edge_n1_type ON relationship_edges(node_id_1, rel_type);
CREATE UNIQUE INDEX IF NOT EXISTS idx_edge_unique
    ON relationship_edges(node_id_1, node_id_2, rel_type);


-- ============================================================
-- 4. PATH MAP
-- ============================================================
CREATE TABLE IF NOT EXISTS path_nodes (
    path_node_id    UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    position        GEOMETRY(PointZ, 0) NOT NULL,
    floor_level     INT  DEFAULT 0,
    heading_deg     FLOAT,
    pitch_deg       FLOAT,
    visited_at      TIMESTAMPTZ DEFAULT NOW(),
    visit_count     INT DEFAULT 1,
    tags            TEXT[]
);

CREATE INDEX IF NOT EXISTS idx_path_position_3d ON path_nodes USING GIST(position);
CREATE INDEX IF NOT EXISTS idx_path_floor       ON path_nodes(floor_level);
CREATE INDEX IF NOT EXISTS idx_path_visited     ON path_nodes(visited_at DESC);


CREATE TABLE IF NOT EXISTS path_edges (
    path_edge_id        UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    from_node_id        UUID NOT NULL REFERENCES path_nodes(path_node_id) ON DELETE CASCADE,
    to_node_id          UUID NOT NULL REFERENCES path_nodes(path_node_id) ON DELETE CASCADE,
    distance_3d_m       FLOAT,
    distance_2d_m       FLOAT,
    delta_z_m           FLOAT,
    traversal_cost      FLOAT DEFAULT 1.0,
    traversal_count     INT DEFAULT 1,
    last_traversed      TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT no_self_path CHECK (from_node_id <> to_node_id)
);

CREATE INDEX IF NOT EXISTS idx_pedge_from ON path_edges(from_node_id);
CREATE INDEX IF NOT EXISTS idx_pedge_to   ON path_edges(to_node_id);


-- Temporal path log (written by every record_position() call)
CREATE TABLE IF NOT EXISTS temporal_path_log (
    log_id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id      UUID NOT NULL,
    x               FLOAT NOT NULL,
    y               FLOAT NOT NULL,
    z               FLOAT NOT NULL DEFAULT 0.0,
    floor_level     INT  DEFAULT 0,
    heading_deg     FLOAT,
    pitch_deg       FLOAT,
    tags            TEXT[],
    recorded_at     TIMESTAMPTZ DEFAULT NOW(),
    flushed         BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_tpl_session  ON temporal_path_log(session_id);
CREATE INDEX IF NOT EXISTS idx_tpl_recorded ON temporal_path_log(recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_tpl_floor    ON temporal_path_log(floor_level);
CREATE INDEX IF NOT EXISTS idx_tpl_flushed  ON temporal_path_log(flushed) WHERE flushed = FALSE;


-- ============================================================
-- 5. ENTITY–PATH ANCHOR
-- ============================================================
CREATE TABLE IF NOT EXISTS entity_path_anchors (
    anchor_id       UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_id       UUID NOT NULL REFERENCES entity_nodes(node_id) ON DELETE CASCADE,
    path_node_id    UUID NOT NULL REFERENCES path_nodes(path_node_id) ON DELETE CASCADE,
    anchored_at     TIMESTAMPTZ DEFAULT NOW(),
    confidence      FLOAT DEFAULT 1.0,
    UNIQUE (entity_id, path_node_id)
);

CREATE INDEX IF NOT EXISTS idx_anchor_entity ON entity_path_anchors(entity_id);
CREATE INDEX IF NOT EXISTS idx_anchor_path   ON entity_path_anchors(path_node_id);


-- ============================================================
-- 6. TEMPORAL INTERACTIONS  (per-session entity encounter log)
--    DELETED (not just flushed) after LLM consolidation.
-- ============================================================
CREATE TABLE IF NOT EXISTS temporal_interactions (
    interaction_id  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id      UUID NOT NULL,
    entity_id       UUID REFERENCES entity_nodes(node_id) ON DELETE SET NULL,
    entity_name     TEXT,
    location_snap   GEOMETRY(PointZ, 0),
    floor_level     INT DEFAULT 0,
    path_log_ref    UUID REFERENCES temporal_path_log(log_id) ON DELETE SET NULL,
    interaction_ts  TIMESTAMPTZ DEFAULT NOW(),
    notes           TEXT
);

CREATE INDEX IF NOT EXISTS idx_temporal_session  ON temporal_interactions(session_id);
CREATE INDEX IF NOT EXISTS idx_temporal_entity   ON temporal_interactions(entity_id);
CREATE INDEX IF NOT EXISTS idx_temporal_location ON temporal_interactions USING GIST(location_snap);
CREATE INDEX IF NOT EXISTS idx_temporal_floor    ON temporal_interactions(floor_level);
CREATE INDEX IF NOT EXISTS idx_temporal_ts       ON temporal_interactions(interaction_ts DESC);


-- ============================================================
-- 7. RAW TEMPORAL NODES  (unprocessed sensor data)
--
--    Stores: text commands, audio transcripts, video frame metadata,
--    raw conversation chunks, sensor readings — anything unstructured.
--
--    LLM consolidation reads these, extracts entities/edges/info,
--    writes to the knowledge graph, then HARD-DELETES these rows.
--    temporal_interactions is also hard-deleted after consolidation.
-- ============================================================
CREATE TABLE IF NOT EXISTS raw_temporal_nodes (
    raw_id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id          UUID NOT NULL,

    -- 'text_command' | 'audio_transcript' | 'video_frame' | 'conversation'
    -- | 'sensor_reading' | 'observation' | 'image_description'
    data_type           TEXT NOT NULL,

    -- Raw payload
    raw_text            TEXT,   -- transcripts, commands, observations
    raw_json            JSONB,  -- structured: sensor readings, frame metadata

    -- Spatial context at capture time
    x                   FLOAT,
    y                   FLOAT,
    z                   FLOAT DEFAULT 0.0,
    floor_level         INT DEFAULT 0,
    heading_deg         FLOAT,

    captured_at         TIMESTAMPTZ DEFAULT NOW(),

    -- Optional early-binding to a known entity
    related_entity_id   UUID REFERENCES entity_nodes(node_id) ON DELETE SET NULL,

    -- Processing state
    processed           BOOLEAN DEFAULT FALSE,
    consolidation_id    UUID    -- set when processed; links to consolidation_log
);

CREATE INDEX IF NOT EXISTS idx_raw_session  ON raw_temporal_nodes(session_id);
CREATE INDEX IF NOT EXISTS idx_raw_type     ON raw_temporal_nodes(data_type);
CREATE INDEX IF NOT EXISTS idx_raw_unproc   ON raw_temporal_nodes(processed) WHERE processed = FALSE;
CREATE INDEX IF NOT EXISTS idx_raw_captured ON raw_temporal_nodes(captured_at DESC);
CREATE INDEX IF NOT EXISTS idx_raw_floor    ON raw_temporal_nodes(floor_level);


-- ============================================================
-- 8. CONSOLIDATION LOG  (audit trail of LLM consolidation runs)
-- ============================================================
CREATE TABLE IF NOT EXISTS consolidation_log (
    consolidation_id    UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id          UUID,
    started_at          TIMESTAMPTZ DEFAULT NOW(),
    finished_at         TIMESTAMPTZ,
    raw_nodes_processed INT DEFAULT 0,
    entities_created    INT DEFAULT 0,
    entities_updated    INT DEFAULT 0,
    edges_created       INT DEFAULT 0,
    info_nodes_created  INT DEFAULT 0,
    llm_model           TEXT,
    llm_calls           INT DEFAULT 0,
    status              TEXT DEFAULT 'running',  -- 'running' | 'done' | 'failed'
    error_msg           TEXT,
    summary_text        TEXT
);

CREATE INDEX IF NOT EXISTS idx_consol_session ON consolidation_log(session_id);
CREATE INDEX IF NOT EXISTS idx_consol_status  ON consolidation_log(status);
CREATE INDEX IF NOT EXISTS idx_consol_started ON consolidation_log(started_at DESC);


-- ============================================================
-- 9. HELPER VIEWS
-- ============================================================

CREATE OR REPLACE VIEW vw_entity_runtime AS
SELECT
    e.node_id,
    e.name,
    e.summary,
    e.weight,
    ST_X(e.location::geometry)  AS x,
    ST_Y(e.location::geometry)  AS y,
    ST_Z(e.location::geometry)  AS z,
    e.floor_level,
    e.facing_deg,
    e.pitch_deg,
    e.bbox_dx, e.bbox_dy, e.bbox_dz,
    e.location_ts,
    e.entity_type,
    e.tags,
    e.runtime_embedding,
    ARRAY(
        SELECT crucial_words FROM info_nodes i
        WHERE i.entity_id = e.node_id
        ORDER BY i.weight DESC LIMIT 1
    ) AS top_words
FROM entity_nodes e;


CREATE OR REPLACE VIEW vw_path_nodes_3d AS
SELECT
    pn.path_node_id,
    ST_X(pn.position::geometry) AS x,
    ST_Y(pn.position::geometry) AS y,
    ST_Z(pn.position::geometry) AS z,
    pn.floor_level,
    pn.heading_deg,
    pn.pitch_deg,
    pn.visited_at,
    pn.visit_count,
    pn.tags
FROM path_nodes pn;


CREATE OR REPLACE VIEW vw_raw_pending AS
SELECT r.* FROM raw_temporal_nodes r WHERE r.processed = FALSE;


-- ============================================================
-- MIGRATION: run once on existing DBs to add new tables
-- ============================================================
-- psql -U robot -d robot_memory -f schema.sql
-- (IF NOT EXISTS guards make it safe to re-run)