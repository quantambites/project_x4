# live_demo.py — Robot Memory Live Sensor Demo

A real-time demo that uses your **laptop's camera, microphone, and IP geolocation**
to build a persistent spatial knowledge graph using the `robot_memory` library.

---

## What it does

```
Laptop sensors                robot_memory pipeline
─────────────────             ──────────────────────────────────────────────────
IP / GPS location  ──────────► record_position()  →  RuntimePathMap + temporal_path_log
Microphone (always)──────────► ambient: 'conversation' raw_temporal_node every 15s
SPACE pressed       ─────────► 'audio_transcript' raw_temporal_node  (STT via Whisper)
                               'video_frame'      raw_temporal_node  (VLM caption)
ENTER pressed       ─────────► flush_path_to_db()
                               flush_and_consolidate()  ← LLM parses all raw nodes
                                  → entity_nodes, info_nodes, relationship_edges
                               anchor_entity_to_map()
                               think() / deep_think()   → world map summary
```

---

## Quick start

### 1. Install dependencies

```bash
# Base robot_memory deps (if not already done)
pip install -r requirements.txt

# Live demo extras
pip install -r requirements_live.txt

# Recommended: local Whisper STT (no API key, runs on CPU ~200MB model)
pip install faster-whisper
```

### 2. Ensure your `.env` is configured

At minimum you need:

```dotenv
# .env (project root)
ROBOT_DB_DSN=postgresql://robot:robot@localhost:5432/robot_memory
ROBOT_LLM_PROVIDER=groq
ROBOT_LLM_API_KEY=gsk_...           # your Groq API key
ROBOT_LLM_MODEL=llama-3.3-70b-versatile
ROBOT_LLM_BASE_URL=https://api.groq.com/openai/v1
```

### 3. Apply the schema (once)

```bash
psql -U robot -d robot_memory -f robot_memory/schema.sql
```

### 4. Run

```bash
# From the project root — IP-based location (no GPS required)
python live_demo.py

# With precise GPS coordinates
python live_demo.py --lat 22.5726 --lon 88.3639

# On floor 1 of a multi-storey building
python live_demo.py --floor 1

# Audio only (no camera)
python live_demo.py --no-camera

# Camera only (no microphone)
python live_demo.py --no-audio
```

---

## Controls

| Key | Action |
|-----|--------|
| `SPACE` | **Toggle recording.** First press starts capturing audio + video. Second press stops, transcribes audio (Whisper/Google STT), optionally captions the last camera frame (Groq vision), and saves both as `raw_temporal_node` rows. |
| `ENTER` | **Flush & consolidate.** Stops all sensors, commits the runtime path map to `path_nodes/path_edges`, runs the LLM consolidation pipeline to extract entities/edges/info from all raw nodes, anchors entities to the map, and prints the full world map summary. |
| `Q` / `Ctrl-C` | Quit immediately without consolidating. |

---

## What gets stored

| Source | DB table | `data_type` | When |
|--------|----------|-------------|------|
| Robot position tick | `temporal_path_log` + `raw_temporal_nodes` | `observation` | Every `POSITION_INTERVAL` seconds (default 2s) |
| Ambient mic (passive) | `raw_temporal_nodes` | `conversation` | Every `AMBIENT_INTERVAL` seconds (default 15s) while idle |
| SPACE recording (mic) | `raw_temporal_nodes` | `audio_transcript` | On SPACE stop |
| SPACE recording (camera) | `raw_temporal_nodes` | `video_frame` | On SPACE stop |
| Post-ENTER: committed path | `path_nodes`, `path_edges` | — | On ENTER |
| Post-ENTER: consolidated knowledge | `entity_nodes`, `info_nodes`, `relationship_edges` | — | On ENTER (LLM) |

---

## Architecture inside live_demo.py

```
main()
│
├── AudioCapture (thread)        — sounddevice stream → deque of numpy chunks
│   └── drain_wav_bytes()        — flush deque to WAV bytes
│
├── CameraCapture (thread)       — OpenCV → deque of (ts, frame) tuples
│   └── drain_frames()           — flush deque to list
│
├── _keyboard_thread (thread)    — raw terminal → key queue (SPACE/ENTER/Q)
│
├── Ambient logger               — every 15s: drain audio → transcribe → add_raw_node
│
├── Position ticker              — every 2s: record_position() + observation node
│
└── Key handler
    ├── SPACE ON   → clear buffers, start capturing
    ├── SPACE OFF  → drain audio → transcribe_wav() → add_raw_node(audio_transcript)
    │              → drain frames → describe_frames() → add_raw_node(video_frame)
    └── ENTER      → restore_terminal()
                   → flush_path_to_db()
                   → flush_and_consolidate()   ← LLM pipeline
                   → anchor_entity_to_map()
                   → think() + deep_think()   ← world map
```

---

## Position model

Since a laptop has no odometry, the demo uses a **slow oval drift** to generate
a realistic path with structure (so the path map has nodes and edges):

```python
radius = 2.0 m
speed  = 0.05 rad/s
x = radius * cos(speed * t)
y = radius * sin(speed * t) * 0.5
```

To use real position data, replace `get_simulated_position()` with your
odometry/SLAM/GPS source. The function just needs to return `(x, y, z, heading_deg)`.

---

## Speech-to-text fallback chain

1. **`faster-whisper`** (local, recommended) — `tiny` model, runs on CPU in ~1-2s per clip
2. **`SpeechRecognition` + Google** — free, requires internet, no API key
3. Fallback string `"[audio captured — STT unavailable]"` stored as placeholder

---

## Camera captioning

When SPACE recording stops, the demo tries to caption the last JPEG frame using
the **Groq vision API** (same API key, same LLM client). If the current model
doesn't support vision or the call fails, it falls back to a simple metadata
description (frame count, duration, position).

For a model that definitely supports vision, set in `.env`:
```dotenv
ROBOT_LLM_MODEL=llama-3.2-11b-vision-preview
```

---

## Tunable constants (top of live_demo.py)

| Constant | Default | Description |
|----------|---------|-------------|
| `POSITION_INTERVAL` | `2.0` s | How often the robot's position is logged |
| `AMBIENT_INTERVAL` | `15.0` s | How often ambient audio is passively captured |
| `AUDIO_CHUNK_SEC` | `0.5` s | Audio buffer chunk size |
| `AUDIO_BUFFER_MAXSEC` | `120` s | Max audio history kept in RAM |
| `VIDEO_FPS` | `4` fps | Camera capture rate |
| `VIDEO_BUFFER_MAXSEC` | `30` s | Rolling frame buffer size |
| `CAMERA_INDEX` | `0` | OpenCV device index (`0` = default webcam) |
| `SAMPLE_RATE` | `16000` Hz | Microphone sample rate (Whisper prefers 16kHz) |

---

## Common issues

**`sounddevice` PortAudio error**
```bash
# Linux
sudo apt-get install portaudio19-dev
pip install sounddevice

# macOS
brew install portaudio
pip install sounddevice
```

**Camera not found**
Try `--no-camera` to verify everything else works, then check:
```python
import cv2
cap = cv2.VideoCapture(0)
print(cap.isOpened())   # should be True
```
Change `CAMERA_INDEX = 1` (or 2) if you have multiple cameras.

**Terminal garbled after crash**
The raw-mode keyboard listener may leave the terminal in a broken state.
Fix it with:
```bash
stty sane
# or
reset
```

**`No unprocessed raw nodes found` after ENTER**
This means the DB was not reachable during `add_raw_node()` calls. Check your
`ROBOT_DB_DSN` in `.env` and that PostgreSQL is running.

**LLM consolidation returns `"No new knowledge extracted."`**
The raw nodes were stored but the LLM found nothing entity-worthy (e.g. only
silent audio or blank frames). Speak clearly into the microphone before pressing
SPACE — mention objects, people, or places you can see.