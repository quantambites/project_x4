"""
live_demo.py  —  Robot Memory Live Sensor Demo  (v3)
═════════════════════════════════════════════════════
Resumes from the last known position in the DB, captures camera/mic,
builds the world map on ENTER.

Key changes vs v2
─────────────────
  * Resumes from last committed path_node on startup (no lost position)
  * NO observation nodes in raw_temporal_nodes — path stored in
    temporal_path_log only (cleaner, cheaper, no LLM noise)
  * SPACE recording embeds jpeg_b64 / wav_b64 in raw_json so the
    consolidator can promote them to entity image_ptrs / audio_ptr
  * URL auto-correction for ROBOT_LLM_BASE_URL (/chat/completions stripped)
  * Ambient capture thread passes loop explicitly (Python 3.10+ safe)
  * Terminal restore on every exit path (Windows + Unix)

Controls
--------
  SPACE  — Toggle recording. On stop: transcribes audio, captions frame,
           saves as raw_temporal_node with spatial position + media b64.
  ENTER  — Flush path to DB, run LLM consolidation, anchor entities, show map.
  Q      — Quit without consolidating.

Run
---
  python live_demo.py
  python live_demo.py --lat 22.57 --lon 88.37
  python live_demo.py --floor 1
  python live_demo.py --no-camera --no-audio
"""

from __future__ import annotations
import argparse, asyncio, base64, io, json, math
import os, queue, re, sys, threading, time, wave
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import robot_memory as rm
import robot_memory.db as _rmdb
from robot_memory.pathmap import flush_path_to_db

# ─────────────────────────────────────────────────────────────────────────────
# Tunables
# ─────────────────────────────────────────────────────────────────────────────
POSITION_INTERVAL   = 2.0     # s between position ticks
AMBIENT_INTERVAL    = 20.0    # s between passive audio grabs while idle
AUDIO_CHUNK_SEC     = 0.5
AUDIO_BUFFER_MAXSEC = 120
VIDEO_FPS           = 4
VIDEO_BUFFER_MAXSEC = 30
CAMERA_INDEX        = 0
SAMPLE_RATE         = 16000
CHANNELS            = 1
MAX_MEDIA_BYTES     = 200_000  # don't inline b64 above this size

# ─────────────────────────────────────────────────────────────────────────────
# Terminal colours
# ─────────────────────────────────────────────────────────────────────────────
R="\033[91m"; G="\033[92m"; Y="\033[93m"; B="\033[94m"
M="\033[95m"; C="\033[96m"; W="\033[97m"
DIM="\033[2m"; RST="\033[0m"; BOLD="\033[1m"

def bar(title="", width=72):
    if title:
        pad = max(0, width - len(title) - 6)
        print(f"\n{BOLD}{C}{'='*2}  {title}  {'='*pad}{RST}")
    else:
        print(f"{DIM}{'-'*width}{RST}")


# ─────────────────────────────────────────────────────────────────────────────
# Location helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_ip_location() -> Tuple[float, float]:
    try:
        import requests
        d = requests.get("https://ipapi.co/json/", timeout=5).json()
        lat = float(d.get("latitude", 0))
        lon = float(d.get("longitude", 0))
        print(f"  {G}IP location: {lat:.5f}, {lon:.5f}"
              f"  ({d.get('city','?')}, {d.get('country_name','?')}){RST}")
        return lat, lon
    except Exception as e:
        print(f"  {Y}IP location failed ({e}) — using (0,0){RST}")
        return 0.0, 0.0


async def resume_last_position() -> Optional[Tuple[float, float, float, int]]:
    """Load the most-recently visited path_node from DB."""
    pool = await _rmdb.get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT ST_X(position::geometry) AS x,
                   ST_Y(position::geometry) AS y,
                   ST_Z(position::geometry) AS z,
                   floor_level
            FROM   path_nodes
            ORDER  BY visited_at DESC
            LIMIT  1
        """)
    if row and row["x"] is not None:
        return float(row["x"]), float(row["y"]), float(row["z"]), int(row["floor_level"] or 0)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Simulated position drift  (replace with real odometry / SLAM)
# ─────────────────────────────────────────────────────────────────────────────

def get_position(step: int, ox: float, oy: float,
                 oz: float) -> Tuple[float, float, float, float]:
    t = step * POSITION_INTERVAL
    x = ox + 2.0 * math.cos(0.05 * t)
    y = oy + 1.0 * math.sin(0.05 * t)
    z = oz
    hdg = math.degrees(0.05 * t + math.pi / 2) % 360
    return x, y, z, hdg


# ─────────────────────────────────────────────────────────────────────────────
# Audio capture
# ─────────────────────────────────────────────────────────────────────────────

class AudioCapture:
    def __init__(self):
        self._buf: deque = deque()
        self._lock = threading.Lock()
        self._stream = None
        self.available = False
        try:
            import sounddevice, numpy
            self._sd = sounddevice; self._np = numpy
            self.available = True
        except ImportError:
            print(f"  {Y}sounddevice not available (pip install sounddevice){RST}")

    def start(self):
        if not self.available: return
        chunk = int(SAMPLE_RATE * AUDIO_CHUNK_SEC)
        max_c = int(AUDIO_BUFFER_MAXSEC / AUDIO_CHUNK_SEC)
        def cb(indata, frames, ti, status):
            with self._lock:
                self._buf.append(indata.copy())
                while len(self._buf) > max_c: self._buf.popleft()
        self._stream = self._sd.InputStream(
            samplerate=SAMPLE_RATE, channels=CHANNELS,
            dtype="float32", blocksize=chunk, callback=cb)
        self._stream.start()
        print(f"  {G}Microphone ready{RST}")

    def stop(self):
        if self._stream:
            try: self._stream.stop(); self._stream.close()
            except Exception: pass

    def drain_wav(self) -> Optional[bytes]:
        if not self.available: return None
        import numpy as np
        with self._lock:
            if not self._buf: return None
            frames = np.concatenate(list(self._buf)); self._buf.clear()
        pcm = (frames * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(CHANNELS); wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE); wf.writeframes(pcm.tobytes())
        return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Speech-to-text
# ─────────────────────────────────────────────────────────────────────────────

def transcribe_wav(wav: bytes) -> str:
    try:
        from faster_whisper import WhisperModel
        import tempfile, os as _os
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav); tmp = f.name
        mdl = WhisperModel("tiny", device="cpu", compute_type="int8")
        segs, _ = mdl.transcribe(tmp, beam_size=1)
        _os.unlink(tmp)
        return " ".join(s.text for s in segs).strip()
    except ImportError: pass
    except Exception as e:
        print(f"  {Y}whisper err: {e}{RST}")
    try:
        import speech_recognition as sr
        r = sr.Recognizer()
        with sr.AudioFile(io.BytesIO(wav)) as src:
            audio = r.record(src)
        return r.recognize_google(audio)
    except Exception:
        return "[audio — STT unavailable]"


# ─────────────────────────────────────────────────────────────────────────────
# Camera capture
# ─────────────────────────────────────────────────────────────────────────────

class CameraCapture:
    def __init__(self):
        self._frames: deque = deque()
        self._lock = threading.Lock()
        self._running = False
        self._cap = None
        self.available = False
        try:
            import cv2; self._cv2 = cv2; self.available = True
        except ImportError:
            print(f"  {Y}opencv-python not available (pip install opencv-python){RST}")

    def start(self):
        if not self.available: return
        self._cap = self._cv2.VideoCapture(CAMERA_INDEX)
        if not self._cap.isOpened():
            print(f"  {Y}Camera {CAMERA_INDEX} not accessible{RST}")
            self.available = False; return
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()
        print(f"  {G}Camera ready (device={CAMERA_INDEX}){RST}")

    def _loop(self):
        interval = 1.0 / VIDEO_FPS
        max_f = int(VIDEO_BUFFER_MAXSEC * VIDEO_FPS)
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frames.append((time.time(), frame))
                    while len(self._frames) > max_f: self._frames.popleft()
            time.sleep(interval)

    def stop(self):
        self._running = False
        if self._cap:
            try: self._cap.release()
            except Exception: pass

    def drain_frames(self):
        with self._lock: f = list(self._frames); self._frames.clear()
        return f


def frames_to_jpegs(frames, max_f=8) -> List[bytes]:
    try:
        import cv2
    except ImportError: return []
    step = max(1, len(frames) // max_f)
    out = []
    for _, fr in frames[::step][:max_f]:
        ret, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ret: out.append(bytes(buf))
    return out


def caption_frame(jpeg: bytes, client, model: str) -> str:
    """Call vision API; return empty string on any failure."""
    try:
        import openai
        b64 = base64.b64encode(jpeg).decode()
        base = re.sub(r"/chat/completions/?$", "", str(client.base_url).rstrip("/"))
        sc = openai.OpenAI(api_key=client.api_key, base_url=base)
        r = sc.chat.completions.create(
            model=model, max_tokens=180,
            messages=[{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text",
                 "text": "In 1-2 sentences describe the objects, people, and spatial "
                         "context visible. Be factual and specific."},
            ]}])
        return r.choices[0].message.content.strip()
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Keyboard
# ─────────────────────────────────────────────────────────────────────────────

_key_q: queue.Queue = queue.Queue()
_term_saved = None

def _kb_thread():
    global _term_saved
    if os.name == "nt":
        import msvcrt
        while True:
            if msvcrt.kbhit(): _key_q.put(msvcrt.getwch())
            time.sleep(0.05)
    else:
        import tty, termios, select
        fd = sys.stdin.fileno()
        try:
            old = termios.tcgetattr(fd)
            _term_saved = (fd, old)
            tty.setraw(fd)
            while True:
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    _key_q.put(sys.stdin.read(1))
        except Exception: pass

def restore_term():
    global _term_saved
    if _term_saved and os.name != "nt":
        try:
            import termios
            fd, old = _term_saved
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
            _term_saved = None
        except Exception: pass

def start_kb():
    threading.Thread(target=_kb_thread, daemon=True).start()

def poll_key() -> Optional[str]:
    try: return _key_q.get_nowait()
    except queue.Empty: return None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

async def main(args):
    bar("ROBOT MEMORY  v3  —  LIVE DEMO")
    print(f"  {DIM}SPACE=record  ENTER=flush+consolidate  Q=quit{RST}\n")

    # DB + session
    bar("DATABASE")
    await rm.init_pool()
    session = await rm.TemporalSession.start()
    print(f"  Session: {B}{session.session_id[:18]}…{RST}")

    # Clean up stale raw nodes from any previous unfinished sessions
    # (observation nodes that accumulated without being consolidated)
    try:
        _pool = await _rmdb.get_pool()
        async with _pool.acquire() as _conn:
            _stale = await _conn.execute(
                """
                DELETE FROM raw_temporal_nodes
                WHERE data_type = 'observation'
                   OR (processed = TRUE)
                   OR (captured_at < NOW() - INTERVAL '24 hours')
                """
            )
            _n = int(str(_stale).split()[-1]) if _stale else 0
            if _n:
                print(f"  {Y}Cleaned up {_n} stale raw nodes from previous sessions{RST}")
    except Exception as _e:
        print(f"  {Y}Startup cleanup skipped: {_e}{RST}")

    # LLM URL sanitise
    _raw_url = os.environ.get("ROBOT_LLM_BASE_URL", "")
    if _raw_url and "/chat/completions" in _raw_url:
        _clean = re.sub(r"/chat/completions/?$", "", _raw_url.rstrip("/"))
        print(f"  {Y}ROBOT_LLM_BASE_URL corrected -> {_clean}{RST}")
        os.environ["ROBOT_LLM_BASE_URL"] = _clean
        _rmdb.LLM_BASE_URL = _clean

    from robot_memory.db import get_llm_client, get_llm_model
    llm_client = get_llm_client()
    llm_model  = get_llm_model()
    print(f"  {G}LLM: {llm_model}  ({os.environ.get('ROBOT_LLM_PROVIDER','?')}){RST}")

    # Location
    bar("LOCATION")
    if args.lat is not None and args.lon is not None:
        origin_lat, origin_lon = args.lat, args.lon
    else:
        origin_lat, origin_lon = get_ip_location()

    floor_level = args.floor

    # Resume from last DB position
    last = await resume_last_position()
    if last:
        ox, oy, oz, last_fl = last
        if args.floor == 0:
            floor_level = last_fl
        print(f"  {G}Resumed: ({ox:.2f}, {oy:.2f}, {oz:.2f})  floor={floor_level}{RST}")
    else:
        ox, oy = 0.0, 0.0
        oz = floor_level * 3.0
        print(f"  {Y}No previous path — starting at origin{RST}")

    # Sensors
    bar("SENSORS")
    audio = AudioCapture()
    cam   = CameraCapture()
    if not args.no_audio:  audio.start()
    if not args.no_camera: cam.start()

    start_kb()
    _main_loop = asyncio.get_event_loop()

    recording     = False
    rec_start     = 0.0
    path_step     = 0
    total_raw     = 0
    run           = True
    last_pos_t    = 0.0
    last_amb_t    = time.time()

    bar("LIVE  —  SPACE=record  ENTER=consolidate  Q=quit")
    print()

    try:
        while run:
            now = time.time()

            # Ambient audio while idle
            if (not recording and not args.no_audio and audio.available
                    and now - last_amb_t >= AMBIENT_INTERVAL):
                last_amb_t = now
                wav = audio.drain_wav()
                if wav:
                    px, py, pz, hdg = get_position(path_step, ox, oy, oz)
                    def _ambient(w, sess, _px, _py, _pz, _fl, _hdg, _loop):
                        txt = transcribe_wav(w)
                        if txt and "STT unavailable" not in txt and len(txt.strip()) > 4:
                            asyncio.run_coroutine_threadsafe(
                                sess.add_raw_node(
                                    data_type="conversation",
                                    raw_text=txt,
                                    raw_json={"source": "ambient"},
                                    x=_px, y=_py, z=_pz,
                                    floor_level=_fl, heading_deg=_hdg,
                                ), _loop
                            ).result(timeout=10)
                    threading.Thread(
                        target=_ambient,
                        args=(wav, session, px, py, pz, floor_level, hdg, _main_loop),
                        daemon=True,
                    ).start()

            # Position tick
            if now - last_pos_t >= POSITION_INTERVAL:
                px, py, pz, hdg = get_position(path_step, ox, oy, oz)
                await rm.record_position(
                    px, py, pz,
                    floor_level=floor_level,
                    heading_deg=hdg,
                    session_id=session.session_id,
                )
                last_pos_t = now; path_step += 1
                ind = f"{R}REC{RST}" if recording else f"{DIM}idle{RST}"
                print(f"\r  [{ind}]  ({px:.1f},{py:.1f},{pz:.1f})"
                      f"  hdg={hdg:.0f}  raw={Y}{total_raw}{RST}  pts={path_step}   ",
                      end="", flush=True)

            # Keys
            key = poll_key()
            if key is not None:
                kl = key.lower()

                if key == " ":
                    if not recording:
                        recording = True; rec_start = time.time()
                        print(f"\n  {R}{BOLD}Recording…{RST}")
                        audio.drain_wav()
                        cam.drain_frames()
                    else:
                        recording = False
                        elapsed = time.time() - rec_start
                        print(f"\n  {G}Stopped ({elapsed:.1f}s) — saving…{RST}")
                        px, py, pz, hdg = get_position(path_step, ox, oy, oz)

                        # Audio transcript
                        audio_raw_id: Optional[str] = None
                        if not args.no_audio and audio.available:
                            wav = audio.drain_wav()
                            if wav:
                                print(f"    {C}Transcribing…{RST}", end="", flush=True)
                                transcript = transcribe_wav(wav)
                                print(f"\r    {G}Transcript: {W}{transcript[:90]}{RST}")
                                audio_raw_id = await session.add_raw_node(
                                    data_type="audio_transcript",
                                    raw_text=transcript,
                                    raw_json={
                                        "duration_sec": round(elapsed, 2),
                                        "sample_rate": SAMPLE_RATE,
                                        "wav_b64": base64.b64encode(wav).decode()
                                            if len(wav) < MAX_MEDIA_BYTES else None,
                                    },
                                    x=px, y=py, z=pz,
                                    floor_level=floor_level, heading_deg=hdg,
                                )
                                total_raw += 1
                                print(f"    {DIM}audio={audio_raw_id[:8]}…{RST}")

                        # Video frame
                        if not args.no_camera and cam.available:
                            frames = cam.drain_frames()
                            if frames:
                                jpegs = frames_to_jpegs(frames)
                                cap_text = ""
                                if jpegs:
                                    print(f"    {C}Captioning…{RST}", end="", flush=True)
                                    cap_text = caption_frame(jpegs[-1], llm_client, llm_model)
                                    if not cap_text:
                                        cap_text = (
                                            f"Camera at {datetime.now():%H:%M:%S}: "
                                            f"{len(frames)} frames at "
                                            f"({px:.1f},{py:.1f},{pz:.1f})"
                                        )
                                    print(f"\r    {G}Caption: {W}{cap_text[:90]}{RST}")

                                vid_raw_id = await session.add_raw_node(
                                    data_type="video_frame",
                                    raw_text=cap_text,
                                    raw_json={
                                        "frame_count": len(frames),
                                        "duration_sec": round(elapsed, 2),
                                        "fps": VIDEO_FPS,
                                        "jpeg_b64": base64.b64encode(jpegs[-1]).decode()
                                            if jpegs and len(jpegs[-1]) < MAX_MEDIA_BYTES else None,
                                        "audio_raw_id": audio_raw_id,
                                    },
                                    x=px, y=py, z=pz,
                                    floor_level=floor_level, heading_deg=hdg,
                                )
                                total_raw += 1
                                print(f"    {DIM}video={vid_raw_id[:8]}…{RST}")

                elif key in ("\r", "\n"):
                    print(f"\n\n  {M}{BOLD}ENTER — flushing + consolidating…{RST}\n")
                    run = False

                elif kl in ("q", "\x03"):
                    restore_term()
                    print(f"\n\n  {Y}Quit.{RST}")
                    audio.stop(); cam.stop()
                    await rm.close_pool(); return

            await asyncio.sleep(0.05)

    except KeyboardInterrupt:
        restore_term()
        print(f"\n  {Y}Interrupted.{RST}")

    restore_term()
    audio.stop(); cam.stop()

    # Flush path
    bar("FLUSH PATH")
    nn, ne = await flush_path_to_db(session_id=None)
    print(f"  {G}{nn} nodes  {ne} edges committed{RST}")

    await session.dump_summary()

    # Consolidate
    bar("LLM CONSOLIDATION")
    result = await rm.flush_and_consolidate(
        session_id=session.session_id, session=session, verbose=True)

    sc = G if result.status == "done" else Y
    print(f"\n  {sc}{result.status}{RST}  +{result.entities_created} entities"
          f"  +{result.edges_created} edges  +{result.info_nodes_created} info"
          f"  {result.llm_calls} LLM calls")
    if result.error_msg:
        print(f"  {R}{result.error_msg}{RST}")

    # Anchor
    bar("ANCHOR")
    dctx = await rm.deep_think()
    ok = 0
    for ent in dctx.all_entities:
        try: await rm.anchor_entity_to_map(ent.node_id); ok += 1
        except Exception: pass
    print(f"  {G}{ok}/{len(dctx.all_entities)} entities anchored{RST}")

    # World map
    bar("WORLD MAP")
    px, py, pz, _ = get_position(path_step, ox, oy, oz)
    ctx = await rm.think(robot_x=px, robot_y=py, robot_z=pz,
                         floor_level=floor_level, radius_m=100.0)
    print(ctx.summary())
    print(f"\n  {dctx.summary()}")
    for fl in sorted(set(e.floor_level for e in dctx.all_entities)):
        print(f"  floor {fl}: {[e.name for e in dctx.entities_on_floor(fl)]}")

    bar("DONE")
    print(f"  {G}Done. Run again to continue building.{RST}\n")
    await rm.close_pool()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Robot Memory Live Demo v3")
    ap.add_argument("--lat",       type=float, default=None)
    ap.add_argument("--lon",       type=float, default=None)
    ap.add_argument("--floor",     type=int,   default=0)
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--no-audio",  action="store_true")
    args = ap.parse_args()
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print(f"\n{Y}Interrupted.{RST}")