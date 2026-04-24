"""
live_demo.py — Robot Memory Live Sensor Demo
═════════════════════════════════════════════
Uses your laptop's camera, microphone, and GPS/IP location to build a live
spatial knowledge graph using the robot_memory library.

Controls
────────
  SPACE  — Toggle video+audio recording (press once to start, again to stop).
           Each recording chunk is stored as a raw_temporal_node.
  ENTER  — Stop all streams, flush path to DB, run LLM consolidation,
           anchor entities, and display the resulting world map.
  Q / Ctrl-C — Quit immediately without consolidating.

What it does
────────────
  1. On startup: estimates location via IP geolocation (lat/lon → local x/y metres)
     or from --lat/--lon CLI args for precision.
  2. Every POSITION_INTERVAL seconds: calls record_position() to log the
     robot's (laptop's) current estimated position, heading and floor.
  3. Microphone audio is continuously buffered. When SPACE is pressed the
     current buffer (and ongoing capture) is saved as an audio_transcript
     raw_temporal_node (via speech-to-text using faster-whisper or SpeechRecognition).
  4. Camera frames are captured. When SPACE is pressed a rolling window of
     frames is encoded to an MP4 memory buffer and stored as a video_frame
     raw_temporal_node with a VLM caption (via OpenCV + the Groq vision API
     or a fallback local description).
  5. All raw data accumulates. On ENTER: flush_path_to_db() then
     flush_and_consolidate() runs the LLM pipeline → entity graph + edges.
  6. Prints a ThoughtContext summary centred on the robot's final position.

Dependencies (beyond requirements.txt)
──────────────────────────────────────
  pip install opencv-python sounddevice SpeechRecognition numpy requests

Optional (for better STT):
  pip install faster-whisper   # local Whisper STT (recommended)

Run
───
  python live_demo.py                         # IP-based location
  python live_demo.py --lat 22.57 --lon 88.37 # precise GPS
  python live_demo.py --floor 2               # specify floor
  python live_demo.py --no-camera             # audio + location only
  python live_demo.py --no-audio              # camera + location only
"""

from __future__ import annotations
import argparse
import asyncio
import io
import json
import math
import os
import queue
import sys
import threading
import time
import wave
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# ── Adjust import path so robot_memory is importable ─────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import robot_memory as rm
from robot_memory.pathmap import flush_path_to_db

# ─────────────────────────────────────────────────────────────────────────────
# Tunables
# ─────────────────────────────────────────────────────────────────────────────
POSITION_INTERVAL   = 2.0    # seconds between position updates
AUDIO_CHUNK_SEC     = 0.5    # size of each audio buffer chunk in seconds
AUDIO_BUFFER_MAXSEC = 120    # max audio history to keep in RAM (seconds)
VIDEO_FPS           = 4      # frames per second captured
VIDEO_BUFFER_MAXSEC = 30     # rolling video frame buffer size (seconds)
CAMERA_INDEX        = 0      # OpenCV camera device index
SAMPLE_RATE         = 16000  # audio sample rate (Hz)
CHANNELS            = 1      # mono audio


# ─────────────────────────────────────────────────────────────────────────────
# Colours for terminal output
# ─────────────────────────────────────────────────────────────────────────────
R  = "\033[91m"; G  = "\033[92m"; Y  = "\033[93m"
B  = "\033[94m"; M  = "\033[95m"; C  = "\033[96m"
W  = "\033[97m"; DIM = "\033[2m"; RST = "\033[0m"
BOLD = "\033[1m"

def bar(title="", width=72):
    if title:
        pad = width - len(title) - 4
        print(f"\n{BOLD}{C}{'━'*2}  {title}  {'━'*pad}{RST}")
    else:
        print(f"{DIM}{'─'*width}{RST}")


# ─────────────────────────────────────────────────────────────────────────────
# Location helper
# ─────────────────────────────────────────────────────────────────────────────

def get_ip_location() -> Tuple[float, float, float]:
    """
    Estimate (lat, lon, accuracy_m) from IP geolocation.
    Returns (0.0, 0.0, 10000.0) on failure.
    """
    try:
        import requests
        r = requests.get("https://ipapi.co/json/", timeout=5)
        d = r.json()
        lat = float(d.get("latitude", 0))
        lon = float(d.get("longitude", 0))
        print(f"  {G}✓ IP location: {lat:.5f}, {lon:.5f}  "
              f"({d.get('city','?')}, {d.get('country_name','?')}){RST}")
        return lat, lon, 500.0   # ~500m accuracy for IP
    except Exception as e:
        print(f"  {Y}⚠ IP location failed ({e}) — using (0,0){RST}")
        return 0.0, 0.0, 10000.0


def latlon_to_metres(lat: float, lon: float,
                     origin_lat: float, origin_lon: float) -> Tuple[float, float]:
    """
    Approximate conversion of lat/lon to local x/y metres relative to origin.
    Good enough for building-scale mapping.
    """
    R_EARTH = 6_371_000.0  # metres
    dy = math.radians(lat - origin_lat) * R_EARTH
    dx = math.radians(lon - origin_lon) * R_EARTH * math.cos(math.radians(origin_lat))
    return dx, dy   # (east, north) in metres


def get_compass_heading() -> Optional[float]:
    """
    Attempt to read compass heading.
    On most laptops this is unavailable — returns None (robot faces North).
    """
    return None  # extend with platform-specific sensor reads if available


# ─────────────────────────────────────────────────────────────────────────────
# Runtime embedding (random 128-d unit vector — replace with real encoder)
# ─────────────────────────────────────────────────────────────────────────────
import random as _random

def make_embedding(text: str, dim: int = 128) -> List[float]:
    """
    Cheap deterministic embedding from text hash.
    Replace with a real sentence encoder (e.g. sentence-transformers) for
    meaningful similarity search.
    """
    seed = hash(text) % (2**32)
    rng  = _random.Random(seed)
    vec  = [rng.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(v*v for v in vec)) or 1.0
    return [v/norm for v in vec]


# ─────────────────────────────────────────────────────────────────────────────
# Audio capture
# ─────────────────────────────────────────────────────────────────────────────

class AudioCapture:
    def __init__(self, sample_rate: int = SAMPLE_RATE, channels: int = CHANNELS):
        self.sample_rate = sample_rate
        self.channels    = channels
        self._buffer: deque = deque()  # deque of numpy arrays
        self._buffer_lock   = threading.Lock()
        self._recording     = False
        self._stream        = None
        self._available     = False
        self._init()

    def _init(self):
        try:
            import sounddevice as sd
            import numpy as np
            self._sd = sd
            self._np = np
            self._available = True
        except ImportError:
            print(f"  {Y}⚠ sounddevice not installed — audio capture disabled.{RST}")
            print(f"    Install with: pip install sounddevice")

    def start(self):
        if not self._available:
            return
        import sounddevice as sd
        import numpy as np
        chunk_frames = int(self.sample_rate * AUDIO_CHUNK_SEC)

        def callback(indata, frames, time_info, status):
            if self._recording:
                with self._buffer_lock:
                    self._buffer.append(indata.copy())
                    # trim old audio
                    max_chunks = int(AUDIO_BUFFER_MAXSEC / AUDIO_CHUNK_SEC)
                    while len(self._buffer) > max_chunks:
                        self._buffer.popleft()

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=chunk_frames,
            callback=callback,
        )
        self._stream.start()
        self._recording = True
        print(f"  {G}✓ Microphone started (SR={self.sample_rate}Hz){RST}")

    def stop(self):
        self._recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()

    def drain_wav_bytes(self) -> Optional[bytes]:
        """
        Drain the audio buffer and return raw WAV bytes.
        Returns None if buffer is empty or audio not available.
        """
        if not self._available:
            return None
        import numpy as np
        with self._buffer_lock:
            if not self._buffer:
                return None
            frames = np.concatenate(list(self._buffer), axis=0)
            self._buffer.clear()

        # Convert float32 → int16 PCM
        pcm = (frames * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # int16 = 2 bytes
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm.tobytes())
        return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Speech-to-text
# ─────────────────────────────────────────────────────────────────────────────

def transcribe_wav(wav_bytes: bytes) -> str:
    """
    Transcribe WAV bytes to text.
    Tries faster-whisper first (local), falls back to SpeechRecognition (Google).
    Returns "" if both unavailable.
    """
    # Try faster-whisper (local, no API key)
    try:
        from faster_whisper import WhisperModel
        import io, wave, tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            tmp = f.name
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(tmp, beam_size=1)
        os.unlink(tmp)
        return " ".join(s.text for s in segments).strip()
    except ImportError:
        pass
    except Exception as e:
        print(f"  {Y}⚠ faster-whisper error: {e}{RST}")

    # Try SpeechRecognition (Google free API)
    try:
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio)
    except ImportError:
        print(f"  {Y}⚠ SpeechRecognition not installed.{RST}")
        print(f"    Install: pip install SpeechRecognition  (or faster-whisper)")
        return "[audio captured — STT unavailable]"
    except Exception as e:
        return f"[STT failed: {e}]"


# ─────────────────────────────────────────────────────────────────────────────
# Camera capture
# ─────────────────────────────────────────────────────────────────────────────

class CameraCapture:
    def __init__(self, device: int = CAMERA_INDEX, fps: int = VIDEO_FPS):
        self.device        = device
        self.fps           = fps
        self._frames: deque = deque()   # deque of (timestamp, numpy frame)
        self._lock         = threading.Lock()
        self._running      = False
        self._thread       = None
        self._cap          = None
        self._available    = False
        self._init()

    def _init(self):
        try:
            import cv2
            self._cv2 = cv2
            self._available = True
        except ImportError:
            print(f"  {Y}⚠ opencv-python not installed — camera capture disabled.{RST}")
            print(f"    Install with: pip install opencv-python")

    def start(self):
        if not self._available:
            return
        import cv2
        self._cap = cv2.VideoCapture(self.device)
        if not self._cap.isOpened():
            print(f"  {Y}⚠ Camera device {self.device} not accessible.{RST}")
            self._available = False
            return
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"  {G}✓ Camera started (device={self.device}, ~{self.fps}fps){RST}")

    def _loop(self):
        import cv2
        interval = 1.0 / self.fps
        max_frames = int(VIDEO_BUFFER_MAXSEC * self.fps)
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frames.append((time.time(), frame))
                    while len(self._frames) > max_frames:
                        self._frames.popleft()
            time.sleep(interval)

    def stop(self):
        self._running = False
        if self._cap:
            self._cap.release()

    def drain_frames(self) -> List[Tuple[float, "np.ndarray"]]:
        """Drain and return all buffered frames."""
        with self._lock:
            frames = list(self._frames)
            self._frames.clear()
        return frames

    def capture_snapshot(self) -> Optional["np.ndarray"]:
        """Return the most recent frame without draining."""
        with self._lock:
            if self._frames:
                return self._frames[-1][1]
        return None


def frames_to_jpeg_list(frames, max_frames: int = 10) -> List[bytes]:
    """Subsample and encode frames to JPEG bytes."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return []

    if not frames:
        return []

    step = max(1, len(frames) // max_frames)
    selected = [f for _, f in frames[::step]][:max_frames]
    jpegs = []
    for frame in selected:
        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ret:
            jpegs.append(bytes(buf))
    return jpegs


def describe_frames_simple(frames_data, x: float, y: float, z: float) -> str:
    """
    Simple frame description without VLM API.
    Returns a text description based on frame count and timestamp.
    """
    if not frames_data:
        return "No frames captured."
    n = len(frames_data)
    ts = datetime.now().strftime("%H:%M:%S")
    return (
        f"Video recording at {ts}: {n} frames captured at position "
        f"({x:.1f}, {y:.1f}, {z:.1f}). "
        f"Duration ~{n / VIDEO_FPS:.1f}s. Camera active."
    )


def describe_frames_with_groq(jpeg_bytes: bytes, client, model: str) -> str:
    """
    Use Groq/OpenAI vision API to describe a camera frame.
    Falls back silently (returns "") on any error.

    Note: base_url is extracted from the async client and stripped of any
    trailing /chat/completions path so the sync openai client doesn't double-append it.
    """
    import base64, re as _re
    b64 = base64.b64encode(jpeg_bytes).decode()
    try:
        import openai
        # client.base_url is an httpx.URL — stringify it and strip trailing endpoint
        raw_base = str(client.base_url).rstrip("/")
        clean_base = _re.sub(r"/chat/completions$", "", raw_base)
        sync_client = openai.OpenAI(
            api_key=client.api_key,
            base_url=clean_base,
        )
        resp = sync_client.chat.completions.create(
            model=model,
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text",
                     "text": "Describe what you see in this image in 1-2 sentences. "
                             "Focus on objects, people, text, and spatial context."},
                ],
            }],
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Keyboard (non-blocking cross-platform)
# ─────────────────────────────────────────────────────────────────────────────

_key_queue: queue.Queue = queue.Queue()
_terminal_old_settings = None   # saved before raw mode


def _keyboard_thread():
    """Read keys in a background thread and push to queue."""
    global _terminal_old_settings
    if os.name == "nt":
        # Windows
        import msvcrt
        while True:
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                _key_queue.put(ch)
            time.sleep(0.05)
    else:
        # Unix/macOS — raw terminal
        import tty, termios, select
        fd = sys.stdin.fileno()
        try:
            old = termios.tcgetattr(fd)
            _terminal_old_settings = (fd, old)
            tty.setraw(fd)
            while True:
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    ch = sys.stdin.read(1)
                    _key_queue.put(ch)
        except Exception:
            pass


def restore_terminal():
    """Restore terminal to cooked mode. Call before printing final output."""
    global _terminal_old_settings
    if _terminal_old_settings is not None and os.name != "nt":
        try:
            import termios
            fd, old = _terminal_old_settings
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
            _terminal_old_settings = None
        except Exception:
            pass


def start_keyboard_listener():
    t = threading.Thread(target=_keyboard_thread, daemon=True)
    t.start()


def poll_key() -> Optional[str]:
    try:
        return _key_queue.get_nowait()
    except queue.Empty:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main demo
# ─────────────────────────────────────────────────────────────────────────────

async def main(args):
    bar("ROBOT MEMORY — LIVE SENSOR DEMO")
    print(f"  {DIM}Controls:  SPACE=record  ENTER=flush+consolidate  Q=quit{RST}\n")

    # ── 1. Init DB ────────────────────────────────────────────────────────────
    bar("DATABASE INIT")
    await rm.init_pool()
    session = await rm.TemporalSession.start()
    print(f"  Session ID: {B}{session.session_id}{RST}")

    # ── 2. Resolve location ───────────────────────────────────────────────────
    bar("LOCATION")
    if args.lat is not None and args.lon is not None:
        origin_lat, origin_lon = args.lat, args.lon
        print(f"  {G}✓ Using provided GPS: {origin_lat:.6f}, {origin_lon:.6f}{RST}")
    else:
        origin_lat, origin_lon, _acc = get_ip_location()

    floor_level = args.floor
    z_metres    = floor_level * 3.0   # assume 3m per floor

    # Robot always starts at origin (0, 0) in local frame
    robot_x, robot_y = 0.0, 0.0
    robot_z          = z_metres

    print(f"  Local frame origin = lat={origin_lat:.5f} lon={origin_lon:.5f}")
    print(f"  Starting position  = ({robot_x:.1f}, {robot_y:.1f}, {robot_z:.1f})  "
          f"floor={floor_level}")

    # ── 3. Start sensors ──────────────────────────────────────────────────────
    bar("SENSOR INIT")
    audio_cap  = AudioCapture()
    camera_cap = CameraCapture()

    if not args.no_audio:
        audio_cap.start()
    if not args.no_camera:
        camera_cap.start()

    # ── 4. LLM client (for optional VLM captions) ─────────────────────────────
    # Sanitise ROBOT_LLM_BASE_URL before the openai client reads it.
    # A common mistake: .env points to the full endpoint
    #   https://api.groq.com/openai/v1/chat/completions
    # but the openai SDK appends /chat/completions itself, producing a 404.
    # Strip the trailing endpoint path so only the base /v1 remains.
    _raw_url = os.environ.get("ROBOT_LLM_BASE_URL", "")
    if _raw_url and "/chat/completions" in _raw_url:
        import re as _re
        _clean_url = _re.sub(r"/chat/completions/?$", "", _raw_url.rstrip("/"))
        print(f"  {Y}⚠  ROBOT_LLM_BASE_URL auto-corrected (removed trailing endpoint):{RST}")
        print(f"      was : {_raw_url}")
        print(f"      now : {_clean_url}")
        os.environ["ROBOT_LLM_BASE_URL"] = _clean_url
        import robot_memory.db as _rmdb
        _rmdb.LLM_BASE_URL = _clean_url   # patch already-loaded module constant

    from robot_memory.db import get_llm_client, get_llm_model
    llm_client = get_llm_client()
    llm_model  = get_llm_model()
    print(f"  {G}✓ LLM ready: {llm_model}  (provider={os.environ.get('ROBOT_LLM_PROVIDER','?')}){RST}")

    # ── 5. Keyboard ───────────────────────────────────────────────────────────
    start_keyboard_listener()

    # ── 6. State ──────────────────────────────────────────────────────────────
    recording       = False       # SPACE toggle state
    recording_start = 0.0
    total_raw_nodes = 0
    path_step       = 0
    run             = True

    # Simulated slow drift: laptop "moves" in small circles to demo path map
    # In a real robot this would be replaced by odometry / SLAM output.
    def get_simulated_position(step: int) -> Tuple[float, float, float, float]:
        """Slowly drift in a small oval so the path map has some structure."""
        t      = step * POSITION_INTERVAL
        radius = 2.0   # metres
        speed  = 0.05  # radians/second
        x = robot_x + radius * math.cos(speed * t)
        y = robot_y + radius * math.sin(speed * t) * 0.5
        z = robot_z
        heading = math.degrees(speed * t + math.pi / 2) % 360
        return x, y, z, heading

    # ── 7. Ambient conversation logger ───────────────────────────────────────
    # Every AMBIENT_INTERVAL seconds, if NOT in manual recording mode, drain
    # a small audio slice and store it as a 'conversation' raw_temporal_node.
    AMBIENT_INTERVAL  = 15.0   # seconds between passive audio captures
    last_ambient_time = time.time()
    # Capture the running event loop NOW (in the main coroutine/thread).
    # Background threads must NOT call asyncio.get_event_loop() — it raises
    # RuntimeError on Python 3.10+ when called from a non-async thread.
    _main_loop = asyncio.get_event_loop()

    bar("LIVE CAPTURE  —  press SPACE to record, ENTER to consolidate, Q to quit")
    print()

    last_position_time = 0.0
    tick = 0

    try:
        while run:
            now = time.time()

            # ── Ambient audio capture (passive, between manual recordings) ────
            if (not recording
                    and not args.no_audio
                    and audio_cap._available
                    and now - last_ambient_time >= AMBIENT_INTERVAL):
                last_ambient_time = now
                wav = audio_cap.drain_wav_bytes()
                if wav:
                    px_a, py_a, pz_a, hdg_a = get_simulated_position(path_step)

                    # Pass _main_loop explicitly — never call get_event_loop()
                    # inside a background thread.
                    def _run_ambient(w, sess, _px, _py, _pz, _fl, _hdg, _loop):
                        txt = transcribe_wav(w)
                        if txt and "[STT" not in txt and "[audio" not in txt:
                            future = asyncio.run_coroutine_threadsafe(
                                sess.add_raw_node(
                                    data_type="conversation",
                                    raw_text=txt,
                                    raw_json={"source": "ambient"},
                                    x=_px, y=_py, z=_pz,
                                    floor_level=_fl,
                                    heading_deg=_hdg,
                                ),
                                _loop,   # ← the captured loop, not get_event_loop()
                            )
                            try:
                                future.result(timeout=10)
                            except Exception:
                                pass

                    t_amb = threading.Thread(
                        target=_run_ambient,
                        args=(wav, session, px_a, py_a, pz_a,
                              floor_level, hdg_a, _main_loop),
                        daemon=True,
                    )
                    t_amb.start()

            # ── Position update ───────────────────────────────────────────────
            if now - last_position_time >= POSITION_INTERVAL:
                px, py, pz, heading = get_simulated_position(path_step)
                rt_id = await rm.record_position(
                    px, py, pz,
                    floor_level=floor_level,
                    heading_deg=heading,
                    session_id=session.session_id,
                )
                last_position_time = now
                path_step += 1

                # Log position as observation
                await session.add_raw_node(
                    data_type="observation",
                    raw_text=f"Robot at ({px:.2f}, {py:.2f}, {pz:.2f}) "
                             f"heading={heading:.0f}°  floor={floor_level}",
                    x=px, y=py, z=pz,
                    floor_level=floor_level,
                    heading_deg=heading,
                )
                total_raw_nodes += 1

                # Status line
                rec_indicator = f"{R}● REC{RST}" if recording else f"{DIM}○ idle{RST}"
                print(
                    f"\r  {rec_indicator}  "
                    f"pos=({px:.1f},{py:.1f},{pz:.1f})  hdg={heading:.0f}°  "
                    f"raw_nodes={Y}{total_raw_nodes}{RST}  "
                    f"path_pts={path_step}  "
                    f"t={now:.0f}          ",
                    end="", flush=True
                )

            # ── Key handling ──────────────────────────────────────────────────
            key = poll_key()
            if key is not None:
                key_lower = key.lower()

                # SPACE — toggle recording
                if key in (" ", ):
                    if not recording:
                        recording       = True
                        recording_start = time.time()
                        print(f"\n  {R}{BOLD}● Recording started …{RST}")
                        # Clear stale buffers so we get fresh data
                        audio_cap.drain_wav_bytes()
                        camera_cap.drain_frames()
                    else:
                        recording = False
                        elapsed   = time.time() - recording_start
                        print(f"\n  {G}■ Recording stopped ({elapsed:.1f}s).  Processing …{RST}")

                        # Current position
                        px, py, pz, heading = get_simulated_position(path_step)

                        # ── Audio → raw_temporal_node ─────────────────────────
                        if not args.no_audio and audio_cap._available:
                            wav = audio_cap.drain_wav_bytes()
                            if wav:
                                print(f"    {C}Transcribing audio …{RST}", end="", flush=True)
                                transcript = transcribe_wav(wav)
                                print(f"\r    {G}✓ Transcript: {W}{transcript[:80]}{RST}")
                                rid = await session.add_raw_node(
                                    data_type="audio_transcript",
                                    raw_text=transcript,
                                    raw_json={"duration_sec": elapsed,
                                              "sample_rate": SAMPLE_RATE},
                                    x=px, y=py, z=pz,
                                    floor_level=floor_level,
                                    heading_deg=heading,
                                )
                                total_raw_nodes += 1
                                print(f"    {DIM}audio raw_id={rid[:8]}…{RST}")

                        # ── Camera → raw_temporal_node ────────────────────────
                        if not args.no_camera and camera_cap._available:
                            frames = camera_cap.drain_frames()
                            if frames:
                                jpegs   = frames_to_jpeg_list(frames, max_frames=8)
                                # Try VLM caption on the last frame
                                caption = ""
                                if jpegs:
                                    print(f"    {C}Captioning frame …{RST}", end="", flush=True)
                                    caption = describe_frames_with_groq(
                                        jpegs[-1], llm_client, llm_model
                                    )
                                    if not caption:
                                        caption = describe_frames_simple(frames, px, py, pz)
                                    print(f"\r    {G}✓ Caption: {W}{caption[:80]}{RST}")

                                rid = await session.add_raw_node(
                                    data_type="video_frame",
                                    raw_text=caption,
                                    raw_json={
                                        "frame_count": len(frames),
                                        "duration_sec": elapsed,
                                        "fps": VIDEO_FPS,
                                        "resolution": "camera",
                                    },
                                    x=px, y=py, z=pz,
                                    floor_level=floor_level,
                                    heading_deg=heading,
                                )
                                total_raw_nodes += 1
                                print(f"    {DIM}video raw_id={rid[:8]}…{RST}")

                # ENTER — flush + consolidate
                elif key in ("\r", "\n"):
                    print(f"\n\n  {M}{BOLD}ENTER pressed — stopping and consolidating …{RST}\n")
                    run = False

                # Q — quit
                elif key_lower in ("q", "\x03"):
                    restore_terminal()
                    print(f"\n\n  {Y}Quitting without consolidation.{RST}")
                    audio_cap.stop()
                    camera_cap.stop()
                    await rm.close_pool()
                    return

            await asyncio.sleep(0.05)

    except KeyboardInterrupt:
        restore_terminal()
        print(f"\n\n  {Y}Interrupted.{RST}")

    # ── Restore terminal to cooked mode before any multi-line output ──────────
    restore_terminal()

    # ── Stop sensors ──────────────────────────────────────────────────────────
    print(f"\n  {DIM}Stopping sensors …{RST}")
    audio_cap.stop()
    camera_cap.stop()

    # ── Capture final audio/video burst ──────────────────────────────────────
    if recording:
        print(f"  {Y}Recording was still active — saving final chunk …{RST}")
        px, py, pz, heading = get_simulated_position(path_step)
        if not args.no_audio and audio_cap._available:
            wav = audio_cap.drain_wav_bytes()
            if wav:
                transcript = transcribe_wav(wav)
                await session.add_raw_node(
                    data_type="audio_transcript",
                    raw_text=transcript,
                    x=px, y=py, z=pz,
                    floor_level=floor_level, heading_deg=heading,
                )
                total_raw_nodes += 1

    # ── Flush path to DB ──────────────────────────────────────────────────────
    bar("FLUSH PATH TO DB")
    n_nodes, n_edges = await flush_path_to_db(session_id=None)
    print(f"  {G}✓ Committed {n_nodes} path nodes  {n_edges} path edges{RST}")

    # ── Temporal summary ──────────────────────────────────────────────────────
    bar("TEMPORAL SESSION SUMMARY")
    await session.dump_summary()

    # ── LLM Consolidation ─────────────────────────────────────────────────────
    bar("LLM CONSOLIDATION  (raw nodes → entity graph)")
    print(f"  {C}Total raw nodes to consolidate: {total_raw_nodes}{RST}")
    print(f"  {DIM}Running LLM pipeline … (this may take a few seconds){RST}\n")

    result = await rm.flush_and_consolidate(
        session_id=session.session_id,
        session=session,
        verbose=True,
    )

    bar("CONSOLIDATION RESULT")
    status_color = G if result.status == "done" else Y
    print(f"  Status          : {status_color}{result.status}{RST}")
    print(f"  Raw nodes       : {result.raw_nodes_processed} processed")
    print(f"  Entities created: {G}{result.entities_created}{RST}")
    print(f"  Entities updated: {result.entities_updated}")
    print(f"  Edges created   : {G}{result.edges_created}{RST}")
    print(f"  Info nodes      : {result.info_nodes_created}")
    print(f"  LLM calls       : {result.llm_calls}  (model={result.llm_model})")
    if result.summary_text:
        print(f"\n  {W}Summary:{RST}")
        for line in result.summary_text.split(" | "):
            if line.strip():
                print(f"    {line.strip()}")
    if result.error_msg:
        print(f"\n  {R}Error: {result.error_msg}{RST}")

    # ── Anchor entities to map ────────────────────────────────────────────────
    bar("ANCHOR ENTITIES TO MAP")
    try:
        dctx = await rm.deep_think()
        anchored = 0
        for ent in dctx.all_entities:
            try:
                await rm.anchor_entity_to_map(ent.node_id)
                anchored += 1
            except (ValueError, Exception):
                pass
        print(f"  {G}✓ Anchored {anchored}/{len(dctx.all_entities)} entities{RST}")
    except Exception as e:
        print(f"  {Y}⚠ Anchoring skipped: {e}{RST}")

    # ── World map snapshot ────────────────────────────────────────────────────
    bar("WORLD MAP SNAPSHOT")
    try:
        px, py, pz, _ = get_simulated_position(path_step)
        ctx = await rm.think(
            robot_x=px, robot_y=py, robot_z=pz,
            floor_level=floor_level,
            radius_m=50.0,   # wide radius to capture everything
        )
        print(ctx.summary())
    except Exception as e:
        print(f"  {Y}⚠ think() failed: {e}{RST}")

    # ── Deep graph summary ────────────────────────────────────────────────────
    bar("FULL KNOWLEDGE GRAPH")
    try:
        dctx = await rm.deep_think()
        print(f"  {dctx.summary()}")
        floors = sorted(set(e.floor_level for e in dctx.all_entities))
        for fl in floors:
            ents = dctx.entities_on_floor(fl)
            names = [e.name for e in ents]
            print(f"  Floor {fl}: {names}")

        if dctx.all_edges:
            print(f"\n  {W}Relationships:{RST}")
            for ed in dctx.all_edges[:20]:
                n1 = next((e.name for e in dctx.all_entities if e.node_id == ed.node_id_1), ed.node_id_1[:8])
                n2 = next((e.name for e in dctx.all_entities if e.node_id == ed.node_id_2), ed.node_id_2[:8])
                print(f"  {DIM}{n1}{RST} ─[{C}{ed.rel_type}{RST}]→ {DIM}{n2}{RST}")
    except Exception as e:
        print(f"  {Y}⚠ deep_think() failed: {e}{RST}")

    bar("DONE")
    print(f"  {G}World map built successfully.{RST}")
    print(f"  {DIM}Run again to continue building on existing graph.{RST}\n")
    await rm.close_pool()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Robot Memory Live Demo — builds world map from laptop sensors"
    )
    parser.add_argument("--lat",      type=float, default=None,
                        help="Precise latitude (overrides IP geolocation)")
    parser.add_argument("--lon",      type=float, default=None,
                        help="Precise longitude (overrides IP geolocation)")
    parser.add_argument("--floor",    type=int,   default=0,
                        help="Current floor level (default: 0)")
    parser.add_argument("--no-camera",action="store_true",
                        help="Disable camera capture")
    parser.add_argument("--no-audio", action="store_true",
                        help="Disable microphone capture")
    args = parser.parse_args()

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print(f"\n{Y}Interrupted.{RST}")