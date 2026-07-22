#!/usr/bin/env python3
"""Voice daemon for OpenCode voice-input plugin.

Subcommands: start, stop, status, stream

start  – begin recording raw PCM via arecord (ALSA)
stop   – stop recording, convert to WAV, transcribe via DashScope fun-asr
status – check whether recording is currently active
stream – real-time audio capture + streaming ASR + volume metering (JSONL)

All output is single-line JSON to stdout (stream emits JSONL with flush).
"""

import json
import os
import signal
import subprocess
import sys
import time
import warnings
import wave
from datetime import datetime, timezone
from pathlib import Path
import threading
from urllib.request import urlopen

# Suppress third-party deprecation/dependency warnings from stderr
warnings.filterwarnings("ignore")

import dashscope
from dashscope.audio.asr import Transcription
from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult
from dashscope import Files as DashScopeFiles

# ── Optional streaming dependencies ─────────────────────────────────────────────
try:
    import sounddevice as sd
    import numpy as np
    _SOUNDDEVICE_AVAILABLE = True
    _SOUNDDEVICE_IMPORT_ERROR = ""
except ImportError as e:
    _SOUNDDEVICE_AVAILABLE = False
    _SOUNDDEVICE_IMPORT_ERROR = str(e)

# ── Optional voice filter dependencies ──────────────────────────────────────────
try:
    from scipy import signal as scipy_signal
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

# ── Configuration ──────────────────────────────────────────────────────────────

API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
MODEL = "fun-asr"
LANGUAGE_HINT = "zh"
SAMPLE_RATE = 16000
NUM_CHANNELS = 1
SAMPLE_WIDTH = 2          # 16-bit PCM → 2 bytes per sample
MIN_PCM_BYTES = 1600      # ≈ 0.05 s at 16 kHz mono 16-bit
MAX_KEPT_RECORDINGS = 3   # Keep last N WAV files for debugging

# Streaming ASR constants
BLOCK_SIZE = 2400             # 150 ms at 16 kHz mono 16-bit
CHANNELS = NUM_CHANNELS       # alias for consistency
SILENCE_TIMEOUT = 30          # auto-stop after 30 s of silence
DASHSCOPE_MODEL = "fun-asr-realtime"

# Voice filter constants
VOICE_BANDPASS_LOW = 80        # Hz — cut sub-bass rumble
VOICE_BANDPASS_HIGH = 7600     # Hz — cut high-frequency hiss (speech < 8kHz)
NOISE_GATE_THRESHOLD = 0.01    # RMS threshold — below this = silence
NOISE_GATE_ATTENUATION = 0.1   # Factor — attenuate silent frames to 10%
SPEAKING_RMS_THRESHOLD = 0.015 # RMS threshold — above this = voice detected

CACHE_DIR = Path("~/.cache/opencode/voice-input").expanduser()
STATE_FILE = CACHE_DIR / "state.json"

# ── JSON Helpers ───────────────────────────────────────────────────────────────

def emit_json(data: dict) -> str:
    """Serialize dict to compact single-line JSON."""
    return json.dumps(data, ensure_ascii=False)


def print_json(data: dict) -> None:
    """Print compact JSON to stdout."""
    print(emit_json(data))


def emit_json_line(data: dict) -> None:
    """Print compact JSON to stdout with flush (for streaming JSONL)."""
    print(emit_json(data), flush=True)


# ── Audio Preprocessor ──────────────────────────────────────────────────────────

class AudioPreprocessor:
    """Real-time voice enhancement for 16kHz mono PCM streams.

    Applies: bandpass filter (80-7600Hz) → RMS noise gate.
    Designed for per-frame streaming — zero lookahead, minimal latency.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._sos = None
        if _SCIPY_AVAILABLE:
            # 4th-order Butterworth bandpass: 80–7600 Hz
            self._sos = scipy_signal.butter(
                4, [VOICE_BANDPASS_LOW, VOICE_BANDPASS_HIGH],
                btype="band", fs=sample_rate, output="sos",
            )

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply bandpass filter + RMS noise gate to a single audio frame.

        Args:
            frame: float32 numpy array in range [-1.0, 1.0]

        Returns:
            Processed float32 numpy array of same shape.
        """
        # 1. Bandpass filter (skip if scipy unavailable)
        if self._sos is not None:
            frame = scipy_signal.sosfilt(self._sos, frame)

        # 2. RMS noise gate — attenuate silent frames, keep speech frames
        rms = np.sqrt(np.mean(frame.astype(np.float64) ** 2))
        if rms < NOISE_GATE_THRESHOLD:
            frame = frame * NOISE_GATE_ATTENUATION

        return frame

    @staticmethod
    def is_speaking(frame: np.ndarray) -> bool:
        """Detect whether an audio frame contains voice energy."""
        rms = np.sqrt(np.mean(frame.astype(np.float64) ** 2))
        return rms >= SPEAKING_RMS_THRESHOLD


def fail(error: str) -> dict:
    """Factory for error response dict."""
    return {"ok": False, "error": error}


def ok(**kwargs: object) -> dict:
    """Factory for success response dict."""
    return {"ok": True, **kwargs}


# ── State Management ───────────────────────────────────────────────────────────

def read_state() -> dict | None:
    """Parse state file. Returns None when absent or corrupt."""
    if not STATE_FILE.is_file():
        return None
    try:
        return json.loads(STATE_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def write_state(data: dict) -> None:
    """Persist state dict to disk, creating dirs as needed."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(emit_json(data))


def delete_state() -> None:
    """Remove state file if present."""
    STATE_FILE.unlink(missing_ok=True)


# ── Process Utilities ──────────────────────────────────────────────────────────

def is_process_alive(pid: int) -> bool:
    """Check whether a PID exists (kill 0 is a no-op signal)."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def kill_process(pid: int) -> bool:
    """Graceful SIGTERM → 3 s wait → SIGKILL. Returns True on success."""
    if not is_process_alive(pid):
        return False

    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return False

    for _ in range(30):          # 3 s with 0.1 s intervals
        if not is_process_alive(pid):
            return True
        time.sleep(0.1)

    # Force-kill straggler
    try:
        os.kill(pid, signal.SIGKILL)
        time.sleep(0.2)
    except OSError:
        pass
    return True


# ── Audio I/O ──────────────────────────────────────────────────────────────────

def convert_raw_to_wav(raw_path: Path) -> Path:
    """Prepend a valid WAV header to raw 16-bit mono PCM data.

    Returns the resolved path of the new .wav file.
    """
    pcm_data = raw_path.read_bytes()
    wav_path = Path(str(raw_path).replace(".raw", ".wav"))
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(NUM_CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_data)
    return wav_path


def check_arecord_available() -> None:
    """Fail fast if arecord is not on PATH."""
    try:
        result = subprocess.run(
            ["arecord", "--version"], capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            raise RuntimeError(
                "arecord exited with code %d — is your microphone connected?"
                % result.returncode
            )
    except FileNotFoundError:
        raise RuntimeError(
            "arecord not found — install alsa-utils and ensure a microphone is plugged in"
        )


def rotate_recordings(cache_dir: Path) -> None:
    """Keep only the newest MAX_KEPT_RECORDINGS WAV recordings.

    Scans cache_dir for voice_*.wav files, sorts by modification time
    descending, and removes any beyond the limit.
    """
    wav_files = sorted(
        cache_dir.glob("voice_*.wav"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for stale in wav_files[MAX_KEPT_RECORDINGS:]:
        stale.unlink(missing_ok=True)


# ── Volume Calculation ──────────────────────────────────────────────────────────

def calculate_volume(pcm_bytes: bytes) -> float:
    """Calculate RMS volume from 16-bit PCM data, normalized to 0~1."""
    if not _SOUNDDEVICE_AVAILABLE:
        return 0.0
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    if len(samples) == 0:
        return 0.0
    rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2))
    normalized = min(rms / 10000.0, 1.0)
    return max(normalized, 0.0)


# ── STT (DashScope fun-asr) ────────────────────────────────────────────────────

def _friendly_asr_error(code: str) -> str:
    """Map known DashScope ASR error codes to concise user-facing messages.

    Returns a friendly string, or '' if the code is not recognised.
    """
    _ASR_ERRORS = {
        "ASR_RESPONSE_HAVE_NO_WORDS": "no speech detected in recording",
        "DECODE_ERROR": "audio format not recognised — must be 16 kHz mono WAV",
        "FILE_URL_PROTOCOL_NOT_SUPPORTED": "audio upload failed — check network",
        "FILE_DOWNLOAD_FAILED": "audio file not accessible",
        "SERVER_ERROR": "DashScope server error — try again later",
    }
    return _ASR_ERRORS.get(code, "")


def _fetch_transcription_json(oss_url: str) -> dict | None:
    """Download and parse the transcription JSON from the OSS result URL.

    Returns the parsed dict, or None on any failure.
    """
    try:
        with urlopen(oss_url, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def _extract_text(transcript_data: dict) -> str:
    """Walk common key paths in the fun-asr transcript JSON to extract text."""
    if not transcript_data:
        return ""

    # Path 1: transcripts (plural) — actual fun-asr structure
    transcripts = transcript_data.get("transcripts")
    if transcripts and isinstance(transcripts, list):
        parts = []
        for entry in transcripts:
            if isinstance(entry, dict):
                text = entry.get("text", "")
                if text:
                    parts.append(str(text))
        if parts:
            return "\n".join(parts)

    # Path 2: transcript (singular, dict or string)
    transcript = transcript_data.get("transcript")
    if transcript:
        if isinstance(transcript, dict):
            text = transcript.get("text", "")
            if text:
                return str(text)
        elif isinstance(transcript, str):
            return transcript

    # Path 3: bare "text" key
    text = transcript_data.get("text", "")
    if text:
        return str(text)

    # Path 4: sentences at top level
    sentences = transcript_data.get("sentences")
    if sentences and isinstance(sentences, list):
        parts = []
        for s in sentences:
            if isinstance(s, dict):
                t = s.get("text", "")
                if t:
                    parts.append(str(t))
        if parts:
            return " ".join(parts)

    return ""


def _extract_transcription_url(results: list) -> str:
    """Walk the API response list to find the transcription_url.

    The fun-asr response nests results at multiple levels.
    We try common key-paths in order:

    1. entry['transcription_url']                     — top-level shortcut
    2. entry['results'][0]['transcription_url']       — one level deep
    3. entry['output']['results'][0]['transcription_url'] — deepest path

    Returns the URL string, or '' if not found.
    """
    if not results or not isinstance(results, list):
        return ""

    entry = results[0]
    if not isinstance(entry, dict):
        return ""

    # Path 1: direct on entry
    url = entry.get("transcription_url", "")
    if url:
        return str(url)

    # Path 2: entry.results[0].transcription_url
    inner_list = entry.get("results")
    if inner_list and isinstance(inner_list, list):
        for item in inner_list:
            if isinstance(item, dict):
                url = item.get("transcription_url", "")
                if url:
                    return str(url)

    # Path 3: entry.output.results[0].transcription_url
    output = entry.get("output")
    if output and isinstance(output, dict):
        nested = output.get("results")
        if nested and isinstance(nested, list):
            for item in nested:
                if isinstance(item, dict):
                    url = item.get("transcription_url", "")
                    if url:
                        return str(url)

    return ""


def transcribe_audio(wav_path: str) -> str:
    """Upload WAV to DashScope, then transcribe via fun-asr.

    1. Upload WAV to DashScope Files (purpose=inference)
    2. Use the resulting signed OSS URL as file_urls for Transcription.async_call
    3. Wait for transcription, fetch the transcript JSON, extract text

    Raises RuntimeError on any pipeline failure so the caller can surface it.
    """
    dashscope.api_key = API_KEY

    # ── Step 1: Upload WAV to DashScope Files ──
    upload_resp = DashScopeFiles.upload(
        file_path=wav_path,
        purpose="inference",
    )
    if upload_resp.status_code != 200:
        raise RuntimeError(
            "File upload failed — status %d: %s"
            % (upload_resp.status_code, upload_resp.message)
        )

    uploaded = (upload_resp.output or {}).get("uploaded_files", [])
    if not uploaded:
        raise RuntimeError("File upload returned no files")

    file_id = uploaded[0].get("file_id", "")
    if not file_id:
        raise RuntimeError("File upload returned no file_id")

    # ── Step 2: Retrieve the signed OSS URL for the uploaded file ──
    file_info = DashScopeFiles.get(file_id)
    if file_info.status_code != 200:
        raise RuntimeError(
            "Failed to retrieve file URL — status %d: %s"
            % (file_info.status_code, file_info.message)
        )

    file_url = (file_info.output or {}).get("url", "")
    if not file_url:
        raise RuntimeError("File info missing OSS URL")

    # ── Step 3: Submit async transcription ──
    result = Transcription.async_call(
        model=MODEL,
        file_urls=[file_url],
        language_hint=LANGUAGE_HINT,
    )

    if result.status_code != 200:
        raise RuntimeError(
            "DashScope transcription request failed — status %d: %s"
            % (result.status_code, result.message)
        )

    transcription = Transcription.wait(result)

    # ── Step 4: Inspect the outer results list ──
    raw_output = transcription.get("output", transcription)
    file_results = raw_output.get("results")
    if not file_results:
        raise RuntimeError("DashScope response missing 'results'")

    entry = file_results[0] if isinstance(file_results, list) else file_results

    # ── Step 5: Check subtask status ──
    subtask_status = entry.get("subtask_status", "") if isinstance(entry, dict) else ""
    if subtask_status and subtask_status != "SUCCEEDED":
        error_code = entry.get("code", subtask_status)
        error_msg = entry.get("message", "")
        friendly = _friendly_asr_error(error_code)
        detail = friendly or ("%s: %s" % (error_code, error_msg) if error_msg else error_code)
        raise RuntimeError(detail)

    # ── Step 6: Extract transcription_url ──
    transcription_url = _extract_transcription_url(
        file_results if isinstance(file_results, list) else [file_results]
    )
    if not transcription_url:
        raise RuntimeError(
            "DashScope response missing transcription_url — status=%s"
            % subtask_status
        )

    # ── Step 7: Fetch the actual transcript JSON ──
    transcript_data = _fetch_transcription_json(transcription_url)
    if not transcript_data:
        raise RuntimeError("Failed to fetch transcript from OSS URL")

    # ── Step 8: Extract text ──
    text = _extract_text(transcript_data)
    if not text:
        raise RuntimeError(
            "Transcription returned empty text — silence or unrecognised audio"
        )

    return text


# ── Subcommand: start ──────────────────────────────────────────────────────────

def cmd_start() -> None:
    """Begin recording raw PCM via arecord."""
    if read_state() is not None:
        print_json(fail("already_recording"))
        return

    check_arecord_available()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc).isoformat()
    raw_path = CACHE_DIR / ("recording_%s.raw" % int(time.time()))

    # Spawn arecord as detached subprocess
    try:
        proc = subprocess.Popen(
            ["arecord", "-f", "S16_LE", "-r", str(SAMPLE_RATE), "-c", str(NUM_CHANNELS), "-t", "raw", str(raw_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        print_json(fail("arecord not found"))
        return
    except Exception as exc:
        print_json(fail("failed to start arecord: %s" % exc))
        return

    # Give arecord a moment to validate the device
    time.sleep(0.3)
    if proc.poll() is not None:
        print_json(fail("arecord exited immediately — is your microphone available?"))
        return

    write_state({
        "pid": proc.pid,
        "raw_file": str(raw_path),
        "started_at": started_at,
    })

    print_json(ok(status="recording", started_at=started_at))


# ── Subcommand: stop ───────────────────────────────────────────────────────────

def cmd_stop() -> None:
    """Stop recording, convert to WAV, transcribe, and keep the recording for debugging."""
    state = read_state()
    if state is None:
        print_json(fail("not_recording"))
        return

    pid = state.get("pid")
    raw_file = state.get("raw_file")

    if pid is None or raw_file is None:
        delete_state()
        print_json(fail("corrupt state — cleared"))
        return

    # Kill the recording process
    kill_process(pid)

    raw_path = Path(raw_file)
    if not raw_path.is_file():
        delete_state()
        print_json(fail("raw audio file missing — recording may have failed"))
        return

    pcm_size = raw_path.stat().st_size
    if pcm_size < MIN_PCM_BYTES:
        raw_path.unlink(missing_ok=True)
        delete_state()
        print_json(fail("recording_too_short"))
        return

    # Convert raw PCM → WAV
    try:
        wav_path = convert_raw_to_wav(raw_path)
    except Exception as exc:
        raw_path.unlink(missing_ok=True)
        delete_state()
        print_json(fail("wav conversion failed: %s" % exc))
        return

    # Save recording with timestamp filename for debugging rotation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    kept_path = CACHE_DIR / "voice_{}.wav".format(timestamp)
    try:
        wav_path.rename(kept_path)
    except OSError:
        # Cross-device fallback — keep the original generated name
        kept_path = wav_path
    wav_path = kept_path

    # Transcribe via DashScope
    try:
        text = transcribe_audio(str(wav_path))
    except Exception as exc:
        raw_path.unlink(missing_ok=True)
        delete_state()
        rotate_recordings(CACHE_DIR)
        print_json(fail("transcription failed: %s" % exc))
        return

    # Clean up raw PCM (WAV is kept for debugging)
    raw_path.unlink(missing_ok=True)
    delete_state()
    rotate_recordings(CACHE_DIR)

    print_json(ok(text=text))


# ── Subcommand: status ─────────────────────────────────────────────────────────

def cmd_status() -> None:
    """Check whether a recording session is active."""
    state = read_state()

    if state is None:
        print_json(ok(recording=False))
        return

    pid = state.get("pid")
    started_at = state.get("started_at")

    if pid is not None and is_process_alive(pid):
        print_json(ok(recording=True, started_at=started_at))
        return

    # Process is dead but state remains — clean up stale state
    delete_state()
    print_json(ok(recording=False))


# ── Subcommand: stream ─────────────────────────────────────────────────────────

def cmd_stream() -> None:
    """Real-time audio capture + streaming ASR + volume metering."""
    if not API_KEY:
        emit_json_line({"event": "status", "state": "error",
                        "message": "DASHSCOPE_API_KEY not set"})
        sys.exit(1)

    if not _SOUNDDEVICE_AVAILABLE:
        emit_json_line({"event": "status", "state": "error",
                        "message": "Missing dependencies: %s" % _SOUNDDEVICE_IMPORT_ERROR})
        emit_json_line({"event": "status", "state": "info",
                        "message": "Fallback: use 'start'/'stop' for batch recording"})
        sys.exit(1)

    dashscope.api_key = API_KEY

    stop_event = threading.Event()
    speech_lock = threading.Lock()

    class StreamCallback(RecognitionCallback):
        def __init__(self):
            super().__init__()
            self._speaking = False

        def on_open(self):
            emit_json_line({"event": "status", "state": "recording"})

        def on_close(self):
            emit_json_line({"event": "status", "state": "completed"})
            stop_event.set()

        def on_event(self, result):
            sentences = result.get_sentence()
            if not sentences:
                return
            if isinstance(sentences, dict):
                sentences = [sentences]
            for sentence in sentences:
                text = sentence.get("text", "") if isinstance(sentence, dict) else ""
                if not text:
                    continue
                with speech_lock:
                    if not self._speaking:
                        self._speaking = True
                        emit_json_line({"event": "status", "state": "speaking"})

                if RecognitionResult.is_sentence_end(sentence):
                    # Final sentence: emit full text
                    emit_json_line({"event": "transcription", "text": text})
                else:
                    # Partial: emit full text so far (plugin silently ignores)
                    emit_json_line({"event": "partial", "text": text})

        def on_error(self, result):
            msg = str(result) if result else "unknown error"
            emit_json_line({"event": "status", "state": "error", "message": msg})
            stop_event.set()

    preprocessor = AudioPreprocessor(SAMPLE_RATE) if _SOUNDDEVICE_AVAILABLE else None
    cb = StreamCallback()
    recognition = Recognition(
        model=DASHSCOPE_MODEL,
        format="pcm",
        sample_rate=SAMPLE_RATE,
        callback=cb,
        disfluency_removal_enabled=True,
        language_hints=["zh"],
    )

    def audio_cb(indata, frames, time_info, status):
        if stop_event.is_set():
            raise sd.CallbackStop()
        if status:
            print("[audio] %s" % status, file=sys.stderr, flush=True)
        try:
            # Convert int16 → float32 for processing
            float_data = indata.astype(np.float32) / 32768.0

            # 1. Apply voice enhancement (bandpass + noise gate)
            if preprocessor is not None:
                float_data = preprocessor.process_frame(float_data)

            # 2. Volume level (before noise gate for accurate meter)
            level = calculate_volume(indata.tobytes())
            emit_json_line({"event": "volume", "level": round(level, 3)})

            # 3. Speaking detection (RMS-based, faster than ASR sentence detection)
            if preprocessor is not None and preprocessor.is_speaking(float_data):
                with speech_lock:
                    if not cb._speaking:
                        cb._speaking = True
                        emit_json_line({"event": "status", "state": "speaking"})

            # 4. Convert float32 → int16 PCM for ASR
            pcm = (float_data * 32767.0).astype(np.int16).tobytes()

            # 5. Send to ASR
            recognition.send_audio_frame(pcm)
        except sd.CallbackStop:
            raise
        except Exception as exc:
            emit_json_line({"event": "status", "state": "error",
                            "message": "Audio callback failure: %s" % exc})
            stop_event.set()
            raise sd.CallbackStop()

    def on_terminate(signum, frame):
        emit_json_line({"event": "status", "state": "transcribing"})
        time.sleep(0.3)   # Flush remaining buffered audio to ASR before stopping
        stop_event.set()

    signal.signal(signal.SIGTERM, on_terminate)
    signal.signal(signal.SIGINT, on_terminate)

    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=BLOCK_SIZE,
            callback=audio_cb,
        )
    except sd.PortAudioError as exc:
        emit_json_line({"event": "status", "state": "error",
                        "message": "Microphone error: %s" % exc})
        sys.exit(1)

    recognition_started = False
    try:
        try:
            recognition.start()
        except Exception as exc:
            emit_json_line({"event": "status", "state": "error",
                            "message": "DashScope connection failed: %s" % exc})
            sys.exit(1)
        recognition_started = True

        stream.start()

        check_interval = 0.1
        elapsed = 0.0
        while not stop_event.is_set() and elapsed < SILENCE_TIMEOUT:
            stop_event.wait(check_interval)
            elapsed += check_interval

        if elapsed >= SILENCE_TIMEOUT and not stop_event.is_set():
            emit_json_line({"event": "status", "state": "warning",
                            "message": "Silence timeout (%ds) reached" % SILENCE_TIMEOUT})

        emit_json_line({"event": "status", "state": "transcribing"})
    finally:
        try:
            stream.abort()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass
        stream = None
        if recognition_started:
            try:
                recognition.stop()
            except Exception:
                pass


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        print_json(fail("missing subcommand (start|stop|status|stream)"))
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "stream":
        try:
            cmd_stream()
        except Exception as exc:
            emit_json_line({"event": "status", "state": "error",
                            "message": "panic: %s" % exc})
            sys.exit(1)
        return

    try:
        if cmd == "start":
            cmd_start()
        elif cmd == "stop":
            cmd_stop()
        elif cmd == "status":
            cmd_status()
        else:
            print_json(fail("unknown subcommand: %s" % cmd))
            sys.exit(1)
    except KeyboardInterrupt:
        print_json(fail("interrupted"))
        sys.exit(1)
    except Exception as exc:
        print_json(fail("panic: %s" % exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
