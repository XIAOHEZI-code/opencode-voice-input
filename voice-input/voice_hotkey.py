#!/usr/bin/env python3
"""Voice input global hotkey daemon for OpenCode.

Listens for F5 globally regardless of which app is focused.
First press starts streaming ASR recording (real-time transcription);
second press stops and finalizes. All ASR events flow through stdout
to the parent opencode plugin for TUI injection.

Run:  python3 ~/.config/opencode/plugins/voice-input/voice_hotkey.py
Stop: Ctrl+C
"""

import atexit
import enum
import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────

_ENV_FILE = Path("~/.config/opencode/.env").expanduser()
_DAEMON_PATH = Path("~/.config/opencode/plugins/voice-input/voice_daemon.py").expanduser()

# ── Exit cleanup ───────────────────────────────────────────────────────────────

def _atexit_cleanup() -> None:
    """Best-effort cleanup on process exit."""
    try:
        _stop_streaming()
    except Exception:
        pass


atexit.register(_atexit_cleanup)
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))


# ── State machine ──────────────────────────────────────────────────────────────

class _State(enum.IntEnum):
    IDLE = 0       # Not recording, waiting for first F5 press
    STARTING = 1   # Daemon "start" is running in background
    RECORDING = 2  # Actively recording, waiting for second F5 press
    STOPPING = 3   # Daemon "stop" + transcription + HTTP send in background


_state = _State.IDLE
_lock = threading.Lock()


# ── Print helpers ──────────────────────────────────────────────────────────────

def _info(message: str) -> None:
    """Print a status message to the user."""
    print(message, flush=True)


def _error(message: str) -> None:
    """Print an error message to stderr."""
    print(f"❌ {message}", file=sys.stderr, flush=True)


# ── Environment parsing ────────────────────────────────────────────────────────

def _parse_api_key(env_path: Path) -> str:
    """Extract DASHSCOPE_API_KEY from the env file.

    Returns empty string when the file is missing or the key is absent.
    """
    if not env_path.is_file():
        return ""
    try:
        content = env_path.read_text()
    except OSError:
        return ""

    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("DASHSCOPE_API_KEY="):
            return stripped.split("=", 1)[1].strip()
    return ""


_API_KEY = _parse_api_key(_ENV_FILE)


# ── Streaming daemon ───────────────────────────────────────────────────────────

_stream_proc: subprocess.Popen | None = None
_stream_thread: threading.Thread | None = None


def _reader_thread(proc: subprocess.Popen) -> None:
    """Read stdout lines from the stream daemon and forward to parent opencode.

    Each line is a complete JSON event from voice_daemon.py stream.
    We print it unchanged to our own stdout, which pipes to opencode.
    When the daemon exits, reset state to IDLE.
    """
    global _state
    try:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace").strip() if isinstance(raw, bytes) else str(raw).strip()
            if line:
                print(line, flush=True)
    except (OSError, ValueError):
        # Pipe closed — daemon exited
        pass
    finally:
        proc.wait()
        with _lock:
            _state = _State.IDLE


def _start_streaming() -> None:
    """Spawn voice_daemon.py stream as a long-running child process.

    The daemon handles audio capture + streaming ASR + volume metering
    and emits JSON events on stdout. A reader thread forwards them to opencode.
    """
    global _state, _stream_proc, _stream_thread

    env = os.environ.copy()
    env["DASHSCOPE_API_KEY"] = _API_KEY

    try:
        _stream_proc = subprocess.Popen(
            [sys.executable, str(_DAEMON_PATH), "stream"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
    except Exception as exc:
        _error(f"启动流式识别失败: {exc}")
        with _lock:
            _state = _State.IDLE
        return

    _stream_thread = threading.Thread(
        target=_reader_thread,
        args=(_stream_proc,),
        daemon=True,
    )
    _stream_thread.start()

    with _lock:
        _state = _State.RECORDING


def _stop_streaming() -> None:
    """Send SIGTERM to the stream daemon to stop recording + finalize ASR."""
    global _state

    if _stream_proc is None:
        with _lock:
            _state = _State.IDLE
        return

    if _stream_proc.poll() is None:
        try:
            _stream_proc.terminate()
        except OSError:
            pass

    with _lock:
        _state = _State.STOPPING


# ── Event emitter ──────────────────────────────────────────────────────────────

def _emit_event(event_type: str, payload: dict | None = None) -> None:
    """Emit a JSON event line to stdout for the parent plugin to consume.

    The parent opencode plugin reads these lines from the child process's
    stdout pipe and routes them to the appropriate TUI action.
    """
    event = {"event": event_type}
    if payload:
        event.update(payload)
    # flush=True is critical — the parent plugin reads line-buffered stdout
    print(json.dumps(event, ensure_ascii=False), flush=True)


# ── Background workers ─────────────────────────────────────────────────────────

def _worker_start() -> None:
    """Start streaming recording (runs in background thread)."""
    _start_streaming()


def _worker_stop() -> None:
    """Stop streaming recording (runs in background thread)."""
    _stop_streaming()


# ── Hotkey handler ─────────────────────────────────────────────────────────────

def _on_hotkey() -> None:
    """Called by pynput when the global F5 hotkey is pressed.

    Transitions the recording state machine and dispatches background work.
    If a transition is already in progress (STARTING/STOPPING), the press
    is silently ignored.
    """
    global _state

    with _lock:
        if _state == _State.IDLE:
            _state = _State.STARTING
            _action = "start"
        elif _state == _State.RECORDING:
            _state = _State.STOPPING
            _action = "stop"
        else:
            # STARTING or STOPPING — transition in flight, ignore
            return

    if _action == "start":
        threading.Thread(target=_worker_start, daemon=True).start()
    else:
        threading.Thread(target=_worker_stop, daemon=True).start()


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    """Validate prerequisites and start the global hotkey listener."""
    # ── Fail fast on missing prerequisites ──
    if not _API_KEY:
        _error(f"DASHSCOPE_API_KEY not found in {_ENV_FILE}")
        _info("  请在 .env 文件中设置 DASHSCOPE_API_KEY=你的阿里云百炼API密钥")
        sys.exit(1)

    if not _DAEMON_PATH.is_file():
        _error(f"voice_daemon.py not found at {_DAEMON_PATH}")
        _info("  请确认 voice-input 插件已正确安装")
        sys.exit(1)

    # ── Import pynput ──
    try:
        from pynput.keyboard import GlobalHotKeys
    except ImportError:
        _error("pynput 未安装")
        _info("  安装方法: pip3 install pynput")
        _info("  (可能还需要: sudo apt install python3-xlib 或 pip3 install python-xlib)")
        sys.exit(1)

    # ── Run hotkey listener ──
    _info("🎙️ 语音输入已就绪 — 按 F5 开始/停止录音 (Ctrl+C 退出)")

    try:
        with GlobalHotKeys({"<f5>": _on_hotkey}) as listener:
            listener.join()
    except KeyboardInterrupt:
        _stop_streaming()
        _info("")
        _info("👋 语音输入已退出")
    except Exception as exc:
        _error(f"无法注册全局热键: {exc}")
        _info("  请确认当前桌面环境支持 X11/XWayland")
        sys.exit(1)


if __name__ == "__main__":
    main()
