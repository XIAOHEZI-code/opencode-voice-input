# OpenCode Voice Input Plugin

Voice input plugin for [OpenCode](https://github.com/anomalyco/opencode) using Alibaba Cloud DashScope FunASR streaming speech recognition.

Toggle recording on/off with a hotkey — your speech is transcribed in real-time and inserted into the OpenCode prompt input.

## How It Works

1. Press the voice hotkey to start recording
2. Speak into your microphone
3. Press the hotkey again to stop recording
4. Transcription appears in the OpenCode prompt input bar

## Installation

Copy all plugin files to the OpenCode plugins directory:

```bash
cp voice-input-plugin.ts ~/.config/opencode/plugins/
cp -r voice-input/ ~/.config/opencode/plugins/
```

Or clone directly:

```bash
git clone https://github.com/XIAOHEZI-code/opencode-voice-input.git /tmp/opencode-voice-input
cp /tmp/opencode-voice-input/voice-input-plugin.ts ~/.config/opencode/plugins/
cp -r /tmp/opencode-voice-input/voice-input/ ~/.config/opencode/plugins/
```

Then restart OpenCode.

## Usage

Press **F5** to start/stop recording (default hotkey, configurable via `voice-input-plugin.ts`).

## Requirements

- **Python 3.8+**
- Python packages: `dashscope`, `pynput`, `sounddevice`
- A working microphone
- Alibaba Cloud DashScope API key

## Configuration

Set your DashScope API key in `~/.config/opencode/.env`:

```
DASHSCOPE_API_KEY=your-api-key-here
```

You can obtain an API key from the [Aliyun DashScope Console](https://dashscope.console.aliyun.com/).

### Install Python Dependencies

```bash
pip install -r voice-input/requirements.txt
```

Or manually:

```bash
pip install dashscope pynput sounddevice
```

## Files

| File | Purpose |
|------|---------|
| `voice-input-plugin.ts` | OpenCode TUI plugin entry point |
| `voice-input/voice_daemon.py` | Python daemon for recording and FunASR streaming transcription |
| `voice-input/voice_hotkey.py` | Keyboard hotkey listener (pynput) |
| `voice-input/requirements.txt` | Python package dependencies |
