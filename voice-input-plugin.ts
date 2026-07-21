/**
 * voice-input-plugin
 *
 * Server plugin that manages the voice hotkey daemon as a child process.
 * Listens for F5 globally to start/stop voice recording and transcription.
 *
 * Edge cases handled:
 *  - python3 not found → toast warning, skip
 *  - Script already running (PID file check) → skip silently
 *  - pynput not installed → toast warning, skip
 *  - X11/XWayland not available → toast warning, skip
 */

import type { Plugin } from "@opencode-ai/plugin"
import { spawn, type ChildProcess } from "node:child_process"
import { readFileSync, writeFileSync, unlinkSync } from "node:fs"
import { homedir } from "node:os"
import { join } from "node:path"

// ── Constants ──────────────────────────────────────────────────────────────────

const DAEMON_PATH = join(homedir(), ".config", "opencode", "plugins", "voice-input", "voice_hotkey.py")
const PID_FILE = join(homedir(), ".config", "opencode", "plugins", "voice-input", ".pid")
const SIGKILL_DELAY_MS = 3000

// ── Helpers ────────────────────────────────────────────────────────────────────

/**
 * Check whether python3 is available on the system.
 * Returns true if `python3 --version` exits with code 0.
 */
function pythonAvailable(): Promise<boolean> {
	return new Promise((resolve) => {
		const proc = spawn("python3", ["--version"], { stdio: "ignore" })
		proc.on("close", (code) => resolve(code === 0))
		proc.on("error", () => resolve(false))
	})
}

/**
 * Check whether a previous daemon instance is still running
 * by reading the PID file and probing the process.
 */
function daemonAlreadyRunning(): boolean {
	try {
		const raw = readFileSync(PID_FILE, "utf-8").trim()
		if (!raw) return false

		const pid = Number.parseInt(raw, 10)
		if (Number.isNaN(pid)) return false

		// Signal 0 probes existence without actually sending a signal
		process.kill(pid, 0)
		return true
	} catch {
		// File missing OR process not found → not running
		return false
	}
}

/**
 * Write current PID to the PID file atomically so subsequent plugin
 * loads detect a running instance. Uses exclusive-create (wx) to
 * serialize concurrent loads — only one caller succeeds.
 *
 * Returns true if THIS instance owns the PID file (spawn OK).
 * Returns false if another instance already owns it (duplicate spawn).
 */
function writePidFile(pid: number): boolean {
	try {
		writeFileSync(PID_FILE, String(pid), { flag: "wx", encoding: "utf-8" })
		return true
	} catch (err) {
		const code = (err as NodeJS.ErrnoException).code
		if (code !== "EEXIST") return false

		// PID file exists — probe whether the owning process is alive
		try {
			const raw = readFileSync(PID_FILE, "utf-8").trim()
			const existingPid = Number.parseInt(raw, 10)
			if (!Number.isNaN(existingPid)) {
				process.kill(existingPid, 0) // probe
				return false // alive → another instance owns it
			}
		} catch {
			// Process not found → PID file is stale
		}

		// Stale PID file: remove it and retry once
		try { unlinkSync(PID_FILE) } catch { /* ignore */ }
		try {
			writeFileSync(PID_FILE, String(pid), { flag: "wx", encoding: "utf-8" })
			return true
		} catch {
			return false
		}
	}
}

/**
 * Remove the PID file (called on dispose).
 */
function removePidFile(): void {
	try {
		unlinkSync(PID_FILE)
	} catch {
		// File may already be gone
	}
}

/**
 * Show a non-blocking toast notification in the TUI.
 * Wraps `client.tui.showToast` with a catch to prevent
 * unhandled promise rejections.
 */
function toast(
	client: Parameters<Plugin>[0]["client"],
	message: string,
	variant: "info" | "warning" | "error" = "warning",
): void {
	client.tui
		.showToast({ body: { message, variant } })
		.catch(() => {
			// Toast delivery is best-effort — never crash the plugin
		})
}

/**
 * Translate a child-process close code plus stderr into a
 * human-readable toast message when the daemon exits
 * unexpectedly.
 */
function diagnoseExit(code: number | null, stderr: string): string | null {
	if (code === 0) return null

	if (stderr.includes("pynput")) {
		return "pynput is not installed. Run: pip3 install pynput"
	}
	if (stderr.includes("无法注册全局热键") || stderr.includes("X11") || stderr.includes("XWayland") || stderr.includes("global hotkey")) {
		return "X11/XWayland not available. Voice hotkeys require a graphical session."
	}
	if (stderr.includes("DASHSCOPE_API_KEY")) {
		return "DASHSCOPE_API_KEY not set in ~/.config/opencode/.env"
	}
	if (stderr.includes("voice_daemon.py")) {
		return "voice_daemon.py not found. Reinstall the voice-input plugin."
	}

	return null
}

// ── Plugin ─────────────────────────────────────────────────────────────────────

export const server: Plugin = async ({ client }) => {
	// Guard 1: Already running? Skip (Law 1: Early Exit)
	if (daemonAlreadyRunning()) {
		return { dispose: async () => {} }
	}

	// Guard 2: Python available? (Law 1: Early Exit)
	const hasPython = await pythonAvailable()
	if (!hasPython) {
		toast(client, "python3 not found. Install Python 3 to use voice input.")
		return { dispose: async () => {} }
	}

	let child: ChildProcess | null = null
	let killTimer: ReturnType<typeof setTimeout> | null = null
	let stderr = ""

	// ── Phase 2: UI state tracking ──────────────────────────────────────────────
	let completedTimer: ReturnType<typeof setTimeout> | null = null
	let hadTranscription = false
	let lastVolumeToastTime = 0
	let lastVolumeLevel = -1    // Track last displayed level to avoid flicker

	function cancelCompletedTimer(): void {
		if (completedTimer) {
			clearTimeout(completedTimer)
			completedTimer = null
		}
	}

	// Spawn the voice hotkey daemon
	child = spawn("python3", [DAEMON_PATH], {
		stdio: ["ignore", "pipe", "pipe"],
		env: { ...process.env },
	})

	// Track the PID for subsequent loads (atomic — detects duplicate spawns)
	if (!writePidFile(child.pid ?? 0)) {
		// Another instance already owns the PID file — kill our duplicate child
		child.kill("SIGTERM")
		return { dispose: async () => {} }
	}

	// Collect stderr for error diagnosis (capped to prevent unbounded growth)
	child.stderr?.on("data", (data: Buffer) => {
		stderr += data.toString()
		if (stderr.length > 100_000) {
			stderr = stderr.slice(-10_000)
		}
	})

	// Buffer for incomplete stdout lines
	let stdoutBuffer = ""

	// Parse stdout JSON events from the voice hotkey daemon.
	// voice_hotkey.py emits one JSON object per line, e.g.:
	//   {"event": "transcription", "text": "你好世界"}
	child.stdout?.on("data", (data: Buffer) => {
		stdoutBuffer += data.toString()
		if (stdoutBuffer.length > 100_000) {
			console.warn("[voice-input] stdoutBuffer exceeded 100KB, truncating")
			stdoutBuffer = stdoutBuffer.slice(-10_000)
		}
		const lines = stdoutBuffer.split("\n")
		// The last element may be incomplete — keep it in the buffer
		stdoutBuffer = lines.pop() ?? ""

		for (const line of lines) {
			const trimmed = line.trim()
			if (!trimmed) continue

			let event: Record<string, unknown>
			try {
				event = JSON.parse(trimmed)
			} catch {
				// Non-JSON line (status message) — surface for diagnostics
				console.warn("[voice-input] non-JSON stdout:", trimmed.slice(0, 100))
				continue
			}

			// Route known events from voice_daemon.py stream → TUI actions
			switch (event.event) {
				case "status": {
					const state = String(event.state ?? "")
					const message = String(event.message ?? "")

					// Cancel pending cleanup timer when new recording starts
					if (state === "recording" || state === "speaking") {
						cancelCompletedTimer()
						hadTranscription = false
						lastVolumeToastTime = 0
						lastVolumeLevel = -1
					}

					// Build recovery hint for error states
					let hint = ""
					if (state === "error") {
						const lower = message.toLowerCase()
						if (lower.includes("sounddevice") || lower.includes("portaudio") || lower.includes("mic")) {
							hint = "请检查麦克风是否可用"
						} else if (lower.includes("dashscope") || lower.includes("api") || lower.includes("key")) {
							hint = "请检查 DASHSCOPE_API_KEY 配置"
						} else if (lower.includes("connect") || lower.includes("network") || lower.includes("timeout")) {
							hint = "请检查网络连接"
						}
						console.error("[voice-input] daemon error:", state, message)
					}
					if (state === "warning") {
						console.warn("[voice-input] daemon warning:", message)
					}

					const statusMap: Record<string, { text: string; variant: "info" | "warning" | "error" }> = {
						recording:    { text: "🎤 正在录音...",    variant: "info" },
						speaking:     { text: "🗣️ 检测到语音...",  variant: "info" },
						transcribing: { text: "⏳ 识别中...",      variant: "info" },
						completed:    { text: "✅ 识别完成",       variant: "info" },
						error:        { text: `❌ ${message || "识别错误"}${hint ? ` (${hint})` : ""}`, variant: "error" },
						warning:      { text: `⚠️ ${message || "警告"}`,   variant: "warning" },
					}

					const entry = statusMap[state]
					if (entry) {
						client.tui.showToast({
							body: { message: entry.text, variant: entry.variant },
						}).catch(() => {})
					}

					// After completion without transcription, clear volume bar artifacts after 3s
					if (state === "completed" && !hadTranscription) {
						completedTimer = setTimeout(() => {
							client.tui.appendPrompt({ body: { text: "" } }).catch(() => {})
							completedTimer = null
						}, 3000)
					}
					break
				}

				case "volume": {
					const level = Math.min(Math.max(Number(event.level ?? 0), 0), 1)
					// Only update toast if level changed significantly or enough time passed
					const now = Date.now()
					const levelPct = Math.round(level * 100)
					if (now - lastVolumeToastTime < 300 && Math.abs(levelPct - lastVolumeLevel) < 5) break
					lastVolumeToastTime = now
					lastVolumeLevel = levelPct

					const filled = Math.round(level * 8)
					const bar = "█".repeat(filled) + "░".repeat(8 - filled)
					client.tui.showToast({
						body: { message: `🎤 ${bar} ${levelPct}%`, variant: "info" },
					}).catch(() => {})
					break
				}

				case "partial": {
					// Partial ASR text is informational only — don't write to prompt
					// (each partial update is the full text so far; appending would concatenate duplicates)
					break
				}

				case "transcription": {
					const text = String(event.text ?? "").trim()
					hadTranscription = true
					if (!text) break

					client.tui
						.appendPrompt({ body: { text } })
						.then(() => {
							client.tui.showToast({
								body: { message: `✅ 识别完成: ${text.slice(0, 50)}${text.length > 50 ? "..." : ""}`, variant: "info" },
							}).catch(() => {})
						})
						.catch((err: unknown) => {
							console.error("[voice-input] appendPrompt failed:", err)
							client.tui.showToast({
								body: { message: `⚠️ 文字注入失败: ${String(err)}`, variant: "error" },
							}).catch(() => {})
						})
					break
				}

				case "error": {
					const message = String(event.message ?? "unknown error")
					client.tui.showToast({
						body: { message: `❌ ${message}`, variant: "error" },
					}).catch(() => {})
					break
				}
			}
		}
	})

	// Handle unexpected exit (Law 4: Fail Loud → toast is as loud as we get server-side)
	child.on("close", (code: number | null, _signal: NodeJS.Signals | null) => {
		removePidFile()

		// Exit code 0 means clean shutdown (e.g. from dispose)
		const message = diagnoseExit(code, stderr)
		if (message) {
			toast(client, message)
		}
	})

	// Handle spawn errors (e.g. ENOENT for python3, despite our check)
	child.on("error", (err: NodeJS.ErrnoException) => {
		removePidFile()
		if (err.code === "ENOENT") {
			toast(client, "python3 not found. Install Python 3 to use voice input.", "error")
		}
	})

	return {
		dispose: async () => {
			cancelCompletedTimer()

			// Nothing to clean up
			if (!child || child.exitCode !== null) return

			// Register close listener BEFORE sending signals (Fix 2: prevents lost close event)
			const closePromise = new Promise<void>((resolve) => {
				const done = () => {
					if (killTimer) clearTimeout(killTimer)
					killTimer = null
					resolve()
				}

				child?.on("close", done)
				// Safety timeout: resolve even if process hangs
				setTimeout(done, SIGKILL_DELAY_MS + 1000)
			})

			// Send SIGTERM first
			child.kill("SIGTERM")

			// Schedule SIGKILL as fallback after timeout
			killTimer = setTimeout(() => {
				if (child && child.exitCode === null) {
					child.kill("SIGKILL")
				}
			}, SIGKILL_DELAY_MS)

			await closePromise
			removePidFile()
		},
	}
}

export default server
