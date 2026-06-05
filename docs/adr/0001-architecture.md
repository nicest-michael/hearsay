# ADR 0001 — Hearsay architecture & the MisoTTS reality check

**Status:** accepted · **Date:** 2026-06-05

## Context

Goal: a local macOS app — click **Go**, have a spoken, interruptible (barge-in) conversation, with the AI replying in the MisoTTS "Miso" emotive voice. Hardware: MacBook Pro M5 Pro, 48 GB. Forked from `earshot` (hexagonal Rust + Tauri transcription app) for the real-time STT half.

The plan (`docs/plans/2026-06-05-hearsay-voice-agent.md`) and the agent council (Codex + Gemini + GPT-5, HIGH consensus) agreed: the dominant risk is **MisoTTS real-time-factor (RTF) on Apple MPS**, and it must be **measured before building the pipeline around it**. The council pre-authorized a fallback (Kokoro via mlx-audio, behind the same `SpeechSynthesizer` port) if RTF > ~2.5.

## The measurement (spike, `scripts/spike_miso.py`)

MisoTTS 8B = Sesame-CSM architecture: 7.7B Llama-3.2 backbone + 300M depth decoder, **Mimi** 24 kHz codec, 32 RVQ codebooks. Per 80 ms audio frame (12.5 fps), `Model.generate_frame` runs **1× backbone(8B) decode + 31× sequential decoder(300M) decodes**.

Measured on this M5 Pro, **fp16 on MPS confirmed** (`param_dtype=torch.float16`, `mask_device=mps:0`), grad disabled, Mimi+watermark on CPU, model warm:

| Sentence | audio | gen time | **RTF** | median frame |
|---|---|---|---|---|
| "Hey, what's up?" | 0.32 s | 4.63 s | **14.5** | 893 ms |
| "I can hear you loud and clear." | 2.48 s | 24.68 s | **9.95** | 766 ms |

- **RTF ≈ 10×** (≈ 800 ms to synthesize each 80 ms frame). Cold load 108 s; peak RSS 26.9 GB.
- **Root cause:** the 32-deep sequential RVQ decode per frame. Each of the 31 tiny 300M-decoder forwards is **kernel-launch / latency bound** on Metal (CUDA hides this with graph capture / fast launch; MPS does not). This is an architecture tax, not a tuning miss — quantization (bandwidth) won't fix a launch-overhead bottleneck. Optimization might reach ~4–5×; still far from real-time.

**Verdict:** MisoTTS 8B cannot drive a real-time conversation on this Mac. A reply would lag ~10× its spoken length — unusable for the "click Go and talk" experience. This is reported honestly (no pretending it's fast).

## Decision

1. **Default conversational voice = Kokoro 82M via `mlx-audio`** — **VERIFIED on this M5 Pro**: warm RTF ≈ **0.05** (74 ms to synthesize "Yeah, totally"; 269 ms for a 5.35 s sentence), load 5.3 s cold / 0.11 s warm, voice `af_heart` (natural, expressive English). That is ~20× faster than real-time and ~200× faster than Miso — ideal for real-time barge-in. Needs the `misaki[en]` g2p package. First generation compiles MLX kernels (~23 s), so the sidecar **warms with a dummy synth on load**. Because a whole sentence is ~150 ms, **no frame-streaming/pipelining is needed** (synthesize per sentence, enqueue; barge-in = playback flush + cancel-between-sentences). Guaranteed fallback if ever needed: macOS `AVSpeechSynthesis` (built-in, instant).
2. **The Miso voice still ships**, behind the **same `SpeechSynthesizer` port**, as a selectable **"Miso 8B — highest quality, NOT real-time"** option. The user can hear the real emotive Miso voice speak the agent's replies; it's just slow. This honors the explicit want as far as the hardware allows. Only one TTS loads at a time (RAM), chosen by the active voice.
3. **The port boundary is what makes this free** — the FSM, engine, LLM, capture, and playback don't change regardless of which synthesizer is active. This is exactly why the council called the hexagon "not over-engineering."

## Other decisions (validated by council)

- **LLM:** small `Qwen2.5-Instruct-4bit` (3B/7B) via `mlx_lm.server` (OpenAI-compatible HTTP). NOT the 30B coder (wrong personality, too heavy). Terse 1–2-sentence persona — natural banter and keeps replies short.
- **Barge-in:** pure `conversation::Dialog` FSM (Idle/Listening/Thinking/Speaking) + monotonic turn ids that invalidate stale output; VAD speech-onset during Speaking → cancel LLM+TTS + flush playback (instant audible cut) → Listening. Echo handled by headphones for v1 (+ sustained-speech guard); macOS VoiceProcessingIO AEC is a future ADR.
- **Memory ("don't leave models loaded"):** LLM + TTS run as **child-process sidecars the Tauri app owns**. Stop keeps them warm (fast resume); window-close / Quit / idle-timeout frees them. Hardened beyond kill-on-Drop: sidecars self-exit when the parent dies (`os.getppid()`), plus a startup sweep of stale sidecars. Killing the process returns unified memory to the OS.
- **Architecture:** keep earshot's hexagonal layering (pure `core` + ports, `adapters` for I/O, `engine` threads, Tauri wire-DTO seam). New ports: `LlmClient`, `SpeechSynthesizer`, `AudioPlayer`.

## "Tauri + Electron"

The user said "Rust so tauri + electron." Tauri is the Rust-native desktop shell (Electron is JS and would abandon the Rust core). We use **Tauri only**; "Electron" is read as "a desktop webview app," which Tauri provides.

## Consequences

- The headline experience works and is real-time, via Kokoro. The Miso voice is present and usable, just not real-time — an honest reflection of what an 8B autoregressive RVQ TTS can do on a laptop GPU today.
- If MisoLabs ships the promised hosted API (110 ms) or an MLX/quantized build later, it drops in behind the existing `SpeechSynthesizer` port with no other changes.
