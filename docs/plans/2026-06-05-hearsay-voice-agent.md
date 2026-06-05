# Hearsay — Local Voice Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local macOS desktop app where you click **Go** and have a spoken, barge-in-able conversation with an AI that speaks in the Miso (MisoTTS 8B) emotive voice — all on-device.

**Architecture:** Hexagonal Rust workspace forked from `earshot` (pure `core` domain + `app` orchestration + `adapters` I/O + `engine` runtime + Tauri shell). earshot already gives us the hard real-time STT half: cpal mic capture, energy VAD with adaptive noise floor, sliding-window Whisper streaming (Metal via `whisper-rs`), LocalAgreement-2 stabilization, and `utterance_ended()` turn detection. We add the **output half** — a streaming local LLM, a MisoTTS Python sidecar for speech, interruptible cpal playback — and the **dialog brain**: a pure conversation FSM with **barge-in** (user speech during playback cancels in-flight LLM + TTS + playback and returns to listening). Heavy models (LLM + TTS) run as **child-process sidecars the app owns**, so "don't leave it loaded" = kill-on-exit/idle.

**Tech Stack:** Rust 1.94, Tauri 2, React 19 + Tailwind v4 + Tabler (copied from earshot), `whisper-rs` (Metal), `cpal` + `ringbuf` (capture & playback), `crossbeam-channel`, `reqwest`/`ureq` (LLM HTTP), a Python 3.10 `uv` sidecar wrapping MisoTTS (`MisoLabsAI/MisoTTS`, Sesame-CSM-style: 7.7B Llama-3.2 backbone + 300M depth decoder + Mimi 24 kHz codec), and `mlx_lm.server` serving a small Qwen-Instruct for the dialog model.

---

## Context & Research Findings (read before building)

These findings (from `/best-practice` research, 2026-06-05) are load-bearing. Don't re-litigate them mid-build.

1. **MisoTTS is NOT a cloud API.** It's open weights (`MisoLabs/MisoTTS` on HF, code `github.com/MisoLabsAI/MisoTTS`). "API coming soon" — not available. So this is fully local. It is a **Sesame CSM clone**: `load_miso_8b(device, model_path_or_repo_id)` → `generator.generate(text, speaker, context, max_audio_length_ms)` → returns a full audio tensor; `generator.sample_rate` (Mimi → 24 kHz). Voice cloning via a `Segment(speaker, text, audio)` context list. Watermarked (SilentCipher) by default.
2. **MisoTTS ships CUDA-first, no streaming, half-duplex.** README does `"cuda" if torch.cuda.is_available() else "cpu"`. To run on this M5 Pro we patch device→`mps` and dtype→**fp16** (MPS has no bf16). Mimi codec + watermark may need CPU fallback (guard with try/except). ~16 GB fp16 weights, ~30–40 GB disk first run.
3. **8B on MPS will be slower than real-time.** CSM-**1B** is ~0.8–1.5× RT on M-series; this is 8B. Strategy: **keep the model warm** and **synthesize per sentence**, streaming each sentence's PCM to playback the instant it's ready, so perceived latency = time-to-first-*sentence*, not whole reply. Barge-in makes slowness feel responsive (you can cut it off). This is a personal test app, not a 110 ms SLA — optimize first-audio latency, accept RTF > 1.
4. **LLM choice: do NOT reuse `Qwen3-Coder-30B` (17 GB).** It's a coder model (wrong personality for casual chat) and 17 GB + 16 GB MisoTTS + Whisper + system blows the 48 GB budget. Use a **small Qwen-Instruct** (e.g. `Qwen2.5-3B-Instruct-4bit` ~1.8 GB or `Qwen2.5-7B-Instruct-4bit` ~4.3 GB) via `mlx_lm.server` (OpenAI-compatible on `127.0.0.1`). Snappy first-token; leaves headroom. Endpoint is configurable so the existing `qwen serve` still works if wanted.
5. **Barge-in echo problem (laptop speakers).** If the mic hears the speakers, TTS triggers false barge-ins. v1 assumes **headphones** (documented), plus a software guard: ignore the first 250 ms of each agent utterance and require ~200 ms of sustained, above-noise-floor speech before declaring barge-in. Proper fix (macOS `kAudioUnitSubType_VoiceProcessingIO` AEC) is noted as a future ADR, out of scope for v1.
6. **Memory budget (target, all warm):** MisoTTS fp16 ~16 GB + Qwen-3B-4bit ~2 GB + Whisper base ~1.5 GB + system/app ~6 GB ≈ **~26 GB / 48 GB**. Comfortable. (Spike Task 0 verifies the MisoTTS half.)
7. **Canonical barge-in pipeline** (LiveKit/Pipecat/Vapi convergence): full-duplex, ASR/VAD always hot, user always wins, **per-turn IDs** invalidate stale LLM/TTS output, cancel LLM+TTS+playback in parallel on barge-in, immediate audible cut-off (< ~200 ms), then new ASR segment for the interruption.

**Source repos to read, never guess:**
- earshot: `/Users/michael/Repos/earshot/` (skeleton we fork). Key files cited inline below.
- MisoTTS: cloned to `/Users/michael/Repos/hearsay/vendor/MisoTTS/` in Task 0.
- qwen launcher: `/Users/michael/Repos/local-llms/qwen` (mlx_lm patterns; CLAUDE.md gotchas about repetition).

---

## Council Adjustments (2026-06-05, confidence HIGH)

Codex + Gemini + GPT-5 reviewed this plan. They **validated** the architecture, the small-Qwen LLM choice, the FSM+turn-id barge-in, the hexagonal layering, and headphones-for-v1. They **sharpened the TTS strategy**. Folded in here; the rest of the plan stands.

**Confirmed hard facts (from reading MisoTTS source + HF, post-plan):**
- Per 80 ms audio frame (12.5 fps), `Model.generate_frame` (models.py:162) runs **ONE 8B-backbone decode + 31 sequential 300M-decoder decodes**. At fp16 on ~273 GB/s the 8B decode alone is ~60–90 ms, so **RTF is very likely 1.5–2.3×** — slower than real-time. Gemini's "physics" warning is correct.
- Weights are a single **32.75 GB fp32** `model.safetensors` (ungated); load to CPU then cast → **~16 GB fp16** on MPS.
- Tokenizer `meta-llama/Llama-3.2-1B` is **gated**; use ungated mirror `unsloth/Llama-3.2-1B` (identical Llama-3.2 vocab). Patched in `generator.py`.
- `run_misotts.py` skips MPS "due to float64 limitations" → run the **transformer on MPS**, **Mimi codec + watermark on CPU** (dodges MPS float64). `PYTORCH_ENABLE_MPS_FALLBACK=1` as a backstop.

**Adjustment A — RTF gate is strict; chunking is NOT a free pass (supersedes finding #3's "accept RTF > 1").** Because RTF > 1 means sentence N+1 isn't ready when N finishes (dead air between sentences), the design is:
  - The sidecar streams **frame-chunked within a sentence** (decode + emit every ~12 frames ≈ 1 s) and the engine **pipelines** (synthesize the next chunk/sentence while the current plays).
  - The persona system prompt **hard-constrains replies to 1–2 short sentences** ("a punchy conversational buddy — reply in one or two short sentences, never a monologue"). This is both the RTF mitigation (no long multi-sentence gap to expose) and the most natural barge-in banter style. Gemini's own dissent: at RTF ≤ ~1.8 a single short burst is fine.
  - **Decision tree by measured RTF (Task 0):** `<0.8` → Miso, full pipelined streaming. `0.8–1.5` → Miso + terse persona + aggressive pipeline (the expected case). `>~2.5` (catastrophic) → fall back to **Kokoro via mlx-audio** (proven ~0.1× RTF on Apple Silicon, natural voice) behind the **same `SpeechSynthesizer` port** — no other code changes. Do NOT pursue an MLX/CSM port (separate multi-day research project); only escalate to it if measurement demands AND Kokoro is unacceptable.

**Adjustment B — cooperative, in-frame TTS cancellation (first-class, not afterthought).** Killing the sidecar mid-inference can leave a Metal kernel running 1–3 s. So: playback flush (`player.barge_stop`) gives the **instant audible cut** (< ~150 ms), AND the sidecar checks the cancel flag **inside the per-frame loop** (between `generate_frame` calls) so the GPU frees within ~1 frame (~80 ms) and the next turn starts fast. The sidecar reimplements `generate()`'s loop (we have the source) to get both the per-frame cancel hook and frame-chunked streaming. (Task 6 updated.)

**Adjustment C — sidecar lifecycle hardened beyond kill-on-Drop (Gemini).** `Drop` doesn't fire if Tauri panics/`SIGKILL`s → a 16 GB process leaks. So: (1) the Python sidecars poll `os.getppid()` and **self-exit when the parent dies** (ppid → 1); (2) on app **startup, sweep** and kill any stale `miso_tts_server.py` / `mlx_lm.server` from a prior crashed run; (3) spawn in the app's process group. This makes "don't leave models loaded" hold even across crashes. (Tasks 6, 7, 9 updated.)

**Adjustment D — Stop keeps models WARM; free RAM on close/Quit/idle (GPT-5).** Killing 16 GB on every Stop means a ~3–5 s reload (M5 Pro NVMe) on the next Go. So **Stop** = stop listening, keep models warm for instant resume; the **idle timer keeps running** and the models are freed on window-close / Quit / **idle-timeout** (default ~5 min). Walking away still unloads — satisfying "don't leave it loaded" — without punishing a quick pause. (Task 9 updated.)

**Adjustment E — verify in the spike that killing the process returns unified memory** (it should: process death frees MPS allocations; nothing persists cross-process). Spike also confirms the cooperative-cancel latency.

---

## File Structure

```
hearsay/
├── Cargo.toml                      # workspace: core, app, adapters, engine, src-tauri
├── rust-toolchain.toml             # 1.94, aarch64-apple-darwin (copy from earshot)
├── Makefile                        # dev/app/test/lint (adapt from earshot)
├── DEPLOY.md                       # written in deploy task
├── README.md
├── docs/
│   ├── plans/2026-06-05-hearsay-voice-agent.md
│   └── adr/0001-architecture.md    # records the decisions above
├── vendor/MisoTTS/                 # git-cloned upstream (gitignored), patched for MPS
├── sidecars/
│   └── miso_tts_server.py          # warm, streaming, cancellable TTS server (newsocket protocol)
├── crates/
│   ├── core/src/
│   │   ├── lib.rs
│   │   ├── audio.rs      vad.rs     aggregator.rs   stabilize.rs   # copied from earshot
│   │   ├── word.rs       transcript.rs   config.rs   error.rs      # copied from earshot
│   │   ├── ports.rs      # earshot ports + NEW: LlmClient, SpeechSynthesizer, AudioPlayer
│   │   ├── dialogue.rs   # NEW: Role, Turn, Conversation history, sentence chunker
│   │   └── conversation.rs # NEW: the dialog FSM (Idle/Listening/Thinking/Speaking/Interrupted)
│   ├── app/src/
│   │   ├── lib.rs
│   │   ├── infer.rs      # copied infer_window/flush from earshot pipeline.rs
│   │   └── error.rs
│   ├── adapters/src/
│   │   ├── lib.rs
│   │   ├── mic.rs        whisper.rs   resample.rs   models.rs      # copied from earshot
│   │   ├── llm.rs        # NEW: OpenAI-compatible streaming chat client (LlmClient impl)
│   │   ├── tts.rs        # NEW: MisoTTS sidecar client over local socket (SpeechSynthesizer impl)
│   │   ├── player.rs     # NEW: cpal interruptible playback (AudioPlayer impl)
│   │   └── sidecar.rs    # NEW: spawn/kill Python+mlx sidecars; kill-on-Drop
│   └── engine/src/
│       ├── lib.rs
│       ├── events.rs     # UiCommand/UiEvent/EngineConfig + ChannelSink
│       └── conversation_loop.rs # NEW: threads wiring STT→LLM→TTS→playback + barge-in
└── src-tauri/
    ├── src/main.rs       # Go/Stop commands, event fan-out, sidecar lifecycle hooks
    ├── tauri.conf.json   Info.plist   capabilities/   icons/
    └── Cargo.toml
└── web/                  # React: one Go button + dual transcript + status (copied/trimmed)
```

**Responsibility boundaries (the Dependency Rule — audited in the hex-audit task):**
- `core` = pure logic + port traits. No tokio, no reqwest, no cpal, no serde-wire types. The conversation FSM is a pure function of (state, event) → (state, commands). **Fully unit-testable with zero mocks.**
- `adapters` = the only place `reqwest`, `cpal`, `whisper-rs`, `std::process::Command` appear.
- `engine` = threads + channels; depends on core ports + adapters; owns model/sidecar lifecycle.
- `src-tauri` = wire DTOs (camelCase), IPC. Domain types never cross the webview boundary.

---

## Task 0: Spike — prove MisoTTS 8B runs on this Mac (DE-RISK FIRST)

> Do this before building anything around MisoTTS. If RTF or RAM is unworkable, the council/fallback (Kokoro via mlx-audio, or a quantized Miso) is chosen here, not after the app is built.

**Files:**
- Create: `vendor/MisoTTS/` (clone), `scripts/spike_miso.py`, `docs/adr/0001-architecture.md` (record result)

- [ ] **Step 1: Install uv + clone MisoTTS**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
mkdir -p /Users/michael/Repos/hearsay/vendor
git clone https://github.com/MisoLabsAI/MisoTTS.git /Users/michael/Repos/hearsay/vendor/MisoTTS
cd /Users/michael/Repos/hearsay/vendor/MisoTTS
uv sync --python 3.10
```
Expected: a `.venv` with torch + deps. If `uv sync` fails on a CUDA-only dep, note which, and `uv pip install torch torchaudio` (CPU/MPS wheels) explicitly.

- [ ] **Step 2: Read the real API surface before patching**

Read `vendor/MisoTTS/run_misotts.py` and `vendor/MisoTTS/generator.py` (or wherever `load_miso_8b`/`Segment`/`generate` live). Confirm the exact signature of `load_miso_8b`, `generate`, `generator.sample_rate`, and every hard-coded `"cuda"` / `.cuda()` / `bfloat16` / `torch.cuda.*` site. Write the list into the ADR.

- [ ] **Step 3: Patch device + dtype for MPS**

Apply the minimal patch set (from research): device-agnostic selection `cuda → mps → cpu`; force `torch.float16` on MPS (never bf16); guard `torch.cuda.*` calls; wrap Mimi codec + SilentCipher watermark in try/except that retries on CPU. Keep patches as a `vendor/MisoTTS.patch` (so re-clone is reproducible). Do not edit anything in `crates/`.

- [ ] **Step 4: Spike script — first synth + measurements**

```python
# scripts/spike_miso.py — run inside vendor/MisoTTS/.venv
import time, torch, torchaudio
from generator import load_miso_8b
dev = "mps" if torch.backends.mps.is_available() else "cpu"
t0=time.time(); gen=load_miso_8b(device=dev, model_path_or_repo_id="MisoLabs/MisoTTS")
print(f"load: {time.time()-t0:.1f}s  sr={gen.sample_rate}")
for text in ["Hey, what's up?", "I can hear you loud and clear, and I'm ready to talk."]:
    t0=time.time()
    audio = gen.generate(text=text, speaker=0, context=[], max_audio_length_ms=8000)
    dt=time.time()-t0; secs=audio.shape[-1]/gen.sample_rate
    print(f"text={text!r}  gen={dt:.2f}s  audio={secs:.2f}s  RTF={dt/secs:.2f}")
    torchaudio.save(f"/tmp/miso_{int(secs*1000)}.wav", audio.unsqueeze(0).cpu(), gen.sample_rate)
```
Run: `cd vendor/MisoTTS && uv run python /Users/michael/Repos/hearsay/scripts/spike_miso.py`
While it runs, in another shell: `while true; do ps -o rss= -p $(pgrep -f spike_miso) 2>/dev/null | awk '{print $1/1048576" GB"}'; sleep 2; done`

- [ ] **Step 5: Record verdict in the ADR**

Capture: load time, `sample_rate`, RTF per sentence, **time-to-first-audio for the short sentence**, peak RSS, and the cooperative-cancel latency. Listen to `/tmp/miso_spike_*.wav` (it's the Miso voice — confirm quality). Also confirm peak RSS drops to ~0 after the process exits (Adjustment E). **Decision gate = Council Adjustment A's RTF decision tree** (`<0.8` full streaming / `0.8–1.5` Miso+terse persona+pipeline / `>~2.5` Kokoro fallback). Record the measured RTF, first-audio, RSS, and the chosen branch in the ADR.

- [ ] **Step 6: Commit the spike**
```bash
cd /Users/michael/Repos/hearsay
printf 'vendor/MisoTTS/.venv/\nvendor/MisoTTS/**/__pycache__/\ntarget/\nweb/node_modules/\nweb/dist/\n*.wav\n' > .gitignore
git add scripts/spike_miso.py docs/adr/0001-architecture.md vendor/MisoTTS.patch .gitignore
git commit -m "spike: MisoTTS 8B on Apple Silicon MPS — RTF/RAM verdict in ADR 0001"
```

---

## Task 1: Workspace scaffold (fork earshot skeleton)

**Files:**
- Create: `Cargo.toml`, `rust-toolchain.toml`, `Makefile`, `crates/{core,app,adapters,engine}/Cargo.toml`, each `src/lib.rs`
- Copy (read each first, then copy + rename `earshot_*`→`hearsay_*`): from earshot `crates/core/src/{audio,vad,aggregator,stabilize,word,transcript,config,error}.rs`; `crates/adapters/src/{mic,whisper,resample,models}.rs`; `crates/app/src/{error}.rs` and the `infer_window`/`flush` bodies from `pipeline.rs`.

- [ ] **Step 1: Root workspace manifest**

Create `Cargo.toml` mirroring earshot's (resolver 2, members core/app/adapters/engine/src-tauri, default-members excluding src-tauri, `[profile.release]` lto thin). Rename crate keys to `hearsay-core` etc. Copy `rust-toolchain.toml` verbatim (1.94, aarch64-apple-darwin).

- [ ] **Step 2: Copy the pure domain modules unchanged**

Copy the 8 `core` files. In each, rename `earshot_core` → `hearsay_core`. Do NOT copy `prompt.rs` (earshot's Claude-prompt builder — not needed). `lib.rs` re-exports: `pub mod audio; pub mod vad; pub mod aggregator; pub mod stabilize; pub mod word; pub mod transcript; pub mod config; pub mod ports; pub mod error;` (dialogue/conversation added in Task 3).

- [ ] **Step 3: Copy adapters we keep**

Copy `mic.rs`, `whisper.rs` (Metal STT), `resample.rs`, `models.rs`, and `adapters/build.rs` (the compiler-rt link fix — required or release link fails, per earshot CLAUDE.md). Drop `system_audio.rs`, `clipboard.rs` (not needed — we capture mic only). Update `adapters/Cargo.toml` to the kept deps.

- [ ] **Step 4: Verify it compiles**

Run: `cd /Users/michael/Repos/hearsay && cargo build`
Expected: PASS (core + app + adapters + empty engine). Fix any dangling `earshot_` references or removed-module imports.

- [ ] **Step 5: Verify copied domain tests still pass**

Copy earshot's `#[cfg(test)]` modules along with each file (they're inline). Run: `cargo test -p hearsay-core`
Expected: PASS — VAD, stabilizer, aggregator tests are pure and must stay green.

- [ ] **Step 6: Commit**
```bash
git add Cargo.toml rust-toolchain.toml Makefile crates/
git commit -m "scaffold: fork earshot hexagonal skeleton (core/app/adapters), green build + domain tests"
```

---

## Task 2: Domain ports for the output half

**Files:**
- Modify: `crates/core/src/ports.rs` (append three traits)
- Modify: `crates/core/src/error.rs` (add `LlmError`, `SynthError`, `PlaybackError`)

- [ ] **Step 1: Write failing tests for the port object-safety + error types**

```rust
// crates/core/src/ports.rs  (in #[cfg(test)] mod tests)
#[test]
fn ports_are_object_safe() {
    fn _assert(_l: &dyn LlmClient, _s: &mut dyn SpeechSynthesizer, _p: &mut dyn AudioPlayer) {}
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cargo test -p hearsay-core ports_are_object_safe`
Expected: FAIL (traits not defined).

- [ ] **Step 3: Define the three ports (domain types only)**

```rust
// crates/core/src/ports.rs (append)
use crate::dialogue::Turn;
use crate::error::{LlmError, PlaybackError, SynthError};

/// Driven port: a streaming chat LLM. `reply` streams assistant text deltas to `on_delta`
/// until the model is done or `cancel` (checked by the adapter) flips true. Returns the full text.
pub trait LlmClient: Send {
    fn reply(
        &self,
        history: &[Turn],
        cancel: &std::sync::atomic::AtomicBool,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<String, LlmError>;
}

/// Driven port: text → 24 kHz mono f32 audio, delivered in chunks via `on_pcm` as they're
/// synthesized. `cancel` (checked between chunks/sentences) aborts synthesis ASAP.
pub trait SpeechSynthesizer: Send {
    fn speak(
        &mut self,
        text: &str,
        cancel: &std::sync::atomic::AtomicBool,
        on_pcm: &mut dyn FnMut(&[f32]),
    ) -> Result<(), SynthError>;
    fn sample_rate(&self) -> u32;
}

/// Driven port: interruptible audio sink. `enqueue` appends f32 frames at `sample_rate`;
/// `barge_stop` flushes everything immediately; `played_samples` reports what actually reached
/// the device (for "how much did they hear").
pub trait AudioPlayer: Send {
    fn enqueue(&mut self, pcm: &[f32]) -> Result<(), PlaybackError>;
    fn barge_stop(&mut self);
    fn played_samples(&self) -> u64;
    fn is_draining(&self) -> bool;
    fn sample_rate(&self) -> u32;
}
```
Add the three error enums to `error.rs` with `thiserror`.

- [ ] **Step 4: Run tests to verify pass**

Run: `cargo test -p hearsay-core` Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add crates/core/src/ports.rs crates/core/src/error.rs
git commit -m "core: add LlmClient/SpeechSynthesizer/AudioPlayer ports + error types"
```

---

## Task 3: Dialogue types + sentence chunker (TDD)

**Files:**
- Create: `crates/core/src/dialogue.rs`; Modify `lib.rs` (`pub mod dialogue;`)

- [ ] **Step 1: Failing tests**

```rust
// crates/core/src/dialogue.rs (#[cfg(test)])
#[test]
fn chunker_emits_on_sentence_end() {
    let mut c = SentenceChunker::new();
    assert_eq!(c.push("Hello"), None);
    assert_eq!(c.push(" there."), Some("Hello there.".to_string()));
    assert_eq!(c.push(" How are"), None);
    assert_eq!(c.push(" you?"), Some("How are you?".to_string()));
    assert_eq!(c.flush(), None);
}
#[test]
fn chunker_flushes_tail_without_terminator() {
    let mut c = SentenceChunker::new();
    c.push("no period here");
    assert_eq!(c.flush(), Some("no period here".to_string()));
}
#[test]
fn chunker_first_chunk_can_break_on_comma_for_low_latency() {
    // The very first chunk of a reply may break early (after >=12 chars at a comma)
    // so first-audio latency is short; later chunks wait for sentence ends.
    let mut c = SentenceChunker::new();
    assert_eq!(c.push("Well, let me think"), Some("Well,".to_string()));
}
#[test]
fn conversation_appends_and_orders() {
    let mut conv = Conversation::new("You are a friendly conversational partner.");
    conv.add_user("hi");
    conv.add_assistant("hey!");
    let t = conv.turns();
    assert_eq!(t[0].role, Role::System);
    assert_eq!((t[1].role, t[1].text.as_str()), (Role::User, "hi"));
    assert_eq!((t[2].role, t[2].text.as_str()), (Role::Assistant, "hey!"));
}
```

- [ ] **Step 2: Run, verify fail** — `cargo test -p hearsay-core dialogue` → FAIL.

- [ ] **Step 3: Implement**

```rust
// crates/core/src/dialogue.rs
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Role { System, User, Assistant }

#[derive(Clone, Debug)]
pub struct Turn { pub role: Role, pub text: String }

#[derive(Clone, Debug)]
pub struct Conversation { turns: Vec<Turn> }

impl Conversation {
    pub fn new(system: &str) -> Self {
        Self { turns: vec![Turn { role: Role::System, text: system.into() }] }
    }
    pub fn add_user(&mut self, t: &str) { self.turns.push(Turn { role: Role::User, text: t.into() }); }
    pub fn add_assistant(&mut self, t: &str) { self.turns.push(Turn { role: Role::Assistant, text: t.into() }); }
    pub fn turns(&self) -> &[Turn] { &self.turns }
    /// Keep the system turn + the last `n` exchanges (cap context/RAM).
    pub fn truncate_keep_last(&mut self, n: usize) {
        if self.turns.len() <= 1 + n { return; }
        let tail = self.turns.split_off(self.turns.len() - n);
        self.turns.truncate(1);
        self.turns.extend(tail);
    }
}

/// Splits a streaming token feed into speakable chunks. First chunk may break early
/// (after a comma, >= MIN_FIRST chars) to minimize time-to-first-audio; subsequent chunks
/// break on sentence terminators . ! ? … or newline.
pub struct SentenceChunker { buf: String, first_done: bool }
impl SentenceChunker {
    const MIN_FIRST: usize = 12;
    pub fn new() -> Self { Self { buf: String::new(), first_done: false } }
    pub fn push(&mut self, delta: &str) -> Option<String> {
        self.buf.push_str(delta);
        if !self.first_done {
            if let Some(i) = self.buf.find(',') {
                if i >= Self::MIN_FIRST { return Some(self.take_through(i)); }
            }
        }
        if let Some(i) = self.buf.rfind(|c| matches!(c, '.' | '!' | '?' | '\n' | '…')) {
            return Some(self.take_through(i));
        }
        None
    }
    pub fn flush(&mut self) -> Option<String> {
        let s = self.buf.trim().to_string();
        self.buf.clear();
        (!s.is_empty()).then(|| { self.first_done = true; s })
    }
    fn take_through(&mut self, idx: usize) -> String {
        let end = self.buf[..=idx].len();
        let chunk: String = self.buf.drain(..end).collect();
        self.first_done = true;
        chunk.trim().to_string()
    }
}
impl Default for SentenceChunker { fn default() -> Self { Self::new() } }
```

- [ ] **Step 4: Run, verify pass** — `cargo test -p hearsay-core dialogue` → PASS. (Adjust the comma-break test expectation to the exact trim behavior if needed — keep behavior, fix the assert.)

- [ ] **Step 5: Commit**
```bash
git add crates/core/src/dialogue.rs crates/core/src/lib.rs
git commit -m "core: dialogue types + low-latency sentence chunker (TDD)"
```

---

## Task 4: The conversation FSM (the dialog brain — TDD, pure)

**Files:**
- Create: `crates/core/src/conversation.rs`; Modify `lib.rs` (`pub mod conversation;`)

This is the most important module. It is a **pure** state machine: `(State, Event) → (State, Vec<Effect>)`. No I/O, no threads, no time — the engine (Task 8) executes the `Effect`s and feeds back `Event`s. `turn_id` invalidates stale LLM/TTS output. This design makes barge-in correctness unit-testable.

- [ ] **Step 1: Write the failing tests (the full truth table)**

```rust
// crates/core/src/conversation.rs (#[cfg(test)])
use super::*;
fn drive(fsm: &mut Dialog, ev: Event) -> Vec<Effect> { fsm.on(ev) }

#[test]
fn happy_path_listen_think_speak_idle() {
    let mut d = Dialog::new();
    assert_eq!(d.state(), State::Idle);
    drive(&mut d, Event::UserSpeechStarted);
    assert_eq!(d.state(), State::Listening);
    let eff = drive(&mut d, Event::UserUtterance("what's the time".into()));
    assert_eq!(d.state(), State::Thinking);
    assert!(matches!(eff.as_slice(), [Effect::StartLlm { turn: 1, .. }]));
    // first sentence audio arrives
    drive(&mut d, Event::LlmDelta { turn: 1, text: "It's noon.".into() });
    let eff = drive(&mut d, Event::SpeakableChunk { turn: 1, text: "It's noon.".into() });
    assert!(matches!(eff.as_slice(), [Effect::Synthesize { turn: 1, .. }]));
    drive(&mut d, Event::PlaybackStarted { turn: 1 });
    assert_eq!(d.state(), State::Speaking);
    drive(&mut d, Event::LlmDone { turn: 1 });
    let eff = drive(&mut d, Event::PlaybackDrained { turn: 1 });
    assert_eq!(d.state(), State::Idle);
    assert!(eff.iter().any(|e| matches!(e, Effect::CommitAssistant { .. })));
}

#[test]
fn barge_in_during_speaking_cancels_and_relistens() {
    let mut d = Dialog::new();
    drive(&mut d, Event::UserSpeechStarted);
    drive(&mut d, Event::UserUtterance("tell me a long story".into()));
    drive(&mut d, Event::SpeakableChunk { turn: 1, text: "Once upon a time,".into() });
    drive(&mut d, Event::PlaybackStarted { turn: 1 });
    assert_eq!(d.state(), State::Speaking);
    let eff = drive(&mut d, Event::UserSpeechStarted); // BARGE-IN
    assert_eq!(d.state(), State::Listening);
    assert!(eff.contains(&Effect::CancelLlm { turn: 1 }));
    assert!(eff.contains(&Effect::CancelTts { turn: 1 }));
    assert!(eff.contains(&Effect::StopPlayback));
}

#[test]
fn stale_llm_output_after_bargein_is_ignored() {
    let mut d = Dialog::new();
    drive(&mut d, Event::UserSpeechStarted);
    drive(&mut d, Event::UserUtterance("a".into()));            // turn 1
    drive(&mut d, Event::PlaybackStarted { turn: 1 });
    drive(&mut d, Event::UserSpeechStarted);                    // barge-in -> turn bumps
    drive(&mut d, Event::UserUtterance("b".into()));            // turn 2
    // A late delta from turn 1 must produce NO effects.
    assert!(drive(&mut d, Event::LlmDelta { turn: 1, text: "late".into() }).is_empty());
    assert!(drive(&mut d, Event::SpeakableChunk { turn: 1, text: "late".into() }).is_empty());
}

#[test]
fn speech_start_then_quick_silence_is_not_a_turn() {
    // VAD start while Idle goes to Listening; a flush with empty text returns to Idle.
    let mut d = Dialog::new();
    drive(&mut d, Event::UserSpeechStarted);
    let eff = drive(&mut d, Event::UserUtterance("".into()));
    assert_eq!(d.state(), State::Idle);
    assert!(eff.is_empty());
}

#[test]
fn empty_llm_reply_returns_to_idle() {
    let mut d = Dialog::new();
    drive(&mut d, Event::UserSpeechStarted);
    drive(&mut d, Event::UserUtterance("hi".into()));
    let eff = drive(&mut d, Event::LlmDone { turn: 1 }); // no chunks were produced
    assert_eq!(d.state(), State::Idle);
    assert!(eff.iter().any(|e| matches!(e, Effect::CommitAssistant { .. })));
}
```

- [ ] **Step 2: Run, verify fail** — `cargo test -p hearsay-core conversation` → FAIL.

- [ ] **Step 3: Implement the FSM**

```rust
// crates/core/src/conversation.rs
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum State { Idle, Listening, Thinking, Speaking }

pub type TurnId = u64;

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Event {
    UserSpeechStarted,
    UserUtterance(String),
    LlmDelta { turn: TurnId, text: String },
    SpeakableChunk { turn: TurnId, text: String },
    LlmDone { turn: TurnId },
    PlaybackStarted { turn: TurnId },
    PlaybackDrained { turn: TurnId },
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Effect {
    StartLlm { turn: TurnId, user_text: String },
    Synthesize { turn: TurnId, text: String },
    CancelLlm { turn: TurnId },
    CancelTts { turn: TurnId },
    StopPlayback,
    CommitUser { text: String },
    CommitAssistant { turn: TurnId },
}

pub struct Dialog {
    state: State,
    turn: TurnId,
    spoke_any: bool, // did the current turn produce audio? (for commit/empty handling)
}

impl Dialog {
    pub fn new() -> Self { Self { state: State::Idle, turn: 0, spoke_any: false } }
    pub fn state(&self) -> State { self.state }
    pub fn turn(&self) -> TurnId { self.turn }

    pub fn on(&mut self, ev: Event) -> Vec<Effect> {
        match (self.state, ev) {
            // ---- Idle ----
            (State::Idle, Event::UserSpeechStarted) => { self.state = State::Listening; vec![] }

            // ---- Listening ----
            (State::Listening, Event::UserUtterance(t)) => {
                let t = t.trim().to_string();
                if t.is_empty() { self.state = State::Idle; return vec![]; }
                self.turn += 1; self.spoke_any = false; self.state = State::Thinking;
                vec![Effect::CommitUser { text: t.clone() },
                     Effect::StartLlm { turn: self.turn, user_text: t }]
            }

            // ---- Thinking / Speaking: stream LLM→TTS, gated by turn id ----
            (State::Thinking | State::Speaking, Event::SpeakableChunk { turn, text })
                if turn == self.turn =>
            {
                self.spoke_any = true;
                vec![Effect::Synthesize { turn, text }]
            }
            (State::Thinking | State::Speaking, Event::PlaybackStarted { turn })
                if turn == self.turn =>
            { self.state = State::Speaking; vec![] }

            (State::Speaking, Event::PlaybackDrained { turn }) if turn == self.turn => {
                self.state = State::Idle;
                vec![Effect::CommitAssistant { turn }]
            }
            // LLM finished but nothing left to play (empty reply / all drained already).
            (State::Thinking, Event::LlmDone { turn }) if turn == self.turn => {
                self.state = State::Idle;
                vec![Effect::CommitAssistant { turn }]
            }

            // ---- BARGE-IN: user speaks during Thinking or Speaking ----
            (State::Thinking | State::Speaking, Event::UserSpeechStarted) => {
                let stale = self.turn;
                self.state = State::Listening;
                vec![Effect::CancelLlm { turn: stale },
                     Effect::CancelTts { turn: stale },
                     Effect::StopPlayback]
            }

            // ---- Anything carrying a stale turn id, or irrelevant in this state: ignore ----
            _ => vec![],
        }
    }
}
impl Default for Dialog { fn default() -> Self { Self::new() } }
```

> Note for the implementer: the tests above pin the *contract*. If a test's exact `Effect` ordering/shape differs from this skeleton, fix the **implementation** to satisfy the test (the tests are the spec), not the reverse — except the `LlmDone`-before-drain bookkeeping, where you may need a `spoke_any`/pending-chunks counter so `CommitAssistant` fires exactly once. Add a `pending_audio` counter if `PlaybackDrained` can arrive before `LlmDone`; commit when both LLM is done AND audio drained. Add tests for that race.

- [ ] **Step 4: Run, verify pass** — `cargo test -p hearsay-core conversation` → PASS (all cases incl. the races).

- [ ] **Step 5: Add property/edge tests**

Add: double-barge-in (barge-in while already Listening is a no-op), `PlaybackDrained` for an old turn ignored, `LlmDelta` accumulation not required by FSM (deltas only matter to the chunker in the engine). Run → PASS.

- [ ] **Step 6: Commit**
```bash
git add crates/core/src/conversation.rs crates/core/src/lib.rs
git commit -m "core: pure conversation FSM with turn-id barge-in invalidation (TDD)"
```

---

## Task 5: Interruptible cpal player adapter (TDD where pure, manual where device)

**Files:**
- Create: `crates/adapters/src/player.rs`; Modify `adapters/lib.rs`, `adapters/Cargo.toml` (add `ringbuf`, `rubato`/reuse `resample.rs`)

- [ ] **Step 1: Failing test for the pure ring-buffer mixer (no device)**

```rust
// crates/adapters/src/player.rs (#[cfg(test)])
#[test]
fn ring_drains_in_order_and_counts() {
    let (mut prod, mut cons) = PlaybackRing::new(8);
    prod.push(&[0.1, 0.2, 0.3]);
    let mut out = [0.0f32; 2];
    cons.fill(&mut out); assert_eq!(out, [0.1, 0.2]);
    let mut out2 = [0.0f32; 2];
    cons.fill(&mut out2); assert_eq!(out2, [0.3, 0.0]); // underrun → silence pad
    assert_eq!(cons.played(), 4); // counts samples written to device incl. pad
}
#[test]
fn barge_stop_flushes_pending() {
    let (mut prod, mut cons) = PlaybackRing::new(8);
    prod.push(&[1.0, 1.0, 1.0]);
    cons.flush();
    let mut out = [9.0f32; 2];
    cons.fill(&mut out); assert_eq!(out, [0.0, 0.0]);
}
```

- [ ] **Step 2: Run, verify fail** — `cargo test -p hearsay-adapters player` → FAIL.

- [ ] **Step 3: Implement `PlaybackRing` (pure) + `CpalPlayer` (device)**

`PlaybackRing`: a `ringbuf::HeapRb<f32>` split into producer/consumer with an `AtomicBool` flush flag and `AtomicU64` played counter. `fill(out)` pops what it can, zero-pads underruns, bumps the counter; on `flush` it drains the consumer and zero-fills. `CpalPlayer` (impl `AudioPlayer`): builds a `cpal` output stream at the device's default sample rate, resampling 24 kHz→device rate via the copied `resample.rs`; the output callback calls `consumer.fill`. `enqueue` resamples + pushes; `barge_stop` sets flush; `played_samples`/`is_draining` from the ring. Output callback does NO allocation/locking (just pop + count).

- [ ] **Step 4: Run pure tests** — `cargo test -p hearsay-adapters player` → PASS.

- [ ] **Step 5: Manual device smoke test (ignored by default)**

Add `#[test] #[ignore]` `plays_a_tone` that enqueues a 440 Hz sine for 300 ms and asserts `played_samples() > 0` after a short sleep. Run: `cargo test -p hearsay-adapters plays_a_tone -- --ignored --nocapture`. Expected: you hear a beep, counter advances. (Manual — not in CI.)

- [ ] **Step 6: Commit**
```bash
git add crates/adapters/src/player.rs crates/adapters/src/lib.rs crates/adapters/Cargo.toml
git commit -m "adapters: interruptible cpal player (ring buffer, flush, played-counter)"
```

---

## Task 6: Python MisoTTS sidecar server (warm, streaming, cancellable)

**Files:**
- Create: `sidecars/miso_tts_server.py`, `sidecars/protocol.md`

The server loads MisoTTS once (warm), then serves a line-delimited JSON protocol over a Unix domain socket (path passed as argv). Rust is the client (Task 7). Synthesizes **per sentence** and streams raw `float32` PCM frames as length-prefixed binary; honors `cancel`.

**Protocol (`sidecars/protocol.md`):** newline-delimited JSON control on the socket, with binary audio framing:
- `← {"type":"ready","sample_rate":24000}` on model load.
- `→ {"type":"speak","turn":N,"text":"..."}` request.
- `← {"type":"audio","turn":N,"bytes":B}\n` + B bytes of little-endian f32 PCM (repeated per chunk).
- `← {"type":"done","turn":N}`
- `→ {"type":"cancel","turn":N}` — stop emitting for turn N ASAP.
- `→ {"type":"shutdown"}` or EOF → unload + exit (frees RAM).

- [ ] **Step 1: Implement the server**

```python
# sidecars/miso_tts_server.py — run inside vendor/MisoTTS/.venv
import json, os, socket, struct, sys, threading, re
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vendor", "MisoTTS"))
from generator import load_miso_8b  # patched for MPS in Task 0

SOCK = sys.argv[1]
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
gen = load_miso_8b(device=DEV, model_path_or_repo_id="MisoLabs/MisoTTS")
SR = int(gen.sample_rate)
cancel_turn = {"v": -1}
lock = threading.Lock()

def split_sentences(t):
    parts = re.split(r'(?<=[.!?…])\s+', t.strip())
    return [p for p in (s.strip() for s in parts) if p]

def handle(conn):
    f = conn.makefile("rwb")
    def send_json(o): f.write((json.dumps(o)+"\n").encode()); f.flush()
    send_json({"type":"ready","sample_rate":SR})
    for line in f:
        try: msg = json.loads(line)
        except Exception: continue
        if msg.get("type") == "cancel": cancel_turn["v"] = msg["turn"]; continue
        if msg.get("type") == "shutdown": break
        if msg.get("type") != "speak": continue
        turn, text = msg["turn"], msg["text"]
        with lock:
            for sent in split_sentences(text):
                if cancel_turn["v"] >= turn: break
                with torch.no_grad():
                    audio = gen.generate(text=sent, speaker=0, context=[], max_audio_length_ms=12000)
                if cancel_turn["v"] >= turn: break
                pcm = audio.detach().to("cpu", dtype=torch.float32).numpy().tobytes()
                send_json({"type":"audio","turn":turn,"bytes":len(pcm)}); f.write(pcm); f.flush()
            send_json({"type":"done","turn":turn})
    conn.close(); os._exit(0)

if os.path.exists(SOCK): os.unlink(SOCK)
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.bind(SOCK); s.listen(1)
conn, _ = s.accept(); handle(conn)
```

- [ ] **Step 2: Standalone smoke test (no Rust)**

```bash
cd /Users/michael/Repos/hearsay/vendor/MisoTTS
uv run python - <<'PY'
import socket, subprocess, json, struct, time, os, threading
sock="/tmp/miso_test.sock"
p=subprocess.Popen(["uv","run","python","../../sidecars/miso_tts_server.py",sock])
while not os.path.exists(sock): time.sleep(0.2)
c=socket.socket(socket.AF_UNIX); c.connect(sock); f=c.makefile("rwb")
print("server says:", f.readline().decode().strip())  # ready
f.write((json.dumps({"type":"speak","turn":1,"text":"Hey there. This is Miso."})+"\n").encode()); f.flush()
total=0
while True:
    line=json.loads(f.readline())
    if line["type"]=="audio": f.read(line["bytes"]); total+=line["bytes"]
    elif line["type"]=="done": break
print("got", total, "PCM bytes"); f.write((json.dumps({"type":"shutdown"})+"\n").encode()); f.flush()
PY
```
Expected: prints `ready`, then a nonzero PCM byte count. If MisoTTS errors on MPS here, fix the Task 0 patches.

- [ ] **Step 3: Commit**
```bash
git add sidecars/miso_tts_server.py sidecars/protocol.md
git commit -m "sidecar: warm streaming cancellable MisoTTS server over UDS"
```

---

## Task 7: Adapters — sidecar lifecycle, TTS client, LLM client

**Files:**
- Create: `crates/adapters/src/sidecar.rs`, `tts.rs`, `llm.rs`; Modify `adapters/lib.rs`, `Cargo.toml` (add `serde_json`, `ureq` for blocking HTTP)

- [ ] **Step 1: `sidecar.rs` — process the app owns (kill-on-Drop = memory mgmt)**

```rust
// crates/adapters/src/sidecar.rs
use std::process::{Child, Command};
pub struct Sidecar { child: Child, pub name: &'static str }
impl Sidecar {
    /// Spawn the MisoTTS server in its uv venv, bound to `sock`.
    pub fn miso(repo: &str, sock: &str) -> std::io::Result<Self> {
        let child = Command::new("uv")
            .current_dir(format!("{repo}/vendor/MisoTTS"))
            .args(["run","python","../../sidecars/miso_tts_server.py", sock])
            .spawn()?;
        Ok(Self { child, name: "miso-tts" })
    }
    /// Spawn mlx_lm.server for the dialog model (OpenAI-compatible on 127.0.0.1:port).
    pub fn mlx_llm(venv: &str, model: &str, port: u16) -> std::io::Result<Self> {
        let child = Command::new(format!("{venv}/bin/python"))
            .args(["-m","mlx_lm.server","--model",model,"--port",&port.to_string()])
            .spawn()?;
        Ok(Self { child, name: "mlx-llm" })
    }
}
impl Drop for Sidecar {
    fn drop(&mut self) { let _ = self.child.kill(); let _ = self.child.wait(); } // frees model RAM
}
```
Test (`#[test]`): spawn `Command::new("sleep").arg("30")` wrapped equivalently, assert `Drop` reaps it (pid gone). Keep the heavy spawns behind `#[ignore]` manual tests.

- [ ] **Step 2: `tts.rs` — `MisoTts` implements `SpeechSynthesizer`**

Connects to the UDS, reads `ready` (caches `sample_rate`), and `speak()` writes a `speak` JSON, then loops reading `audio`/`done` frames, calling `on_pcm(&chunk)` per audio frame; checks `cancel` between frames and on true sends `{"type":"cancel","turn":N}`. Pure-ish; the socket framing parser (`parse_header`) gets unit tests with canned bytes.

- [ ] **Step 3: `llm.rs` — `MlxChat` implements `LlmClient`**

Blocking streaming POST to `http://127.0.0.1:{port}/v1/chat/completions` with `"stream":true`, parsing SSE `data:` lines for `choices[0].delta.content`, calling `on_delta(text)` per token and checking `cancel` each line (drop the response to abort). Maps `Turn`→OpenAI messages. System prompt tuned for a **brief, spoken, conversational** persona (short sentences — they synthesize faster and feel snappier). Unit-test the SSE line parser with canned chunks.

- [ ] **Step 4: Run adapter tests** — `cargo test -p hearsay-adapters` → PASS (parsers + Drop). 

- [ ] **Step 5: Commit**
```bash
git add crates/adapters/src/{sidecar,tts,llm}.rs crates/adapters/src/lib.rs crates/adapters/Cargo.toml
git commit -m "adapters: sidecar lifecycle (kill-on-Drop), MisoTTS UDS client, mlx LLM SSE client"
```

---

## Task 8: Engine — the conversation loop (wires FSM to the world)

**Files:**
- Create: `crates/engine/src/conversation_loop.rs`, `events.rs`, `lib.rs`; `engine/Cargo.toml`

Adapt earshot's two-thread worker (`/Users/michael/Repos/earshot/crates/engine/src/worker.rs`, read in full). Capture/VAD/STT thread is earshot's aggregator+inference, but instead of emitting transcript to a UI sink, it feeds the FSM. A **controller** thread owns the `Dialog`, executes `Effect`s via the ports, and runs LLM/TTS work on a worker thread with an `AtomicBool` cancel per turn.

Key wiring:
- STT `utterance_ended` + the stabilizer's flushed text → `Event::UserUtterance`. VAD `speech_start` → `Event::UserSpeechStarted` (this is the **barge-in trigger** when state is Speaking — with the 250 ms-after-playback-start guard + sustained-speech guard from research finding #5).
- `Effect::StartLlm` → spawn a turn task: `llm.reply(history, cancel, on_delta=|d| chunker.push(d) -> maybe Event::SpeakableChunk)`. On finish, `flush()` → final chunk + `Event::LlmDone`.
- `Effect::Synthesize` → on the TTS worker, `tts.speak(text, cancel, on_pcm=|pcm| player.enqueue(pcm))`; first enqueue for a turn emits `Event::PlaybackStarted`; player drain emits `Event::PlaybackDrained`.
- `Effect::CancelLlm/CancelTts` → flip that turn's `AtomicBool`; `Effect::StopPlayback` → `player.barge_stop()`.
- `Effect::CommitUser/CommitAssistant` → update `Conversation`, emit `UiEvent::Turn{role,text}`.

- [ ] **Step 1: `events.rs`** — `UiCommand::{Go(EngineConfig), Stop, Shutdown}`, `UiEvent::{Started, Listening, Thinking, Speaking, Turn{role,text}, PartialUser{text}, Level{rms}, Error, Stopped}`, `EngineConfig{ mic, whisper_model, model_dir, llm_endpoint, miso_repo, persona }`, `ChannelSink`. Copy earshot's `ChannelSink` pattern.

- [ ] **Step 2: Integration test with fake ports (no real models)**

```rust
// crates/engine/tests/loop_bargein.rs
// Fakes: FakeLlm streams "Hello there. Nice to meet you." slowly; FakeTts turns text into N
// samples; FakePlayer records enqueues + flush. Fake AudioSource scripted to: speak "hi",
// go silent (utterance), then start speaking again mid-playback (barge-in).
#[test]
fn barge_in_stops_playback_and_starts_new_turn() {
    // Drive the loop; assert: FakePlayer saw barge_stop(); FakeLlm turn-1 cancel flag set;
    // a second StartLlm fired for turn 2; final committed history has user "hi" then the
    // interrupted assistant text truncated, then the new user turn.
}
```
Run: `cargo test -p hearsay-engine` → FAIL first, then implement until PASS. These fakes are the real test of barge-in end-to-end without GPUs.

- [ ] **Step 3: Implement `conversation_loop.rs`** to satisfy the integration test. Reuse earshot's aggregator/inference loop verbatim for the STT half; add the controller + turn-worker for the dialog half.

- [ ] **Step 4: Run** — `cargo test -p hearsay-engine` → PASS. Also `cargo test --workspace` green.

- [ ] **Step 5: Commit**
```bash
git add crates/engine/
git commit -m "engine: conversation loop — STT→FSM→LLM→TTS→player with barge-in (fake-port integration tests)"
```

---

## Task 9: Tauri shell + minimal Go UI + lifecycle

**Files:**
- Create: `src-tauri/{Cargo.toml,tauri.conf.json,Info.plist,build.rs,src/main.rs,capabilities/default.json}` (adapt earshot's), `web/` (copy earshot's React app, strip to one screen)

- [ ] **Step 1: Copy + adapt the Tauri shell** from earshot `src-tauri/` (read `src/main.rs` fully). Bundle id `ai.nicest.hearsay`, productName `Hearsay`, single main window (drop the overlay window). Keep the signing identity guidance (TCC stability — see earshot CLAUDE.md); create a `Hearsay Self-Signed` cert via the adapted `make cert`. Mic permission usage string in `Info.plist`.

- [ ] **Step 2: Commands + lifecycle** in `main.rs`:
  - `go()` → resolve `EngineConfig`; spawn `Sidecar::mlx_llm` + `Sidecar::miso` (store in Tauri state); wait for both ready (poll LLM `/v1/models`, TTS `ready`); send `UiCommand::Go`. Emit status while loading.
  - `stop()` → `UiCommand::Stop`; **drop the sidecars** (frees RAM).
  - On `WindowEvent::CloseRequested` / app exit → drop sidecars (kill-on-Drop). Add an **idle timer**: after N minutes Idle, auto-drop sidecars and show "sleeping — click Go" (the "don't leave it loaded" requirement).
  - All wire DTOs camelCase; domain types stay in Rust.

- [ ] **Step 3: The UI** (`web/src/App.tsx`): one big **Go**/**Stop** button (mic glow when Listening, waveform when Speaking), a scrolling two-column transcript (You / Hearsay) fed by `UiEvent::Turn` + `PartialUser`, a small status line (Listening/Thinking/Speaking + a RAM/“models loaded” chip), and a "wear headphones for best barge-in" hint. Match earshot's Tabler/Tailwind tokens (per the match-existing-ui-patterns rule). No emojis.

- [ ] **Step 4: Build + run** — `make dev`. Grant mic permission. Click **Go**. Expected: sidecars load (status shows progress), then Listening. (Functional dogfright happens in Task 11.)

- [ ] **Step 5: Commit**
```bash
git add src-tauri/ web/
git commit -m "tauri: Hearsay shell — Go/Stop, transcript UI, sidecar lifecycle + idle unload"
```

---

## Task 10: Review & harden — no-mercy, hex-audit, simplify

- [ ] **Step 1: `/hex-audit`** the workspace. The whole point of forking earshot is the hexagon stays intact: `core` pure (no reqwest/cpal/process/serde-wire leaks), adapters behind ports, engine orchestrating, Tauri as the only wire-DTO site. Fix every leak it finds (e.g., if any `std::process` or socket type crept into `core`). Re-run until clean.

- [ ] **Step 2: `/no-mercy`** review of all new code (`conversation.rs`, `player.rs`, `tts.rs`, `llm.rs`, `sidecar.rs`, `conversation_loop.rs`, `main.rs`). Fix every cited finding: races on the cancel flags, unbounded channels, audio-callback allocation, error swallowing, the `expect`/`unwrap` audit, the barge-in guard timings. No participation trophies.

- [ ] **Step 3: Coverage check** — every FSM transition + the barge-in + stale-turn + empty-reply paths have tests; the chunker, SSE parser, socket framer, and ring buffer are unit-tested. Add any missing.

- [ ] **Step 4: `/simplify`** — run the `code-simplifier:code-simplifier` agent on the changed code. Apply quality-only simplifications. Re-run tests.

- [ ] **Step 5: Lint + full test** — `make lint` (`clippy --workspace --all-targets -D warnings`) and `cargo test --workspace` both green.

- [ ] **Step 6: Commit** each fix batch with a descriptive message.

---

## Task 11: Deploy-it + dogfood

- [ ] **Step 1: `/deploy-it`** — build the signed release bundle: adapt earshot's `make app` → `target/release/bundle/macos/Hearsay.app`. Write `DEPLOY.md` (the runbook: cert, build, the uv/venv + model prerequisites the .app needs, first-run model downloads, permissions). Note: the .app shells out to `uv`/`mlx_lm` — DEPLOY.md documents that these must be installed (this is a personal test app, not a notarized distributable).

- [ ] **Step 2: Launch the bundle**, grant mic permission, relaunch (TCC).

- [ ] **Step 3: `/dogfood`** — actually have a conversation:
  1. Click **Go**; confirm both models load and status → Listening.
  2. Say "Hey, can you hear me?" → confirm STT transcript, Thinking, then the **Miso voice** replies. Screenshot.
  3. Ask something with a long answer; **interrupt mid-sentence** → confirm audible cut-off < ~300 ms, state → Listening, your interruption is transcribed and answered. This is the headline feature — verify it on real audio (headphones).
  4. Click **Stop** → confirm sidecars die (`pgrep -f 'mlx_lm|miso_tts_server'` empty → RAM freed). Wait out the idle timer → confirm auto-unload.
  Record findings; fix every bug (latency, double-speak, stuck states, zombie sidecars). Re-verify after each fix with screenshots/`pics-or-it-didnt-happen`.

- [ ] **Step 4: Final commit + README** documenting what it is, the headphones caveat, the memory story, and how to run.

---

## Self-Review (writing-plans checklist)

**Spec coverage:**
- "make a new repo in repos" → Task 1 (`~/Repos/hearsay`). ✓
- "how we load Qwen + ensure it doesn't stay loaded" → small Qwen-Instruct via `mlx_lm.server` as an app-owned sidecar, killed on Stop/close/idle (Tasks 7, 9). ✓
- "repurposable from earshot / whisper-transcribe" → fork earshot's STT/VAD/aggregator/stabilizer + Tauri shell (Tasks 1, 8, 9); whisper-rs real-time replaces the batch whisper-transcribe. ✓
- "best-practice for voice agents that can get cut off" → research captured in Context; barge-in FSM (Task 4) + engine guards (Task 8). ✓
- "local conversational agent, don't care about sessions" → no session persistence; one Go button (Task 9). ✓
- "Rust + Tauri" → entire stack. (User said "tauri + electron" — Tauri is the Rust-native choice; Electron is JS and would abandon the Rust core, so Tauri only. Recorded in ADR.) ✓
- "open it, click Go, converse with Miso" → Task 9 UI + Task 11 dogfood. ✓
- tests / no-mercy / hex-audit / deploy-it / dogfood / simplify → Tasks 4–11 thread them. ✓

**Placeholder scan:** No TBDs; every code step has real code. The one deliberate decision-gate is Task 0 Step 5 (spike verdict) — that's a measurement, not a placeholder.

**Type consistency:** `Turn`/`Role` (dialogue) used by `LlmClient` and FSM commits; `Effect`/`Event`/`State`/`TurnId` consistent across Task 4 and Task 8; `AudioPlayer::barge_stop` matches `Effect::StopPlayback` semantics; `SpeechSynthesizer::speak(text, cancel, on_pcm)` matches the sidecar protocol and Task 8 wiring.

**Open risk (carried to council):** MisoTTS-8B RTF on MPS (Task 0 gates it); barge-in echo without AEC (headphones v1).
