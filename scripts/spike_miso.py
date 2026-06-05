#!/usr/bin/env python
"""Hearsay TTS spike: prove MisoTTS 8B on Apple MPS and MEASURE real-time-factor.

Council directive (2026-06-05): measure RTF before building the pipeline around it.
This also prototypes the eventual sidecar's hot loop: transformer on MPS, Mimi codec
+ watermark on CPU (dodges the MPS float64 limitation upstream's run_misotts.py flags),
with a per-frame timing/cancel hook.

Run:  cd vendor/MisoTTS && uv run python ../../scripts/spike_miso.py
"""
import os, sys, time, resource
os.environ["NO_TORCH_COMPILE"] = "1"
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # let unimplemented ops fall to CPU
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../vendor/MisoTTS")

import torch, torchaudio
from generator import load_miso_8b

def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)  # macOS: bytes

FRAME_MS = 80.0  # MisoTTS: max_generation_len = ms/80 -> 12.5 fps Mimi frames

def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[spike] device={dev} torch={torch.__version__}", flush=True)
    t0 = time.time()
    gen = load_miso_8b(device=dev, model_path_or_repo_id="MisoLabs/MisoTTS", dtype=torch.float16)
    print(f"[spike] model loaded in {time.time()-t0:.1f}s  sample_rate={gen.sample_rate}  rss={rss_gb():.1f}GB", flush=True)

    # Move the Mimi codec + watermarker to CPU to avoid MPS float64 ops in the codec/DSP.
    try:
        gen._audio_tokenizer.to("cpu")
        print("[spike] moved Mimi codec -> CPU", flush=True)
    except Exception as e:
        print(f"[spike] WARN could not move Mimi to CPU: {e}", flush=True)

    def synth(text, max_ms=12000, frame_chunk=12, cancel_after=None):
        """Replicates Generator.generate frame loop with timing + chunked CPU decode.
        Returns (audio_cpu, stats). cancel_after: stop after N frames (cooperative-cancel test)."""
        gen._model.reset_caches()
        seg_tok, seg_mask = gen._tokenize_text_segment(text, speaker=0)
        prompt = seg_tok.long().to(gen.device)
        prompt_mask = seg_mask.bool().to(gen.device)
        curr = prompt.unsqueeze(0)
        curr_mask = prompt_mask.unsqueeze(0)
        pos = torch.arange(0, prompt.size(0)).unsqueeze(0).long().to(gen.device)
        max_len = int(max_ms / FRAME_MS)
        frames, frame_ms = [], []
        first_audio_s = None
        loop_t0 = time.time()
        for i in range(max_len):
            ft = time.time()
            sample = gen._model.generate_frame(curr, curr_mask, pos, 0.9, 50)
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
            frame_ms.append((time.time() - ft) * 1000)
            if torch.all(sample == 0):
                break
            frames.append(sample)
            # cooperative-cancel / first-chunk decode probe
            if first_audio_s is None and len(frames) >= frame_chunk:
                codes = torch.stack(frames).permute(1, 2, 0).to("cpu")
                _ = gen._audio_tokenizer.decode(codes)
                first_audio_s = time.time() - loop_t0
            if cancel_after is not None and len(frames) >= cancel_after:
                break
            curr = torch.cat([sample, torch.zeros(1, 1).long().to(gen.device)], dim=1).unsqueeze(1)
            curr_mask = torch.cat(
                [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(gen.device)], dim=1
            ).unsqueeze(1)
            pos = pos[:, -1:] + 1
        gen_t = time.time() - loop_t0
        codes = torch.stack(frames).permute(1, 2, 0).to("cpu")
        dec_t0 = time.time()
        audio = gen._audio_tokenizer.decode(codes).squeeze(0).squeeze(0)
        dec_t = time.time() - dec_t0
        audio_s = len(frames) * FRAME_MS / 1000.0
        stats = dict(frames=len(frames), gen_t=gen_t, dec_t=dec_t, audio_s=audio_s,
                     rtf=gen_t / max(audio_s, 1e-9),
                     med_frame_ms=sorted(frame_ms)[len(frame_ms)//2] if frame_ms else 0,
                     first_audio_s=first_audio_s)
        return audio, stats

    tests = [
        "Hey, what's up?",
        "I can hear you loud and clear.",
        "Honestly, that's a great question, and I think there's a lot to unpack there.",
    ]
    # warm up MPS kernels (first frame is always slow)
    print("[spike] warmup...", flush=True)
    synth("Hi.", max_ms=2000)
    print(f"[spike] warmup done, rss={rss_gb():.1f}GB\n", flush=True)

    for t in tests:
        audio, s = synth(t)
        print(f"[spike] text={t!r}", flush=True)
        print(f"        frames={s['frames']}  audio={s['audio_s']:.2f}s  gen={s['gen_t']:.2f}s  "
              f"RTF={s['rtf']:.2f}  median_frame={s['med_frame_ms']:.0f}ms  "
              f"first_audio~{(s['first_audio_s'] or 0):.2f}s  mimi_decode={s['dec_t']*1000:.0f}ms", flush=True)
        fn = f"/tmp/miso_spike_{abs(hash(t))%9999}.wav"
        torchaudio.save(fn, audio.unsqueeze(0).cpu().float(), gen.sample_rate)
        print(f"        wrote {fn}", flush=True)

    # cooperative-cancel probe: how fast does the loop stop?
    ct0 = time.time(); synth(tests[2], cancel_after=5); print(f"\n[spike] cancel-after-5-frames returned in {time.time()-ct0:.2f}s (GPU-free latency on barge-in)", flush=True)
    print(f"[spike] PEAK RSS = {rss_gb():.1f}GB", flush=True)
    print("[spike] DONE", flush=True)

if __name__ == "__main__":
    main()
