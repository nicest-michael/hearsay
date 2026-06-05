# TTS spike results (2026-06-05, M5 Pro 48GB)

## MisoTTS 8B (PyTorch MPS, fp16) — REJECTED for real-time
param_dtype=float16, mask_device=mps:0 (fp16-on-MPS confirmed), Mimi+watermark on CPU, warm:
- "Hey, what's up?"            audio 0.32s  gen  4.63s  RTF 14.5  median frame 893ms
- "I can hear you loud and clear."  audio 2.48s  gen 24.68s  RTF  9.95  median frame 766ms
Load 108s, peak RSS 26.9GB. Cause: 31 sequential 300M-decoder forwards per 80ms frame
are kernel-launch bound on Metal. Architecture tax, not tuning.

## Kokoro 82M (mlx-audio) — SELECTED as default real-time voice
load 5.3s cold / 0.11s warm; first gen compiles kernels (~23s, warm on startup); voice af_heart:
- "Yeah, totally."                       audio 1.68s  gen 0.074s  RTF 0.044
- "Hey, what's up? I can hear you..."     audio 3.05s  gen 0.162s  RTF 0.053
- "That's a great question, ... let me think."  audio 5.35s  gen 0.269s  RTF 0.050
~20x faster than real-time. Needs misaki[en]. sr=24000.
