#!/usr/bin/env python
"""Hearsay TTS sidecar — MisoTTS 8B via PyTorch/MPS ("highest quality, NOT real-time").

Speaks the same Hearsay TTS protocol (see protocol.md) as the Kokoro sidecar, so the
Rust `SidecarTts` client is identical. MisoTTS is the emotive 8B voice the project was
built around; on Apple MPS it runs ~10x slower than real-time (see ADR 0001), so this is
an optional voice, not the default. The transformer runs on MPS; the Mimi codec +
watermarker run on CPU (dodges the MPS float64 limitation). Cancellation is checked
**every frame** so a barge-in frees the GPU within ~one frame.

Usage:  python miso_server.py <socket_path> [speaker]
Run inside vendor/MisoTTS/.venv (the patched MisoTTS env).
"""
import json
import os
import re
import socket
import sys
import threading
import time

os.environ.setdefault("NO_TORCH_COMPILE", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

SOCK = sys.argv[1]
SPEAKER = int(sys.argv[2]) if len(sys.argv) > 2 else 0

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vendor", "MisoTTS"))
import numpy as np
import torch
from generator import load_miso_8b  # tokenizer patched to ungated mirror

FRAME_MS = 80.0
_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")


def split_sentences(text):
    return [s for s in (p.strip() for p in _SENT_SPLIT.split(text.strip())) if s]


def watch_parent(ppid):
    while True:
        if os.getppid() != ppid:
            os._exit(0)
        time.sleep(1.0)


def main():
    threading.Thread(target=watch_parent, args=(os.getppid(),), daemon=True).start()

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    gen = load_miso_8b(device=dev, model_path_or_repo_id="MisoLabs/MisoTTS", dtype=torch.float16)
    torch.set_grad_enabled(False)
    # Mimi codec on CPU avoids MPS float64 issues; it runs once per sentence (cheap).
    try:
        gen._audio_tokenizer.to("cpu")
    except Exception:
        pass
    sr = int(gen.sample_rate)
    # warm one frame so the first real frame isn't a cold-kernel outlier
    _warm(gen)

    if os.path.exists(SOCK):
        os.unlink(SOCK)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCK)
    srv.listen(1)
    while True:
        conn, _ = srv.accept()
        if serve_connection(conn, gen, sr):
            break
    os._exit(0)


def _warm(gen):
    try:
        list(synth_frames(gen, "Hi.", lambda: False, max_ms=1500))
    except Exception as e:
        print(f"[miso] warmup warning: {e}", file=sys.stderr)


def synth_frames(gen, text, cancelled, max_ms=12000):
    """Yield 24kHz f32 PCM for one sentence, checking `cancelled()` every frame."""
    gen._model.reset_caches()
    seg_tok, seg_mask = gen._tokenize_text_segment(text, speaker=SPEAKER)
    curr = seg_tok.long().to(gen.device).unsqueeze(0)
    curr_mask = seg_mask.bool().to(gen.device).unsqueeze(0)
    pos = torch.arange(0, seg_tok.size(0)).unsqueeze(0).long().to(gen.device)
    frames = []
    for _ in range(int(max_ms / FRAME_MS)):
        if cancelled():
            break
        sample = gen._model.generate_frame(curr, curr_mask, pos, 0.9, 50)
        if torch.all(sample == 0):
            break
        frames.append(sample)
        curr = torch.cat([sample, torch.zeros(1, 1).long().to(gen.device)], dim=1).unsqueeze(1)
        curr_mask = torch.cat(
            [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(gen.device)], dim=1
        ).unsqueeze(1)
        pos = pos[:, -1:] + 1
    if not frames:
        return
    codes = torch.stack(frames).permute(1, 2, 0).to("cpu")
    audio = gen._audio_tokenizer.decode(codes).squeeze(0).squeeze(0)
    yield np.ascontiguousarray(audio.detach().to("cpu", dtype=torch.float32).numpy()).tobytes()


def serve_connection(conn, gen, sr):
    rf = conn.makefile("rb")
    wf = conn.makefile("wb")
    lock = threading.Lock()
    state = {"cancel_through": -1, "shutdown": False, "explicit_shutdown": False}
    speak_queue = []
    cv = threading.Condition()

    def send_json(o):
        with lock:
            wf.write((json.dumps(o) + "\n").encode())
            wf.flush()

    def send_audio(turn, pcm):
        with lock:
            wf.write((json.dumps({"type": "audio", "turn": turn, "bytes": len(pcm)}) + "\n").encode())
            wf.write(pcm)
            wf.flush()

    def reader():
        for line in rf:
            try:
                msg = json.loads(line)
            except Exception:
                continue
            t = msg.get("type")
            if t == "speak":
                with cv:
                    speak_queue.append((msg["turn"], msg["text"]))
                    cv.notify()
            elif t == "cancel":
                state["cancel_through"] = max(state["cancel_through"], int(msg["turn"]))
            elif t == "shutdown":
                with cv:
                    state["shutdown"] = True
                    state["explicit_shutdown"] = True
                    cv.notify()
                return
        with cv:
            state["shutdown"] = True
            cv.notify()

    threading.Thread(target=reader, daemon=True).start()
    send_json({"type": "ready", "sample_rate": sr})

    while True:
        with cv:
            while not speak_queue and not state["shutdown"]:
                cv.wait()
            if state["shutdown"] and not speak_queue:
                break
            turn, text = speak_queue.pop(0)

        def cancelled():
            return state["cancel_through"] >= turn or state["shutdown"]

        try:
            for sentence in split_sentences(text):
                if cancelled():
                    break
                for pcm in synth_frames(gen, sentence, cancelled):
                    if cancelled():
                        break
                    send_audio(turn, pcm)
            send_json({"type": "done", "turn": turn})
        except Exception as e:
            send_json({"type": "error", "turn": turn, "message": str(e)})

    try:
        conn.close()
    except Exception:
        pass
    return bool(state.get("explicit_shutdown"))


if __name__ == "__main__":
    main()
