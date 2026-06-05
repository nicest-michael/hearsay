#!/usr/bin/env python
"""Hearsay TTS sidecar — Kokoro 82M via mlx-audio (default, real-time voice).

Loads Kokoro once, warms it (the first synth compiles MLX kernels), then serves the
Hearsay TTS protocol (see protocol.md) over a Unix domain socket. Synthesizes
sentence-by-sentence and streams little-endian f32 PCM; honors cancel between
sentences; self-exits if the parent process dies.

Usage:  python kokoro_server.py <socket_path> [voice] [model_repo]
Run inside sidecars/.venv (mlx-audio + misaki[en] installed).
"""
import json
import os
import re
import socket
import sys
import threading
import time

SOCK = sys.argv[1]
VOICE = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("HEARSAY_TTS_VOICE", "af_heart")
MODEL = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("HEARSAY_TTS_MODEL", "prince-canuma/Kokoro-82M")

import numpy as np
from mlx_audio.tts.utils import load_model

_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")


def split_sentences(text):
    return [s for s in (p.strip() for p in _SENT_SPLIT.split(text.strip())) if s]


def watch_parent(ppid):
    """Self-terminate if the parent (the app) dies — no leaked model process."""
    while True:
        if os.getppid() != ppid:
            os._exit(0)
        time.sleep(1.0)


def main():
    parent = os.getppid()
    threading.Thread(target=watch_parent, args=(parent,), daemon=True).start()

    model = load_model(MODEL)
    # Warm: first generation compiles MLX kernels (~20s). Do it before announcing ready
    # so the first real reply is fast.
    sr = 24000
    for seg in model.generate(text="Hello.", voice=VOICE, speed=1.0):
        sr = int(seg.sample_rate)

    if os.path.exists(SOCK):
        os.unlink(SOCK)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCK)
    srv.listen(1)
    conn, _ = srv.accept()

    rf = conn.makefile("rb")
    wf = conn.makefile("wb")
    lock = threading.Lock()
    state = {"cancel_through": -1, "shutdown": False}

    def send_json(obj):
        with lock:
            wf.write((json.dumps(obj) + "\n").encode())
            wf.flush()

    def send_audio(turn, pcm_bytes):
        with lock:
            wf.write((json.dumps({"type": "audio", "turn": turn, "bytes": len(pcm_bytes)}) + "\n").encode())
            wf.write(pcm_bytes)
            wf.flush()

    # Reader thread: control messages -> a speak queue + cancel/shutdown flags.
    speak_queue = []
    cv = threading.Condition()

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
                    cv.notify()
                return
        # EOF -> client gone
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
        if state["cancel_through"] >= turn:
            send_json({"type": "done", "turn": turn})
            continue
        try:
            for sentence in split_sentences(text):
                if state["cancel_through"] >= turn or state["shutdown"]:
                    break
                for seg in model.generate(text=sentence, voice=VOICE, speed=1.0):
                    if state["cancel_through"] >= turn or state["shutdown"]:
                        break
                    pcm = np.ascontiguousarray(np.asarray(seg.audio, dtype=np.float32)).tobytes()
                    send_audio(turn, pcm)
            send_json({"type": "done", "turn": turn})
        except Exception as e:  # never let one bad utterance kill the sidecar
            send_json({"type": "error", "turn": turn, "message": str(e)})

    try:
        conn.close()
    finally:
        os._exit(0)


if __name__ == "__main__":
    main()
