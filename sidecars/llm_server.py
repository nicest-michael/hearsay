#!/usr/bin/env python
"""Watchdog wrapper around `mlx_lm.server`.

`mlx_lm.server` is third-party and has no parent-death guard, so on a hard crash of the
Hearsay app it would leak (~2 GB resident). This wrapper runs it as a child, kills that
child if the parent (the app) dies or on SIGTERM, so the LLM sidecar is as crash-safe as
the TTS ones. All argv are forwarded to `mlx_lm.server`.

Usage:  python llm_server.py --model <id> --port <n> --host 127.0.0.1
Run inside sidecars/.venv.
"""
import os
import signal
import subprocess
import sys
import threading
import time

VENV_BIN = os.path.dirname(os.path.abspath(sys.executable))
child = subprocess.Popen([os.path.join(VENV_BIN, "mlx_lm.server"), *sys.argv[1:]])


def reap(*_):
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            child.kill()
    os._exit(0)


signal.signal(signal.SIGTERM, reap)
signal.signal(signal.SIGINT, reap)


def watch_parent(ppid):
    while True:
        if os.getppid() != ppid:
            reap()
        time.sleep(1.0)


threading.Thread(target=watch_parent, args=(os.getppid(),), daemon=True).start()
child.wait()
