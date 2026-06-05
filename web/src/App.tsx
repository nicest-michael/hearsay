import { useEffect, useRef, useState } from "react";
import {
  IconMicrophone,
  IconPlayerStopFilled,
  IconHeadphones,
} from "@tabler/icons-react";
import { api, events, type ConvState, type Turn } from "./api";

type Phase = "stopped" | "starting" | "running";

const STATE_META: Record<ConvState, { label: string; color: string; ring: string }> = {
  idle: { label: "Listening for you", color: "#6b7280", ring: "rgba(107,114,128,0.35)" },
  listening: { label: "Listening…", color: "#38bdf8", ring: "rgba(56,189,248,0.55)" },
  thinking: { label: "Thinking…", color: "#f59e0b", ring: "rgba(245,158,11,0.55)" },
  speaking: { label: "Speaking…", color: "#34d399", ring: "rgba(52,211,153,0.55)" },
};

export default function App() {
  const [phase, setPhase] = useState<Phase>("stopped");
  const [state, setState] = useState<ConvState>("idle");
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [level, setLevel] = useState(0);
  const [turns, setTurns] = useState<Turn[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const unlisten = Promise.all([
      events.onStatus((s) => setStatus(s)),
      events.onStarted(() => {
        setPhase("running");
        setStatus("");
        setError("");
      }),
      events.onState((s) => setState(s)),
      events.onTurn((t) => setTurns((prev) => [...prev, t])),
      events.onLevel((rms) => setLevel(rms)),
      events.onError((m) => setError(m)),
      events.onStopped(() => {
        setPhase("stopped");
        setState("idle");
        setLevel(0);
      }),
    ]);
    return () => {
      unlisten.then((us) => us.forEach((u) => u()));
    };
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [turns]);

  const onGo = async () => {
    setError("");
    setPhase("starting");
    setStatus("Starting…");
    try {
      await api.go();
    } catch (e) {
      setError(String(e));
      setPhase("stopped");
    }
  };

  const onStop = async () => {
    await api.stop();
    setPhase("stopped");
  };

  const meta = STATE_META[state];
  // Listening pulse scales the ring with mic level; other states have a steady glow.
  const pulse = phase === "running" && state === "listening" ? 1 + Math.min(level * 6, 0.6) : 1;

  return (
    <div className="flex h-screen flex-col bg-[#0a0a0b] text-neutral-100">
      <header
        data-tauri-drag-region
        className="flex select-none items-center justify-between px-5 pb-2 pt-8"
      >
        <div>
          <h1 className="text-lg font-semibold tracking-tight">Hearsay</h1>
          <p className="text-xs text-neutral-500">a local voice you can talk over</p>
        </div>
        {phase === "running" && (
          <span
            className="rounded-full px-2.5 py-1 text-[11px] font-medium"
            style={{ background: meta.ring, color: meta.color }}
          >
            {meta.label}
          </span>
        )}
      </header>

      {/* Orb / control */}
      <div className="flex flex-col items-center justify-center gap-4 py-6">
        <button
          onClick={phase === "stopped" ? onGo : onStop}
          disabled={phase === "starting"}
          className="group relative flex h-36 w-36 items-center justify-center rounded-full transition-transform active:scale-95 disabled:cursor-wait"
          aria-label={phase === "stopped" ? "Start conversation" : "Stop conversation"}
        >
          {/* animated ring */}
          <span
            className="absolute inset-0 rounded-full transition-all duration-150"
            style={{
              boxShadow: `0 0 0 2px ${meta.ring}, 0 0 48px 8px ${
                phase === "running" ? meta.ring : "transparent"
              }`,
              transform: `scale(${pulse})`,
            }}
          />
          {phase === "running" && state === "thinking" && (
            <span
              className="absolute inset-2 animate-spin rounded-full border-2 border-transparent"
              style={{ borderTopColor: meta.color, animationDuration: "1.1s" }}
            />
          )}
          {phase === "running" && state === "speaking" && <Waveform color={meta.color} />}
          <span
            className="relative flex h-28 w-28 items-center justify-center rounded-full"
            style={{ background: "#141417", boxShadow: "inset 0 1px 1px rgba(255,255,255,0.05)" }}
          >
            {phase === "stopped" ? (
              <span className="text-xl font-semibold tracking-wide text-neutral-200">Go</span>
            ) : phase === "starting" ? (
              <span
                className="h-6 w-6 animate-spin rounded-full border-2 border-neutral-600 border-t-neutral-200"
                style={{ animationDuration: "0.9s" }}
              />
            ) : state === "speaking" ? (
              <IconMicrophone size={34} className="text-neutral-700" />
            ) : (
              <IconMicrophone size={34} style={{ color: meta.color }} />
            )}
          </span>
        </button>

        <div className="h-5 text-center text-sm text-neutral-400">
          {error ? (
            <span className="text-rose-400">{error}</span>
          ) : phase === "starting" ? (
            status || "Starting…"
          ) : phase === "running" ? (
            <button onClick={onStop} className="inline-flex items-center gap-1.5 text-neutral-500 hover:text-neutral-300">
              <IconPlayerStopFilled size={13} /> Stop
            </button>
          ) : (
            "Click Go and start talking"
          )}
        </div>
      </div>

      {/* Transcript */}
      <div ref={scrollRef} className="flex-1 space-y-3 overflow-y-auto px-5 pb-3">
        {turns.length === 0 && phase !== "starting" && (
          <p className="mt-6 text-center text-sm text-neutral-600">
            Your conversation will appear here.
          </p>
        )}
        {turns.map((t, i) => (
          <Bubble key={i} turn={t} />
        ))}
      </div>

      <footer className="flex select-none items-center justify-center gap-1.5 border-t border-neutral-900 px-5 py-2.5 text-[11px] text-neutral-600">
        <IconHeadphones size={13} />
        Wear headphones so it can hear you interrupt it.
      </footer>
    </div>
  );
}

function Bubble({ turn }: { turn: Turn }) {
  const isUser = turn.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[80%] rounded-2xl px-3.5 py-2 text-[13px] leading-snug ${
          isUser
            ? "rounded-br-sm bg-sky-500/15 text-sky-100"
            : "rounded-bl-sm bg-neutral-800/70 text-neutral-200"
        }`}
      >
        <span className="mb-0.5 block text-[10px] font-medium uppercase tracking-wide opacity-40">
          {isUser ? "You" : "Hearsay"}
        </span>
        {turn.text}
      </div>
    </div>
  );
}

function Waveform({ color }: { color: string }) {
  return (
    <span className="absolute inset-0 flex items-center justify-center gap-1">
      {[0, 1, 2, 3, 4].map((i) => (
        <span
          key={i}
          className="w-1 rounded-full"
          style={{
            background: color,
            height: 14 + (i % 2 === 0 ? 18 : 30),
            opacity: 0.8,
            animation: `hs-bar 0.7s ease-in-out ${i * 0.09}s infinite alternate`,
          }}
        />
      ))}
      <style>{`@keyframes hs-bar { from { transform: scaleY(0.4) } to { transform: scaleY(1) } }`}</style>
    </span>
  );
}
