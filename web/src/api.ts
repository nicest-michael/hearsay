import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

export type ConvState = "idle" | "listening" | "thinking" | "speaking";

export interface Turn {
  role: "user" | "assistant";
  text: string;
}

export const api = {
  go: () => invoke<void>("go"),
  stop: () => invoke<void>("stop"),
};

export const events = {
  onStatus: (cb: (s: string) => void): Promise<UnlistenFn> =>
    listen<string>("status", (e) => cb(e.payload)),
  onStarted: (cb: () => void): Promise<UnlistenFn> => listen("started", () => cb()),
  onState: (cb: (s: ConvState) => void): Promise<UnlistenFn> =>
    listen<ConvState>("state", (e) => cb(e.payload)),
  onTurn: (cb: (t: Turn) => void): Promise<UnlistenFn> =>
    listen<Turn>("turn", (e) => cb(e.payload)),
  onLevel: (cb: (rms: number) => void): Promise<UnlistenFn> =>
    listen<number>("level", (e) => cb(e.payload)),
  onError: (cb: (m: string) => void): Promise<UnlistenFn> =>
    listen<string>("error", (e) => cb(e.payload)),
  onStopped: (cb: () => void): Promise<UnlistenFn> => listen("stopped", () => cb()),
};
