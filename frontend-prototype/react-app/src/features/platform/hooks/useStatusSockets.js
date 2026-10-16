import { useEffect, useRef } from "react";
import { platformApi } from "../../../api/platformClient";

export function useAssetStatusSocket(projectId, { onSnapshot, onFallback } = {}) {
  const callbacks = useRef({ onSnapshot, onFallback });
  callbacks.current = { onSnapshot, onFallback };
  useEffect(() => {
    if (!projectId) return undefined;
    let cancelled = false;
    let socket;
    let fallback;
    const startFallback = () => {
      if (!fallback && !cancelled) fallback = window.setInterval(() => callbacks.current.onFallback?.(), 2000);
    };
    try {
      socket = new WebSocket(platformApi.watchProjectAssetsUrl(projectId));
      socket.onmessage = (event) => {
        if (cancelled) return;
        try {
          const message = JSON.parse(event.data);
          if (message.type === "assets.snapshot") callbacks.current.onSnapshot?.(message.items || []);
        } catch { /* ignore malformed events */ }
      };
      socket.onerror = startFallback;
      socket.onclose = startFallback;
    } catch { startFallback(); }
    return () => { cancelled = true; socket?.close(); if (fallback) window.clearInterval(fallback); };
  }, [projectId]);
}

export function useMotionPipelineSocket(pipelineId, { enabled = true, onSnapshot, onFallback } = {}) {
  const callbacks = useRef({ onSnapshot, onFallback });
  callbacks.current = { onSnapshot, onFallback };
  useEffect(() => {
    if (!pipelineId || !enabled) return undefined;
    let cancelled = false;
    let socket;
    let fallback;
    const applyFallback = () => { if (!cancelled) callbacks.current.onFallback?.(); };
    const startFallback = () => {
      if (!fallback && !cancelled) fallback = window.setInterval(applyFallback, 1500);
    };
    try {
      socket = new WebSocket(platformApi.watchMotionPipelineUrl(pipelineId));
      socket.onmessage = (event) => {
        if (cancelled) return;
        try {
          const message = JSON.parse(event.data);
          if (message.type === "motion_pipeline.snapshot") callbacks.current.onSnapshot?.(message.item);
        } catch { /* ignore malformed events */ }
      };
      socket.onerror = startFallback;
      socket.onclose = startFallback;
    } catch { startFallback(); }
    return () => { cancelled = true; socket?.close(); if (fallback) window.clearInterval(fallback); };
  }, [enabled, pipelineId]);
}
