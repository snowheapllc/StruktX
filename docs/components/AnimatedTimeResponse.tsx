"use client";

import { useEffect, useRef, useState } from "react";
import { gsap } from "gsap";

export type TimeResponse = { city: string; time: string; tz: string };

const DEFAULT_RESPONSES: TimeResponse[] = [
  { city: "Beirut", time: "3:45 PM", tz: "UTC+3" },
  { city: "Tokyo", time: "10:45 PM", tz: "UTC+9" },
  { city: "New York", time: "8:45 AM", tz: "UTC-5" },
  { city: "London", time: "1:45 PM", tz: "UTC+0" },
  { city: "Sydney", time: "11:45 PM", tz: "UTC+10" },
  { city: "Paris", time: "2:45 PM", tz: "UTC+1" },
  { city: "Berlin", time: "2:45 PM", tz: "UTC+1" },
  { city: "Dubai", time: "5:45 PM", tz: "UTC+4" },
];

export function AnimatedTimeResponse({
  className,
  responses = DEFAULT_RESPONSES,
  index: externalIndex,
  autoplay = true,
  intervalMs = 2000,
  prettyJson = false,
}: {
  className?: string;
  responses?: TimeResponse[];
  index?: number;        // external index to sync with other components
  autoplay?: boolean;    // auto-cycle if no external index
  intervalMs?: number;
  prettyJson?: boolean;  // render as pretty JSON string
}) {
  const [internalIndex, setInternalIndex] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);
  const textRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (externalIndex !== undefined || !autoplay) return;
    const id = setInterval(() => {
      setInternalIndex((prev) => (prev + 1) % responses.length);
    }, intervalMs);
    return () => clearInterval(id);
  }, [responses.length, intervalMs, externalIndex, autoplay]);

  useEffect(() => {
    if (!textRef.current) return;
    gsap.fromTo(
      textRef.current,
      { opacity: 0, y: 8 },
      { opacity: 1, y: 0, duration: 0.35, ease: "power2.out" }
    );
  }, [externalIndex, internalIndex]);

  const idx = externalIndex !== undefined ? externalIndex % responses.length : internalIndex;
  const current = responses[idx];

  if (prettyJson) {
    const json = JSON.stringify({
      response: `The current time in ${current.city} is ${current.time} (${current.tz}).`,
      query_type: "time_service"
    }, null, 2);
    return (
      <pre className={`p-4 text-dark-900 dark:text-white font-mono text-sm leading-relaxed overflow-x-auto rounded-xl border border-white/20 dark:border-white/10 bg-white/5 dark:bg-dark-800/50 backdrop-blur-sm shadow-lg ${className ?? ""}`}>
        {json}
      </pre>
    );
  }

  return (
    <div
      ref={containerRef}
      className={`rounded-xl overflow-hidden border border-white/20 dark:border-white/10 bg-white/5 dark:bg-dark-800/50 backdrop-blur-sm shadow-lg ${className ?? ""}`}
    >
      <div className="p-4 flex items-start gap-3">
        <div className="mt-0.5 select-none">🕒</div>
        <div ref={textRef} className="text-sm text-dark-900 dark:text-white">
          <span>The current time in </span>
          <strong className="font-semibold">{current.city}</strong>
          <span> is </span>
          <strong className="font-semibold">{current.time}</strong>
          <span> (</span>
          <span className="text-white/80">{current.tz}</span>
          <span>).</span>
        </div>
      </div>
    </div>
  );
}


