"use client";

import { useEffect, useMemo, useRef } from "react";
import { gsap } from "gsap";

type Props = {
  index?: number; // shared cycle index
  queryPart?: string;
  className?: string;
};

const CITIES = [
  "Beirut",
  "Tokyo",
  "New York",
  "London",
  "Sydney",
  "Paris",
  "Berlin",
  "Dubai",
] as const;

const CITY_TO_TZ: Record<string, string> = {
  Beirut: "Asia/Beirut",
  Tokyo: "Asia/Tokyo",
  "New York": "America/New_York",
  London: "Europe/London",
  Sydney: "Australia/Sydney",
  Paris: "Europe/Paris",
  Berlin: "Europe/Berlin",
  Dubai: "Asia/Dubai",
};

function formatLocalTime(timeZone: string) {
  const now = new Date();
  const date = new Intl.DateTimeFormat("en-GB", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
    timeZone,
    timeZoneName: "short",
  }).format(now);
  return `${date} (${timeZone})`;
}

export function TimeServicePreview({ index = 0, queryPart = "what is the time there?", className }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const bodyRef = useRef<HTMLDivElement>(null);

  const city = CITIES[((index % CITIES.length) + CITIES.length) % CITIES.length];
  const timeZone = CITY_TO_TZ[city];
  const localTime = useMemo(() => formatLocalTime(timeZone), [timeZone]);

  const injectedDocs = useMemo(
    () => [
      `location:city=${city}`,
    ],
    [city]
  );

  useEffect(() => {
    if (!bodyRef.current) return;
    gsap.fromTo(bodyRef.current, { opacity: 0, y: 6 }, { opacity: 1, y: 0, duration: 0.35, ease: "power2.out" });
  }, [city]);

  return (
    <div
      ref={containerRef}
      className={`p-4 text-dark-900 dark:text-white font-mono text-sm leading-relaxed overflow-x-auto rounded-xl border border-white/20 dark:border-white/10 bg-white/5 dark:bg-dark-800/50 backdrop-blur-sm shadow-lg  ${className ?? ""}`}
    >
      <div className="px-3 py-1.5 border-b border-white-500/40 flex items-center gap-2">
        <span className="text-white">Time Service Result</span>
      </div>
      <div ref={bodyRef} className="p-3 leading-6">
        <div>
          <span className="text-blue-300">Query Part:</span>
          <span className="ml-2 text-white">{queryPart}</span>
        </div>
        <div>
          <span className="text-blue-300">Timezone:</span>
          <span className="ml-2 text-white">{timeZone}</span>
        </div>
        <div>
          <span className="text-blue-300">Local Time:</span>
          <span className="ml-2 text-white">{localTime}</span>
        </div>
        <div>
          <span className="text-blue-300">MEM_PROBE:</span>
          <span className="ml-2 text-white">{city}</span>
        </div>
        <div className="mt-1">
          <span className="text-blue-300">Injected Docs:</span>
          <span className="ml-2 text-white">{injectedDocs.length}</span>
          <div className="mt-1">
            {injectedDocs.map((d, i) => (
              <div key={i} className="text-blue-300">
                - {d}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}


