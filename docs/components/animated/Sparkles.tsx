"use client";

import { useEffect, useRef } from "react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

if (typeof window !== "undefined") {
  gsap.registerPlugin(ScrollTrigger);
}

export function Sparkles() {
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const container = ref.current;
    if (!container) return;

    const dots = container.querySelectorAll('.sparkle');
    gsap.set(dots, { opacity: 0 });
    gsap.to(dots, {
      opacity: 0.9,
      duration: 0.8,
      stagger: { each: 0.12, repeat: -1, yoyo: true },
      ease: 'sine.inOut',
    });
  }, []);

  return (
    <div ref={ref} className="pointer-events-none absolute inset-0 -z-10">
      {Array.from({ length: 24 }).map((_, i) => (
        <div
          key={i}
          className="sparkle absolute h-1 w-1 rounded-full"
          style={{
            left: `${(i * 37) % 100}%`,
            top: `${(i * 53) % 100}%`,
            backgroundColor: 'rgba(255,255,255,0.5)',
            boxShadow: '0 0 8px rgba(144,224,239,0.6)',
          }}
        />
      ))}
    </div>
  );
}