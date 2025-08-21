"use client";

import { useEffect, useRef } from "react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

if (typeof window !== "undefined") {
  gsap.registerPlugin(ScrollTrigger);
}

export function Background() {
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const dots = container.querySelectorAll(".bg-dot");

    dots.forEach((dot, i) => {
      const el = dot as HTMLElement;
      const x = (Math.random() - 0.5) * 100;
      const y = (Math.random() - 0.5) * 100;
      const delay = Math.random() * 1.5;

      gsap.to(el, {
        x,
        y,
        opacity: 0.8,
        duration: 2.5,
        ease: "sine.inOut",
        repeat: -1,
        yoyo: true,
        delay,
      });
    });

    // Remove background translation to avoid moving out of bounds
    return () => {
      // no-op cleanup
    };
  }, []);

  return (
    <div
      ref={containerRef}
      className="pointer-events-none absolute inset-0 -z-10 overflow-hidden"
      aria-hidden
    >
      <div className="absolute inset-0 bg-gradient-mesh" />
      <div className="absolute inset-0 bg-noise" />
      {/* floating dots */}
      {Array.from({ length: 30 }).map((_, i) => (
        <div
          key={i}
          className="bg-dot absolute h-2 w-2 rounded-full"
          style={{
            left: `${Math.random() * 100}%`,
            top: `${Math.random() * 100}%`,
            backgroundColor: i % 3 === 0 ? "#90E0EF" : i % 3 === 1 ? "#00B4D8" : "#0077B6",
            opacity: 0.3,
          }}
        />
      ))}
    </div>
  );
}