"use client";

import { ReactNode, useEffect, useRef } from "react";

type GridParallaxProps = {
  children: ReactNode;
  strength?: number; // max px offset at edges
  className?: string;
};

export function GridParallax({ children, strength = 20, className }: GridParallaxProps) {
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const container = ref.current;
    if (!container) return;

    const items = Array.from(container.querySelectorAll<HTMLElement>(".parallax-item"));

    const clamp = (v: number, min: number, max: number) => Math.max(min, Math.min(max, v));
    const onMove = (e: MouseEvent) => {
      const relX = (e.clientX / window.innerWidth) - 0.5; // -0.5..0.5
      const relY = (e.clientY / window.innerHeight) - 0.5;
      items.forEach((el, idx) => {
        const depth = Number(el.dataset.depth || (0.3 + (idx % 3) * 0.2)); // 0..1
        const tx = clamp(relX * strength * depth * 2, -strength, strength);
        const ty = clamp(relY * strength * depth * 2, -strength, strength);
        el.style.transform = `translate3d(${tx}px, ${ty}px, 0)`;
      });
    };

    window.addEventListener("mousemove", onMove);
    return () => window.removeEventListener("mousemove", onMove);
  }, [strength]);

  return (
    <div ref={ref} className={className}>
      {children}
    </div>
  );
}

