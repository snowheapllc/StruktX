"use client";

import { ReactNode, useEffect, useRef } from "react";

type ParallaxProps = {
  strength?: number; // px offset at edges
  className?: string;
  children: ReactNode;
};

export function Parallax({ strength = 20, className, children }: ParallaxProps) {
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const onMove = (e: MouseEvent) => {
      const { innerWidth, innerHeight } = window;
      const dx = (e.clientX / innerWidth - 0.5) * 2; // -1..1
      const dy = (e.clientY / innerHeight - 0.5) * 2;
      el.style.transform = `translate3d(${dx * strength}px, ${dy * strength}px, 0)`;
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