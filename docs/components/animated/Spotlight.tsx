"use client";

import { useEffect, useRef } from "react";
import { gsap } from "gsap";
import { useTheme } from "@/lib/theme";

export function Spotlight() {
  const ref = useRef<HTMLDivElement | null>(null);
  const { theme } = useTheme();

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    gsap.fromTo(
      el,
      { opacity: 0, y: 20 },
      { opacity: 0.22, y: 0, duration: 1.2, ease: "power2.out", delay: 0.3 }
    );

    // Fixed top-center spotlight - soft light from above
    if (theme === 'dark') {
      el.style.background = `radial-gradient(800px at 50% -20%, rgba(255,255,255,0.15), rgba(144,224,239,0.08), rgba(0,0,0,0))`;
    } else {
      el.style.background = `radial-gradient(800px at 50% -20%, rgba(144,224,239,0.12), rgba(0,180,216,0.08), rgba(0,0,0,0))`;
    }
    return () => {
      // no-op
    };
  }, [theme]);

  return <div ref={ref} className="pointer-events-none absolute inset-0 -z-10" />;
}