"use client";

import { useEffect, useRef, useState } from "react";
import { gsap } from "gsap";

const DEFAULT_COUNTRIES = [
	"Beirut",
	"Tokyo",
	"New York",
	"London",
	"Sydney",
	"Paris",
	"Berlin",
	"Dubai",
];

type AnimatedLocationProps = {
	className?: string;
	countries?: string[];
	index?: number;           // external index to sync with other components
	autoplay?: boolean;       // if true and no external index, auto-cycle
	intervalMs?: number;
	onSize?: (width: number, height: number) => void;
};

export function AnimatedLocation({ className, countries = DEFAULT_COUNTRIES, index, autoplay = true, intervalMs = 2000, onSize }: AnimatedLocationProps) {
	const [internalIndex, setInternalIndex] = useState(0);
	const textRef = useRef<HTMLSpanElement>(null);

	// Auto-cycle only if no external index provided
	useEffect(() => {
		if (index !== undefined || !autoplay) return;
		const interval = setInterval(() => {
			setInternalIndex(prev => (prev + 1) % countries.length);
		}, intervalMs);
		return () => clearInterval(interval);
	}, [countries.length, index, autoplay, intervalMs]);

	const currentIndex = index !== undefined ? index % countries.length : internalIndex;

	useEffect(() => {
		if (textRef.current) {
			gsap.fromTo(
				textRef.current,
				{ opacity: 0, y: 6 },
				{ opacity: 1, y: 0, duration: 0.35, ease: "power2.out", onComplete: () => {
					if (onSize && textRef.current) onSize(textRef.current.offsetWidth, textRef.current.offsetHeight);
				} }
			);
			if (onSize) onSize(textRef.current.offsetWidth, textRef.current.offsetHeight);
		}
	}, [currentIndex, onSize]);

	return (
		<span
			ref={textRef}
			className={`pointer-events-none select-none text-white font-extrabold bg-blue-600/90 px-2 py-0.5 rounded shadow-lg ring-1 ring-white/30 ${className ?? ""}`}
		>
			{countries[currentIndex]}
		</span>
	);
}


