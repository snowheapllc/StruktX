"use client";

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';

interface LoadingSkeletonProps {
  onComplete?: () => void;
}

const LoadingSkeleton: React.FC<LoadingSkeletonProps> = ({ onComplete }) => {
  const [isFading, setIsFading] = useState(false);

  // Ensure loading completes after minimum time
  useEffect(() => {
    const timer = setTimeout(() => {
      setIsFading(true);
      setTimeout(() => {
        onComplete?.();
      }, 500); // Fade duration (reduced)
    }, 900); // Shorter minimum loading time
    
    return () => clearTimeout(timer);
  }, [onComplete]);

  return (
    <motion.div 
      className="min-h-screen flex flex-col justify-center items-center relative"
      style={{ backgroundColor: '#0b1220' }}
      animate={{ opacity: isFading ? 0 : 1 }}
      transition={{ duration: 0.8, ease: "easeInOut" }}
    >
      {/* Dark background gradient */}
      <div className="fixed inset-0 bg-gradient-to-br from-dark-800/50 via-dark-900/30 to-dark-800/50 animate-pulse" style={{ backgroundColor: '#0b1220' }}></div>
      
      {/* Floating particles */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        {[...Array(12)].map((_, i) => (
          <div
            key={i}
            className="absolute w-1 h-1 bg-blue-400/30 dark:bg-blue-300/20 rounded-full animate-pulse"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              animationDelay: `${Math.random() * 2}s`,
              animationDuration: `${3 + Math.random() * 2}s`
            }}
          />
        ))}
      </div>

      {/* Main content skeleton */}
      <div className="relative z-10 text-center">
        {/* Centered Square Logo Skeleton */}
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 1.2, ease: "easeOut" }}
          className="mb-8"
        >
          <div className="w-32 h-32 bg-white/10 dark:bg-white/5 backdrop-blur-sm rounded-2xl skeleton-pulse border border-white/20 dark:border-white/10 shadow-lg mx-auto">
            {/* Inner square structure to match the logo */}
            <div className="w-full h-full flex items-center justify-center relative">
              {/* Outer square */}
              <div className="w-20 h-20 border-2 border-white/30 dark:border-white/20 rounded-lg relative">
                {/* Inner square */}
                <div className="absolute inset-2 border border-white/40 dark:border-white/30 rounded"></div>
                {/* Spiral-like path connecting corners */}
                <div className="absolute inset-0">
                  <svg className="w-full h-full" viewBox="0 0 80 80">
                    <path
                      d="M 20 20 L 60 20 L 60 60 L 20 60 L 20 20"
                      stroke="rgba(255,255,255,0.3)"
                      strokeWidth="1"
                      fill="none"
                    />
                    <path
                      d="M 30 30 L 50 30 L 50 50 L 30 50 L 30 30"
                      stroke="rgba(255,255,255,0.4)"
                      strokeWidth="1"
                      fill="none"
                    />
                  </svg>
                </div>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Loading Text */}
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.5 }}
          className="text-white/60 text-sm font-medium mb-4"
        ></motion.p>
        
        {/* Simple loading animation */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.8 }}
          className="w-4 h-4 border-2 border-white/30 dark:border-white/20 border-t-white/60 dark:border-t-white/40 rounded-full animate-spin mx-auto"
        />
      </div>

             {/* Subtle gradient overlay */}
       <div className="fixed inset-0 bg-gradient-to-t from-transparent via-transparent to-black/5 pointer-events-none"></div>
     </motion.div>
   );
 };

export default LoadingSkeleton;
