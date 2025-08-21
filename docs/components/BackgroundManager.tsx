"use client";

import { useEffect, useRef, useState } from 'react';
import { useTheme } from '@/lib/theme';
import Iridescence from './Irid';
import LightRays from './LightRays';

interface BackgroundManagerProps {
  children?: React.ReactNode;
}

const BackgroundManager: React.FC<BackgroundManagerProps> = ({ children }) => {
  const { theme } = useTheme();
  const [isPreloaded, setIsPreloaded] = useState(true);
  const preloadRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Already preloaded by default to ensure immediate mount
  }, []);

  return (
    <div className="fixed inset-0 z-0">
      {/* Hidden preload containers removed; immediate backgrounds render */}
      
      {/* Active backgrounds with smooth transitions; rely on html/body dark paint to mask any initial white */}
      {isPreloaded && (
        <>
          {/* Iridescence Background - Light Mode */}
          <div className={`absolute inset-0 transition-opacity duration-500 ease-out ${
            theme === 'light' ? 'opacity-100' : 'opacity-0'
          }`}>
            <Iridescence
              color={[0.1, 0.3, 0.6]}
              mouseReact={true}
              amplitude={0.1}
              speed={1.0}
            />
          </div>
          
          {/* LightRays Background - Dark Mode */}
          <div className={`absolute inset-0 transition-opacity duration-500 ease-out ${
            theme === 'dark' ? 'opacity-100' : 'opacity-0'
          }`}>
            <LightRays
              raysOrigin="top-center"
              raysColor="#90E0EF"
              raysSpeed={0.2}
              lightSpread={1.2}
              rayLength={2.5}
              pulsating={true}
              fadeDistance={1}
              saturation={1.1}
              followMouse={true}
              mouseInfluence={0.15}
              noiseAmount={0.1}
              distortion={0.05}
            />
          </div>
        </>
      )}
      
      {children}
    </div>
  );
};

export default BackgroundManager;
