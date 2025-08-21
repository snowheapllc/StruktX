"use client";

import { useEffect, useRef } from 'react';
import { Renderer, Program, Triangle, Mesh } from 'ogl';
import Iridescence from './Irid';
import LightRays from './LightRays';

interface WebGLPreloaderProps {
  onPreloadComplete?: () => void;
}

const WebGLPreloader: React.FC<WebGLPreloaderProps> = ({ onPreloadComplete }) => {
  const preloadRef = useRef<HTMLDivElement>(null);
  const isPreloadedRef = useRef(false);

  useEffect(() => {
    if (isPreloadedRef.current) return;

    const preloadWebGLBackgrounds = async () => {
      try {
        // Create hidden containers for preloading
        const lightContainer = document.createElement('div');
        const darkContainer = document.createElement('div');
        
        lightContainer.style.position = 'absolute';
        lightContainer.style.top = '-9999px';
        lightContainer.style.left = '-9999px';
        lightContainer.style.width = '100px';
        lightContainer.style.height = '100px';
        lightContainer.style.overflow = 'hidden';
        
        darkContainer.style.position = 'absolute';
        darkContainer.style.top = '-9999px';
        darkContainer.style.left = '-9999px';
        darkContainer.style.width = '100px';
        darkContainer.style.height = '100px';
        darkContainer.style.overflow = 'hidden';
        
        document.body.appendChild(lightContainer);
        document.body.appendChild(darkContainer);

        // Preload Iridescence (Light theme)
        const lightRenderer = new Renderer({
          canvas: document.createElement('canvas'),
          width: 100,
          height: 100,
          alpha: true,
          premultipliedAlpha: false,
          antialias: true,
          preserveDrawingBuffer: false,
          powerPreference: 'default',
        });

        // Preload LightRays (Dark theme)
        const darkRenderer = new Renderer({
          canvas: document.createElement('canvas'),
          width: 100,
          height: 100,
          alpha: true,
          premultipliedAlpha: false,
          antialias: true,
          preserveDrawingBuffer: false,
          powerPreference: 'default',
        });

        // Initialize basic shaders to warm up WebGL context
        const vertexShader = `
          attribute vec2 position;
          void main() {
            gl_Position = vec4(position, 0.0, 1.0);
          }
        `;

        const fragmentShader = `
          void main() {
            gl_FragColor = vec4(0.0, 0.0, 0.0, 0.0);
          }
        `;

        // Create and render basic programs to initialize WebGL contexts
        const lightProgram = new Program(lightRenderer.gl, {
          vertex: vertexShader,
          fragment: fragmentShader,
        });

        const darkProgram = new Program(darkRenderer.gl, {
          vertex: vertexShader,
          fragment: fragmentShader,
        });

        const geometry = new Triangle(lightRenderer.gl);
        
        const lightMesh = new Mesh(lightRenderer.gl, { geometry, program: lightProgram });
        const darkMesh = new Mesh(darkRenderer.gl, { geometry, program: darkProgram });

        // Render once to initialize contexts
        lightRenderer.render({ scene: lightMesh });
        darkRenderer.render({ scene: darkMesh });

        // Clean up
        lightRenderer.gl.getExtension('WEBGL_lose_context')?.loseContext();
        darkRenderer.gl.getExtension('WEBGL_lose_context')?.loseContext();
        document.body.removeChild(lightContainer);
        document.body.removeChild(darkContainer);

        // Mark as preloaded
        isPreloadedRef.current = true;
        
        // Store preloaded state in session storage
        sessionStorage.setItem('webgl-preloaded', 'true');
        
        onPreloadComplete?.();
        
      } catch (error) {
        console.warn('WebGL preloading failed:', error);
        // Still mark as attempted to prevent infinite retries
        isPreloadedRef.current = true;
        onPreloadComplete?.();
      }
    };

    // Check if already preloaded in this session
    const isAlreadyPreloaded = sessionStorage.getItem('webgl-preloaded') === 'true';
    
    if (isAlreadyPreloaded) {
      isPreloadedRef.current = true;
      onPreloadComplete?.();
    } else {
      // Small delay to ensure DOM is ready
      const timer = setTimeout(preloadWebGLBackgrounds, 100);
      return () => clearTimeout(timer);
    }
  }, [onPreloadComplete]);

  return (
    <div ref={preloadRef} style={{ display: 'none' }}>
      {/* Hidden preloader containers */}
    </div>
  );
};

export default WebGLPreloader;
