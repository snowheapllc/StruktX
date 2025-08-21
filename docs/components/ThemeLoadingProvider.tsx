"use client"

import React, { createContext, useContext, useEffect, useMemo, useState } from 'react'
import { usePathname } from 'next/navigation'

type Theme = 'light' | 'dark'

interface ThemeContextType {
  theme: Theme
  toggleTheme: () => void
}

type TransitionContextType = {
  fadeToDocs: () => void
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined)
export const TransitionContext = React.createContext<TransitionContextType>({
  fadeToDocs: () => {},
})

export function useTheme() {
  const context = useContext(ThemeContext)
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider')
  }
  return context
}

export default function ThemeLoadingProvider({ children }: { children: React.ReactNode }) {
  const [theme, setTheme] = useState<Theme>('light')
  const [overlayVisible, setOverlayVisible] = useState(false)
  const pathname = usePathname()

  useEffect(() => {
    const savedTheme = localStorage.getItem('theme') as Theme
    if (savedTheme) {
      setTheme(savedTheme)
    } else {
      setTheme('light')
    }
  }, [])

  useEffect(() => {
    const root = document.documentElement
    root.classList.remove('light', 'dark')
    root.classList.add(theme)
    localStorage.setItem('theme', theme)
  }, [theme])

  React.useEffect(() => {
    // Hide overlay explicitly on route settle
    const handleLoad = () => setOverlayVisible(false)
    window.addEventListener('load', handleLoad)
    return () => window.removeEventListener('load', handleLoad)
  }, [pathname])

  const toggleTheme = () => {
    // Apply an overlay during theme swap to prevent any background flicker
    setOverlayVisible(true)
    setTheme(prev => prev === 'dark' ? 'light' : 'dark')
    // Hide when first WebGL frame signals, fallback after 1200ms
    const off = () => { setOverlayVisible(false); window.removeEventListener('webgl-frame', off) }
    window.addEventListener('webgl-frame', off)
    setTimeout(off, 1200)
  }

  const fadeToDocs = React.useCallback(() => {
    try {
      if (typeof document !== 'undefined') {
        // Add a dark overlay to mask any repaint flashes during navigation
        const overlay = document.createElement('div')
        overlay.style.position = 'fixed'
        overlay.style.inset = '0'
        overlay.style.background = '#0b1220'
        overlay.style.zIndex = '2147483646'
        overlay.style.pointerEvents = 'none'
        overlay.style.opacity = '0'
        overlay.style.transition = 'opacity 200ms ease-out'
        document.body.appendChild(overlay)
        requestAnimationFrame(() => {
          overlay.style.opacity = '1'
        })
        // Remove overlay when WebGL is ready, or fallback
        const cleanup = () => {
          overlay.style.opacity = '0'
          setTimeout(() => {
            overlay.parentNode && overlay.parentNode.removeChild(overlay)
          }, 220)
          window.removeEventListener('webgl-frame', cleanup)
          window.removeEventListener('load', cleanup)
        }
        window.addEventListener('webgl-frame', cleanup)
        window.addEventListener('load', cleanup)
        setTimeout(cleanup, 1500)
      }
    } catch {}
  }, [])

  // Allow external consumers (like mobile nav) to request overlay
  useEffect(() => {
    const handler = () => fadeToDocs()
    window.addEventListener('docs-nav', handler)
    return () => window.removeEventListener('docs-nav', handler)
  }, [fadeToDocs])

  const themeCtxValue = useMemo(() => ({ theme, toggleTheme }), [theme])
  const transitionCtxValue = useMemo(() => ({ fadeToDocs }), [fadeToDocs])

  return (
    <ThemeContext.Provider value={themeCtxValue}>
      <TransitionContext.Provider value={transitionCtxValue}>
        <div className="relative">
          {/* Overlay during theme transitions */}
          {overlayVisible && (
            <div className="fixed inset-0" style={{ backgroundColor: '#0b1220', zIndex: 2147483646, pointerEvents: 'none', transition: 'opacity 200ms ease-out', opacity: 1 }} />
          )}
          {children}
        </div>
      </TransitionContext.Provider>
    </ThemeContext.Provider>
  )
}

// Global keyframes via tailwind already allowed, but provide a fallback
// Add this to globals if not present
// @keyframes fadeIn { from { opacity: 0 } to { opacity: 1 } }
