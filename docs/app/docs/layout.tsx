"use client"

import React, { PropsWithChildren, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import { ArrowLeft } from 'lucide-react'
import Link from 'next/link'
import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'

export default function DocsLayout({ children }: Readonly<PropsWithChildren>) {
  const layoutRef = useRef<HTMLDivElement>(null)
  const [gate, setGate] = React.useState(true)

  useEffect(() => {
    if (typeof window === 'undefined') return
    
    gsap.registerPlugin(ScrollTrigger)
    const ctx = gsap.context(() => {
      // Animate sections on scroll
      gsap.utils.toArray('.section').forEach((section: any) => {
        gsap.fromTo(section, 
          { 
            opacity: 0, 
            y: 30,
            scale: 0.98
          },
          {
            opacity: 1,
            y: 0,
            scale: 1,
            duration: 0.8,
            ease: "power2.out",
            scrollTrigger: {
              trigger: section,
              start: "top 85%",
              end: "bottom 15%",
              toggleActions: "play none none reverse"
            }
          }
        )
      })

      // Animate sidebar items
      gsap.fromTo('.sidebar-item',
        { opacity: 0, x: -20 },
        {
          opacity: 1,
          x: 0,
          duration: 0.6,
          stagger: 0.1,
          ease: "power2.out",
          delay: 0.3
        }
      )
    }, layoutRef)

    return () => ctx.revert()
  }, [])

  useEffect(() => {
    const t = setTimeout(() => setGate(false), 1500)
    return () => clearTimeout(t)
  }, [])

  return (
    <div ref={layoutRef} className="relative min-h-screen docs-gradient-bg bg-noise" style={{ backgroundColor: '#0b1220' }}>
      {gate && (
        <div className="fixed inset-0 z-40 flex items-center justify-center bg-[#0b1220] docs-gradient-bg bg-noise">
          <div className="w-10 h-10 border-2 border-white/30 border-t-white/70 rounded-full animate-spin" />
        </div>
      )}
      {/* Back Button */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.2 }}
        className="fixed top-6 left-6 z-50"
      >
        <Link href="/" className="group">
          <div className="flex items-center space-x-2 px-4 py-2 rounded-full bg-dark-800/40 backdrop-blur-md border border-white/20 dark:border-white/10 hover:bg-dark-700/60 transition-all duration-300 shadow-lg hover:shadow-xl">
            <ArrowLeft className="h-4 w-4 text-white group-hover:-translate-x-1 transition-transform duration-300" />
            <span className="text-sm font-medium text-white">Back</span>
          </div>
        </Link>
      </motion.div>

      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-24">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          <aside className="lg:col-span-3 order-last lg:order-first">
            <nav className="sticky top-24 space-y-2 text-sm">
              <div className="sidebar-item px-3 py-2 rounded-lg border border-white/10 dark:border-white/5 bg-dark-800/40 backdrop-blur-md hover:bg-dark-700/50 transition-all duration-300">
                <p className="text-xs uppercase tracking-wide text-white/60 mb-2">Getting Started</p>
                <ul className="space-y-1">
                  <li><a className="hover:underline transition-colors duration-200 text-white/80" href="#introduction">Introduction</a></li>
                  <li><a className="hover:underline transition-colors duration-200 text-white/80" href="#quickstart">Quickstart</a></li>
                  <li><a className="hover:underline transition-colors duration-200 text-white/80" href="#architecture">Architecture</a></li>
                  <li><a className="hover:underline transition-colors duration-200 text-white/80" href="#configuration">Configuration</a></li>
                </ul>
              </div>
              <div className="sidebar-item px-3 py-2 rounded-lg border border-white/10 dark:border-white/5 bg-dark-800/40 backdrop-blur-md hover:bg-dark-700/50 transition-all duration-300">
                <p className="text-xs uppercase tracking-wide text-white/60 mb-2">Core Components</p>
                <ul className="space-y-1">
                  <li><a className="hover:underline transition-colors duration-200 text-white/80" href="#providers">Providers</a></li>
                  <li><a className="hover:underline transition-colors duration-200 text-white/80" href="#llm">LLM Clients</a></li>
                  <li><a className="hover:underline transition-colors duration-200 text-white/80" href="#classifier">Classifier</a></li>
                  <li><a className="hover:underline transition-colors duration-200 text-white/80" href="#handlers">Handlers</a></li>
                  <li><a className="hover:underline transition-colors duration-200 text-white/80" href="#middleware">Middleware</a></li>
                  <li><a className="hover:underline transition-colors duration-200 text-white/80" href="#memory">Memory</a></li>
                </ul>
              </div>
              <div className="sidebar-item px-3 py-2 rounded-lg border border-white/10 dark:border-white/5 bg-dark-800/40 backdrop-blur-md hover:bg-dark-700/50 transition-all duration-300">
                <p className="text-xs uppercase tracking-wide text-white/60 mb-2">Ecosystem</p>
                <ul className="space-y-1">
                  <li><a className="hover:underline transition-colors duration-200 text-white/80" href="#langchain">LangChain</a></li>
                  <li><a className="hover:underline transition-colors duration-200 text-white/80" href="#logging">Logging</a></li>
                  <li><a className="hover:underline transition-colors duration-200 text-white/80" href="#memory-extraction">Memory Extraction</a></li>
                  <li><a className="hover:underline transition-colors duration-200 text-white/80" href="#extensions">Extensions</a></li>
                  <li><a className="hover:underline transition-colors duration-200 text-white/80" href="#devices-extension">Devices Extension</a></li>
                </ul>
              </div>
              <div className="sidebar-item px-3 py-2 rounded-lg border border-white/10 dark:border-white/5 bg-dark-800/40 backdrop-blur-md hover:bg-dark-700/50 transition-all duration-300">
                <p className="text-xs uppercase tracking-wide text-white/60 mb-2">Extras</p>
                <ul className="space-y-1">
                  <li><a className="hover:underline transition-colors duration-200 text-white/80" href="#extras">MCP Server</a></li>
                </ul>
              </div>
              <div className="sidebar-item px-3 py-2 rounded-lg border border-white/10 dark:border-white/5 bg-dark-800/40 backdrop-blur-md hover:bg-dark-700/50 transition-all duration-300">
                <p className="text-xs uppercase tracking-wide text-white/60 mb-2">Advanced</p>
                <ul className="space-y-1">
                  <li><a className="hover:underline transition-colors duration-200 text-white/80" href="#query-hints">Query Hints</a></li>
                  <li><a className="hover:underline transition-colors duration-200 text-white/80" href="#augment-source">augment_source</a></li>
                  <li><a className="hover:underline transition-colors duration-200 text-white/80" href="#context">Context & Scoping</a></li>
                  <li><a className="hover:underline transition-colors duration-200 text-white/80" href="#step-by-step">Step-by-Step Guide</a></li>
                  <li><a className="hover:underline transition-colors duration-200 text-white/80" href="#api-overview">API Reference</a></li>
                  <li><a className="hover:underline transition-colors duration-200 text-white/80" href="#best-practices">Best Practices</a></li>
                  <li><a className="hover:underline transition-colors duration-200 text-white/80" href="#faq">FAQ</a></li>
                </ul>
              </div>
            </nav>
          </aside>
          <main className="lg:col-span-9 space-y-24 md:space-y-28">
            {children}
          </main>
        </div>
      </div>
    </div>
  )
}