'use client'

import React, { useState } from 'react'
import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { Menu, X } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { ThemeToggle } from '@/components/ui/theme-toggle'
import { TransitionContext } from './ThemeLoadingProvider'

export function Navigation() {
  const [isOpen, setIsOpen] = useState(false)
  const { fadeToDocs } = React.useContext(TransitionContext)


  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-white/60 dark:bg-dark-900/40 backdrop-blur-xl border-b border-white/20 dark:border-white/10">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <Link href="/" className="flex items-center space-x-3">
            <img src="/logo.svg" alt="StruktX" className="h-8 w-auto brightness-110 drop-shadow-[0_0_20px_rgba(144,224,239,0.25)]" />
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-8">
            <Link
              href="/docs"
              onMouseDown={() => {
                // Darken body immediately to avoid transient white
                try { document.body.style.backgroundColor = '#0b1220' } catch {}
              }}
              onClick={() => {
                fadeToDocs()
              }}
              prefetch
            >
              Docs
            </Link>
          </div>

          {/* Desktop Actions */}
          <div className="hidden md:flex items-center space-x-4">
            <ThemeToggle />
          </div>

          {/* Mobile menu button */}
          <div className="md:hidden">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsOpen(!isOpen)}
              className="w-9 h-9 p-0"
            >
              {isOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
            </Button>
          </div>
        </div>
      </div>

      {/* Mobile Navigation */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="md:hidden bg-white dark:bg-dark-900 border-b border-dark-200 dark:border-dark-700"
          >
            <div className="px-4 py-4 space-y-4">
              <Link href="/docs" className="block text-base" onClick={() => { setIsOpen(false); try { const ev = new Event('docs-nav'); window.dispatchEvent(ev) } catch {} }}>Docs</Link>
              <div className="pt-4 border-t border-dark-200 dark:border-dark-700">
                <div className="flex items-center justify-between">
                <ThemeToggle />
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </nav>
  )
} 