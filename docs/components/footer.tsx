"use client"

import Link from 'next/link'
import { Github, Twitter, Mail, Heart } from 'lucide-react'

export function Footer() {
  const currentYear = new Date().getFullYear()

  return (
    <footer className="bg-transparent text-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          <Link href="/" className="flex items-center space-x-3">
            <img src="/nobg-both-white.png" alt="StruktX" className="h-6 w-auto" />
          </Link>
          <div className="flex items-center gap-4 text-sm text-dark-300">
            <a href="https://struktx.mintlify.app" target="_blank" rel="noopener noreferrer" className="hover:text-white">Docs</a>
            <a href="https://github.com/aymanhs-code/StruktX" target="_blank" rel="noopener noreferrer" className="hover:text-white">GitHub</a>
            <span className="opacity-70">© {currentYear}</span>
          </div>
        </div>
        {/* Snowheap Attribution */}
        <div className="flex justify-center mt-4 pt-4 border-t border-white/10">
          <div className="flex items-center text-white/60 text-xs">
            <span>A project by</span>
            <img src="/snowheap.png" alt="Snowheap" className="h-4 w-auto mx-1" />
            <a
              href="https://snowheap.com"
              target="_blank"
              rel="noopener noreferrer"
              className="font-bold -ml-1.5 mt-0.5 hover:underline"
              style={{ fontFamily: 'DrukWideBold, serif' }}
            >
              snowheap.
            </a>
          </div>
        </div>
      </div>
    </footer>
  )
}