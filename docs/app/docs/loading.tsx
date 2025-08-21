"use client"

import React from 'react'

export default function Loading() {
  return (
    <div className="min-h-screen docs-gradient-bg bg-gradient-mesh bg-noise" style={{ backgroundColor: '#0b1220' }}>
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-24">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Sidebar skeleton */}
          <aside className="lg:col-span-3 order-last lg:order-first">
            <div className="space-y-3">
              {[...Array(5)].map((_, i) => (
                <div key={i} className="h-10 rounded-lg border border-white/10 dark:border-white/5 bg-white/10 dark:bg-dark-800/40 backdrop-blur-sm skeleton-pulse" />
              ))}
            </div>
          </aside>
          {/* Main content skeleton */}
          <main className="lg:col-span-9 space-y-6">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="h-24 rounded-xl border border-white/10 dark:border-white/5 bg-white/10 dark:bg-dark-800/40 backdrop-blur-sm skeleton-fade" />
            ))}
          </main>
        </div>
      </div>
    </div>
  )
}
