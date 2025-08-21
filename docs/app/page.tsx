"use client"

import { Button } from '@/components/ui/button'
import { ArrowRight, Github, BookOpen } from 'lucide-react'
import { motion } from 'framer-motion'
import { useEffect, useRef, useState, Suspense, useContext } from 'react'
import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import { Sparkles } from '@/components/animated/Sparkles'
import { CodeShowcase } from '@/components/sections/CodeShowcase'
import GlassSurface from '@/components/GlassSurface'
import SplitText from '@/components/SplitText'
import MagicBento from '@/components/MagicBento'
import { ThemeToggle } from '@/components/ui/theme-toggle'
import LoadingSkeleton from '@/components/LoadingSkeleton'
import { TransitionContext } from '@/components/ThemeLoadingProvider'
// Background handled globally in RootLayout; theme not needed here
import { Footer } from '@/components/footer'


 

export default function Home() {
  const rootRef = useRef<HTMLDivElement | null>(null)
  const [isPageLoading, setIsPageLoading] = useState(true)
  const [isFullyLoaded, setIsFullyLoaded] = useState(false)
  const { fadeToDocs } = useContext(TransitionContext)

  useEffect(() => {
    if (typeof window === 'undefined') return
    
    gsap.registerPlugin(ScrollTrigger)
    const ctx = gsap.context(() => {
      const q = (sel: string) => document.querySelector(sel)

      // Only run animations if page is fully loaded
      if (!isFullyLoaded) return

      if (q('#showcase') && q('.feature-card')) {
        gsap.from('.feature-card', {
          scrollTrigger: { trigger: '#showcase', start: 'top 80%' },
          y: 40,
          opacity: 0,
          duration: 0.7,
          ease: 'power2.out',
          stagger: 0.12,
        })
      }

      if (q('.code-showcase')) {
        gsap.from('.code-showcase', {
          scrollTrigger: { trigger: '.code-showcase', start: 'top 85%' },
          y: 30,
          opacity: 0,
          duration: 0.8,
          ease: 'power2.out',
        })
      }

      if (q('.cta-card')) {
        gsap.from('.cta-card', {
          scrollTrigger: { trigger: '.cta-card', start: 'top 85%' },
          y: 20,
          opacity: 0,
          duration: 0.7,
          ease: 'power2.out',
        })
      }
    }, rootRef)

    return () => {
      ctx.revert()
    }
  }, [isFullyLoaded])

  // Show loading skeleton while page is loading
  if (isPageLoading) {
    return (
      <>
        <LoadingSkeleton onComplete={() => {
          setIsPageLoading(false)
          // Add a small delay to ensure smooth transition
          setTimeout(() => setIsFullyLoaded(true), 100)
        }} />
      </>
    )
  }

  return (
    <motion.div 
      ref={rootRef} 
      className="relative transition-colors duration-300"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
    >
      {/* Background is now handled globally via BackgroundManager in RootLayout */}
      
      <Suspense fallback={null}>
        <Sparkles />
      </Suspense>
      
             {/* Centered Hero Section - Full Viewport Height */}
       <section className="h-screen flex flex-col justify-center items-center relative z-20">
        {/* Glass Navigation */}
        <div className="absolute top-8 left-1/2 transform -translate-x-1/2 w-full max-w-7xl px-4 sm:px-6 lg:px-8">
          <Suspense fallback={
            <div className="w-full h-15 bg-white/10 backdrop-blur-sm rounded-full border border-white/20 dark:border-white/10 animate-pulse shadow-lg"></div>
          }>
            <GlassSurface
              width="100%"
              height={60}
              borderRadius={50}
              backgroundOpacity={0.1}
              saturation={1}
              blur={11}
              brightness={50}
              opacity={0.93}
            >
              <div className="flex items-center justify-between pl-8 pr-2 w-full">
                <Suspense fallback={
                  <div className="text-xl font-bold tracking-tight text-white">StruktX</div>
                }>
                  <SplitText
                    text="StruktX"
                    className="text-xl font-bold tracking-tight text-white"
                    delay={200}
                    duration={2}
                    ease="elastic.out(1, 0.3)"
                    splitType="words"
                    from={{ opacity: 0, y: 5 }}
                    to={{ opacity: 1, y: 0 }}
                    threshold={0.6}
                    rootMargin="-10px"
                    textAlign="center"
                    onLetterAnimationComplete={() => {}}
                  />
                </Suspense>
                <div className="flex items-center space-x-4">
                  <ThemeToggle />
                </div>
              </div>
            </GlassSurface>
          </Suspense>
        </div>

        {/* Hero Content - Centered */}
        <div className="text-center max-w-4xl mx-auto px-4">
          <div className="mb-10">
            <div className="flex justify-center mb-6">
              {/* <div className="hero-badge inline-flex items-center rounded-full bg-white/10 dark:bg-white/5 px-4 py-2 text-xs font-medium text-dark-600 dark:text-dark-300 ring-1 ring-white/10">
                Introducing
              </div> */}
            </div>
            <div className="mt-6 flex justify-center">
              <img src="/logo-white.png" alt="StruktX" className="h-24 sm:h-32 lg:h-40 w-auto" />
            </div>
            {/* <p className="hero-sub mt-6 text-lg leading-8 text-dark-300 max-w-2xl mx-auto">
              Built for engineers, made to ship.
            <br />
              A lean core with swappable pieces.
            </p> */}
            <Suspense fallback={
              <h1 className="mt-8 text-4xl sm:text-6xl font-extrabold tracking-tight text-dark-900 dark:text-white">
                Natural Language → Action
              </h1>
            }>
              <SplitText
                text="Natural Language → Action"
                className="mt-8 pb-8 text-4xl sm:text-6xl font-extrabold tracking-tight text-dark-900 dark:text-white"
                delay={180}
                duration={1.6}
                ease="elastic.out(1, 0.3)"
                splitType="words"
                from={{ opacity: 0, y: 10 }}
                to={{ opacity: 1, y: 0 }}
                threshold={0.6}
                rootMargin="-10px"
                textAlign="center"
                onLetterAnimationComplete={() => {}}
                animateOnMount={true}
              />
            </Suspense>
            <Suspense fallback={
              <p className="flex items-center justify-center hero-sub mt-6 text-lg leading-8 text-dark-300 max-w-2xl mx-auto">
                A lean core with swappable pieces.
              </p>
            }>
              <SplitText
                text="A lean core with swappable pieces."
                className="flex items-center justify-center hero-sub mt-6 text-lg leading-8 text-dark-300 max-w-2xl mx-auto"
                delay={330}
                duration={2}
                ease="elastic.out(1, 0.3)"
                splitType="words"
                from={{ opacity: 0, y: 5 }}
                to={{ opacity: 1, y: 0 }}
                threshold={0.6}
                rootMargin="-10px"
                textAlign="center"
                onLetterAnimationComplete={() => {}}
                animateOnMount={true}
              />
            </Suspense>
          </div>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mt-10">
            <Suspense fallback={
              <div className="w-45 h-12 bg-white/10 backdrop-blur-sm rounded-full border border-white/20 dark:border-white/10 animate-pulse shadow-lg"></div>
            }>
              <GlassSurface width={180} height={50} borderRadius={25} backgroundOpacity={0.15} saturation={1.1} blur={8} brightness={70} opacity={0.9}>
                <a href="/docs" className="h-full w-full flex items-center justify-center text-white font-medium" onMouseDown={() => { try { document.body.style.backgroundColor = '#0b1220' } catch {} }} onClick={(e) => { try { fadeToDocs() } catch {} }}>
                  Get Started
                  <ArrowRight className="ml-2 h-5 w-5" />
                </a>
              </GlassSurface>
            </Suspense>
            <Suspense fallback={
              <div className="w-45 h-12 bg-white/10 backdrop-blur-sm rounded-full border border-white/20 dark:border-white/10 animate-pulse shadow-lg"></div>
            }>
              <GlassSurface width={180} height={50} borderRadius={25} backgroundOpacity={0.08} saturation={1} blur={8} brightness={60} opacity={0.8}>
                <a href="https://github.com/aymanhs-code/StruktX" target="_blank" rel="noopener noreferrer" className="h-full w-full flex items-center justify-center text-white/90 font-medium">
                  <Github className="mr-2 h-5 w-5" />
                  View on GitHub
                </a>
              </GlassSurface>
            </Suspense>
          </div>
        </div>



      </section>

      {/* Content Below Hero - Appears on Scroll */}
      <div className="relative z-20 bg-transparent">
        {/* Magic Bento Showcase */}
        <section id="showcase" className="pt-12 pb-28 bg-transparent">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
            <div className="text-center pt-20 mb-16">
              <h2 className="text-3xl font-bold tracking-tight text-dark-900 dark:text-white sm:text-4xl mb-4">
                Power of Structure
              </h2>
              <p className="text-lg text-dark-600 dark:text-dark-300 max-w-2xl mx-auto">
                Discover the magic of StruktX
              </p>
            </div>
            
            <Suspense fallback={
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                {[...Array(6)].map((_, i) => (
                  <div 
                    key={i} 
                    className="h-48 bg-white/10 dark:bg-white/5 backdrop-blur-sm rounded-2xl animate-pulse border border-white/20 dark:border-white/10 shadow-lg"
                    style={{
                      animationDelay: `${i * 0.1}s`
                    }}
                  >
                    <div className="p-6 h-full flex flex-col">
                      <div className="w-20 h-4 bg-white/20 dark:bg-white/10 rounded mb-4"></div>
                      <div className="w-full h-6 bg-white/20 dark:bg-white/10 rounded mb-3"></div>
                      <div className="w-3/4 h-4 bg-white/15 dark:bg-white/8 rounded mb-2"></div>
                      <div className="w-1/2 h-4 bg-white/15 dark:bg-white/8 rounded"></div>
                    </div>
                  </div>
                ))}
              </div>
            }>
              <MagicBento
                textAutoHide={false}
                enableStars={true}
                enableSpotlight={true}
                enableBorderGlow={true}
                disableAnimations={false}
                spotlightRadius={400}
                particleCount={15}
                enableTilt={true}
                glowColor="255, 255, 255"
                clickEffect={true}
                enableMagnetism={true}
              />
            </Suspense>
          </div>
        </section>

        {/* Code Showcase */}
        <Suspense fallback={
          <div className="py-24">
            <div className="text-center mb-16">
              <div className="w-48 h-8 bg-white/20 dark:bg-white/10 backdrop-blur-sm rounded mx-auto mb-4 animate-pulse border border-white/20 dark:border-white/10"></div>
              <div className="w-80 h-6 bg-white/15 dark:bg-white/8 backdrop-blur-sm rounded mx-auto animate-pulse border border-white/20 dark:border-white/10"></div>
            </div>
            <div className="bg-white/10 dark:bg-white/5 backdrop-blur-sm rounded-2xl p-8 animate-pulse border border-white/20 dark:border-white/10 shadow-xl">
              <div className="flex items-center justify-between mb-6">
                <div className="w-32 h-4 bg-white/20 dark:bg-white/10 rounded"></div>
                <div className="w-16 h-4 bg-white/20 dark:bg-white/10 rounded"></div>
              </div>
              <div className="space-y-4">
                {[...Array(8)].map((_, i) => (
                  <div 
                    key={i} 
                    className="h-4 bg-white/15 dark:bg-white/8 rounded"
                    style={{ 
                      width: `${Math.random() * 30 + 70}%`,
                      animationDelay: `${i * 0.05}s`
                    }}
                  ></div>
                ))}
              </div>
            </div>
          </div>
        }>
          <CodeShowcase className="py-24 code-showcase" />
        </Suspense>

        {/* CTA Section */}
        <section className="py-24">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 text-center">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              viewport={{ once: true }}
            >
              <div className="mx-auto rounded-3xl border border-white/10 dark:border-white/5 bg-white/5 dark:bg-white/5 backdrop-blur-xl shadow-xl p-12">
                <div className="text-center">
                  <h2 className="text-3xl font-bold tracking-tight text-dark-900 dark:text-white sm:text-4xl mb-3">Ready to Build?</h2>
                  <p className="text-dark-600 dark:text-dark-300 mb-8">Start with the docs or explore the codebase.</p>
                  <div className="flex flex-col sm:flex-row gap-4 justify-center">
                    <a href="/docs">
                      <Button
                        asChild
                        size="lg"
                        className="text-base px-8 py-4 bg-white/10 dark:bg-white/5 hover:bg-white/20 dark:hover:bg-white/10 border-white/10 dark:border-white/5 text-dark-900 dark:text-white"
                      >
                        <>
                          <BookOpen className="mr-2 h-5 w-5" />
                          Read Documentation
                        </>
                      </Button>
                    </a>
                    <a
                      href="https://github.com/aymanHS-code/StruktX"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      <Button
                        asChild
                        variant="outline"
                        size="lg"
                        className="text-base px-8 py-4 bg-white/5 dark:bg-white/5 hover:bg-white/10 dark:hover:bg-white/10 border-white/10 dark:border-white/5 text-dark-900 dark:text-white"
                      >
                        <>
                          <Github className="mr-2 h-5 w-5" />
                          Star on GitHub
                        </>
                      </Button>
                    </a>
                  </div>
                </div>
              </div>
            </motion.div>
          </div>
        </section>
      </div>
      
      {/* Footer */}
      <Footer />
    </motion.div>
  )
} 