import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import ThemeLoadingProvider from '@/components/ThemeLoadingProvider'
import BackgroundManager from '@/components/BackgroundManager'
import { ThemeProvider } from '@/lib/theme'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  metadataBase: new URL('https://struktx.vercel.app'),
  title: {
    default: 'StruktX - A Lean Core with Swappable Pieces',
    template: '%s | StruktX'
  },
  description: 'A lean core with swappable pieces. StruktX is a configurable, typed AI framework with swappable LLM, classifier, handlers, and optional memory. Built for engineers, made to ship.',
  keywords: [
    'AI framework',
    'machine learning',
    'LLM',
    'Python',
    'type safety',
    'configurable',
    'swappable components',
    'developer tools',
    'AI development',
    'structured AI',
    'lean architecture'
  ],
  authors: [{ name: 'StruktX Team' }],
  creator: 'StruktX',
  publisher: 'StruktX',
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
  icons: {
    icon: [
      { url: '/favicon.ico', sizes: '48x48', type: 'image/x-icon' },
      { url: '/favicon.png', sizes: '48x48', type: 'image/png' }
    ],
    apple: [
      { url: '/logo-blue.png', sizes: '180x180', type: 'image/png' }
    ],
    shortcut: '/favicon.ico'
  },
  manifest: '/site.webmanifest',
            openGraph: {
        type: 'website',
        locale: 'en_US',
        url: 'https://struktx.vercel.app',
        siteName: 'StruktX',
        title: 'StruktX - A Lean Core with Swappable Pieces',
        description: 'A lean core with swappable pieces. StruktX is a configurable, typed AI framework with swappable LLM, classifier, handlers, and optional memory. Built for engineers, made to ship.',
        images: [
          {
            url: '/opengraph-image',
            width: 1200,
            height: 630,
            alt: 'StruktX - A Lean Core with Swappable Pieces',
            type: 'image/png',
          },
          {
            url: '/logo-blue.png',
            width: 800,
            height: 800,
            alt: 'StruktX Logo',
            type: 'image/png',
          }
        ],
      },
            twitter: {
        card: 'summary_large_image',
        site: '@struktx',
        creator: '@struktx',
        title: 'StruktX - A Lean Core with Swappable Pieces',
        description: 'A lean core with swappable pieces. StruktX is a configurable, typed AI framework with swappable LLM, classifier, handlers, and optional memory.',
        images: [
          {
            url: '/twitter-image',
            width: 1200,
            height: 630,
            alt: 'StruktX - A Lean Core with Swappable Pieces',
          }
        ],
      },
  alternates: {
    canonical: 'https://struktx.vercel.app',
  },
  category: 'technology',
  classification: 'AI Framework',
  other: {
    'theme-color': '#0066cc',
    'msapplication-TileColor': '#0066cc',
    'apple-mobile-web-app-capable': 'yes',
    'apple-mobile-web-app-status-bar-style': 'default',
    'apple-mobile-web-app-title': 'StruktX',
    'application-name': 'StruktX',
    'msapplication-TileImage': '/logo-blue.png',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
            <html lang="en" className="scroll-smooth dark" style={{ backgroundColor: '#0b1220' }}>
      <head>
        {/* PRELOAD: Execute before anything else */}
        <script dangerouslySetInnerHTML={{ __html: `
          (function(){
            try {
              // Force dark background immediately
              document.documentElement.style.backgroundColor = '#0b1220';
              document.body && (document.body.style.backgroundColor = '#0b1220');
              
              // Also set on window for immediate effect
              if (typeof window !== 'undefined') {
                window.addEventListener('beforeunload', function() {
                  document.documentElement.style.backgroundColor = '#0b1220';
                  document.body && (document.body.style.backgroundColor = '#0b1220');
                });
              }
            } catch(e) {}
          })();
        ` }} />
        
        {/* CRITICAL: Force dark background with maximum priority */}
        <style dangerouslySetInnerHTML={{ __html: `
          html { background-color: #0b1220 !important; background: #0b1220 !important; }
          body { background-color: #0b1220 !important; background: #0b1220 !important; }
        ` }} />
        
        {/* Additional meta tags for better SEO */}
        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=5" />
        <meta name="format-detection" content="telephone=no" />
        <meta name="theme-color" content="#0b1220" />
        <meta name="msapplication-TileColor" content="#0b1220" />
        <meta name="apple-mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-status-bar-style" content="default" />
        <meta name="apple-mobile-web-app-title" content="StruktX" />
        <meta name="application-name" content="StruktX" />
        <meta name="color-scheme" content="dark light" />

        {/* Pre-init theme to avoid FOUC/white flash */}
        <script dangerouslySetInnerHTML={{ __html: `!function(){try{var t=localStorage.getItem('theme');var d=document.documentElement;if(t==='dark'||(!t&&window.matchMedia('(prefers-color-scheme: dark)').matches)){d.classList.add('dark');d.classList.remove('light');}else{d.classList.add('light');d.classList.remove('dark');}}catch(e){}}();` }} />

        {/* Early dark paint to prevent white flash and mark load state */}
        <style dangerouslySetInnerHTML={{ __html: `html{background:#0b1220}body{background:#0b1220}` }} />
        <script dangerouslySetInnerHTML={{ __html: `!function(){try{var d=document;var mark=function(){var b=d.body;b&&b.classList.add('loaded')};d.addEventListener('DOMContentLoaded',mark);window.addEventListener('load',mark);}catch(e){}}();` }} />

        {/* Structured Data for Rich Snippets */}
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              "@context": "https://schema.org",
              "@type": "SoftwareApplication",
              "name": "StruktX",
              "description": "A lean core with swappable pieces. StruktX is a configurable, typed AI framework with swappable LLM, classifier, handlers, and optional memory.",
              "url": "https://struktx.vercel.app",
              "applicationCategory": "DeveloperApplication",
              "operatingSystem": "Any",
              "programmingLanguage": "Python",
              "author": {
                "@type": "Organization",
                "name": "StruktX Team"
              },
              "offers": {
                "@type": "Offer",
                "price": "0",
                "priceCurrency": "USD"
              },
              "image": "https://struktx.vercel.app/logo-blue.png",
              "logo": "https://struktx.vercel.app/logo-blue.png",
              "screenshot": "https://struktx.vercel.app/nobg-both-white.png",
              "keywords": "AI framework, machine learning, LLM, Python, type safety, configurable, swappable components",
              "datePublished": "2024-01-01",
              "dateModified": "2024-01-01"
            })
          }}
        />
        
        {/* Organization Schema */}
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              "@context": "https://schema.org",
              "@type": "Organization",
              "name": "StruktX",
              "url": "https://struktx.vercel.app",
              "logo": "https://struktx.vercel.app/logo-blue.png",
              "description": "A lean core with swappable pieces. StruktX is a configurable, typed AI framework.",
              "sameAs": [
                "https://github.com/struktx/struktx",
                "https://twitter.com/struktx"
              ]
            })
          }}
        />
      </head>
      <body className={`${inter.className} antialiased`} style={{ backgroundColor: '#0b1220' }}>
        <ThemeProvider>
          <ThemeLoadingProvider>
            <BackgroundManager />
            {children}
          </ThemeLoadingProvider>
        </ThemeProvider>
      </body>
    </html>
  )
} 