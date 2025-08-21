"use client"

import * as React from "react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { Check, Copy } from "lucide-react"

type CodeBlockProps = {
  code: string
  language?: string
  filename?: string
  className?: string
  wrap?: boolean
  showExample?: boolean
}

// HTML escape to safely render code (so literals like <specify a location> are preserved)
const escapeHtml = (text: string) =>
  text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

// Manual syntax highlighting function
const highlightCode = (code: string, language: string) => {
  const src = escapeHtml(code)
  if (language === 'bash') {
    return src
      .replace(/(#.*$)/gm, '<span class="text-blue-300">$1</span>') // Comments
      .replace(/\b(uv|pip|install)\b/g, '<span class="text-green-300">$1</span>') // Commands
      .replace(/\b(struktx)\b/g, '<span class="text-yellow-300">$1</span>') // Package name
      .replace(/\[([^\]]+)\]/g, '<span class="text-purple-300">[$1]</span>') // Brackets
  }
  
  if (language === 'python') {
    const keywords = [
      'from', 'import', 'def', 'class', 'lambda', 'return',
      'if', 'else', 'try', 'except', 'with', 'as', 'in', 'is',
      'and', 'or', 'not', 'True', 'False', 'None'
    ]

    // Function to only replace outside of spans
    const safeReplace = (text: string, regex: RegExp, replacement: string) => {
      return text.split(/(<span.*?<\/span>)/g) // keep spans intact
        .map(part => regex.test(part) && !part.startsWith('<span')
          ? part.replace(regex, replacement)
          : part
        )
        .join('')
    }

    let highlighted = src

    // Strings
    highlighted = highlighted.replace(/(\".*?\"|\'.*?\')/g, '<span class="text-green-300">$1</span>')

    // Comments
    highlighted = safeReplace(highlighted, /(#.*$)/gm, '<span class="text-gray-400">$1</span>')

    // Keywords (including from and import)
    keywords.forEach(keyword => {
      const regex = new RegExp(`\\b${keyword}\\b`, 'g')
      highlighted = safeReplace(highlighted, regex, '<span class="text-blue-300">$&</span>')
    })

    // `from ... import ...` - handle the module names
    highlighted = safeReplace(highlighted,
      /<span class="text-blue-300">from<\/span>\s+([\w.]+)\s+<span class="text-blue-300">import<\/span>\s+([\w.,\s*]+)/g,
      '<span class="text-blue-300">from</span> <span class="text-purple-300">$1</span> <span class="text-blue-300">import</span> <span class="text-purple-300">$2</span>'
    )

    // Numbers
    highlighted = safeReplace(highlighted, /\b(\d+)\b/g, '<span class="text-yellow-300">$1</span>')

    // Function calls
    highlighted = safeReplace(highlighted, /\b(\w+)\s*\(/g, '<span class="text-purple-300">$1</span>(')

    // Dicts + lists
    highlighted = safeReplace(highlighted, /(\{[^}]*\})/g, '<span class="text-orange-300">$1</span>')
    highlighted = safeReplace(highlighted, /(\[[^\]]*\])/g, '<span class="text-orange-300">$1</span>')

    return highlighted
  }
  
  return src
}

export function CodeBlock({ code, language = "bash", filename, className, wrap = true, showExample = false }: CodeBlockProps) {
  const [copied, setCopied] = React.useState(false)
  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(code)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      // ignore
    }
  }

  const highlightedCode = highlightCode(code, language)

  return (
    <div className={cn("relative rounded-xl overflow-hidden border border-white/20 dark:border-white/10 bg-white/5 dark:bg-dark-800/50 backdrop-blur-sm shadow-lg", className)}>
      {filename && (
        <div className="flex items-center justify-between px-4 py-2 text-xs bg-white/10 dark:bg-dark-700/50 text-dark-800 dark:text-white/80 border-b border-white/10 dark:border-white/5 rounded-lg">
          <div className="flex items-center gap-2">
            <span className="font-mono">{filename}</span>
            {showExample && (
              <span className="inline-block text-[9px] uppercase tracking-wide px-1.5 py-0.5 rounded bg-primary-600/15 text-primary-700 dark:text-primary-300 ring-1 ring-primary-600/30 font-medium">
                Example
              </span>
            )}
          </div>
          <Button size="sm" variant="ghost" onClick={onCopy} className="h-8 px-2 text-dark-600 dark:text-white/70 hover:text-dark-900 dark:hover:text-white hover:bg-white/10 dark:hover:bg-white/5">
            {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
          </Button>
        </div>
      )}
      {!filename && (
        <div className="absolute right-2 top-2 z-10">
          <Button size="sm" variant="ghost" onClick={onCopy} className="h-8 px-2 text-dark-600 dark:text-white/70 hover:text-dark-900 dark:hover:text-white hover:bg-white/10 dark:hover:bg-white/5 bg-white/10 dark:bg-dark-700/50 backdrop-blur-sm">
            {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
          </Button>
        </div>
      )}
      <pre className="p-4 text-dark-900 dark:text-white font-mono text-sm leading-relaxed overflow-x-auto">
        <code 
          className="text-dark-900 dark:text-white"
          dangerouslySetInnerHTML={{ __html: highlightedCode }}
        />
      </pre>
    </div>
  )
}

