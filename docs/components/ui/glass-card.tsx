import { ReactNode } from "react"
import { cn } from "@/lib/utils"

export function GlassCard({ className, children }: { className?: string; children: ReactNode }) {
  return (
    <div className={cn("glass-panel gradient-border shine p-6", className)}>
      {children}
    </div>
  )
}