"use client";

import { CodeBlock } from "@/components/ui/code-block";
import { cn } from "@/lib/utils";
import { useEffect, useRef, useState } from "react";
import { gsap } from "gsap";
import { AnimatedLocation } from "@/components/AnimatedLocation";
import { AnimatedTimeResponse, TimeResponse } from "@/components/AnimatedTimeResponse";
import { TimeServicePreview } from "@/components/TimeServicePreview";

const installCode = `# Install with core dependencies only
uv pip install struktx

# Install with LLM support (LangChain)
uv pip install struktx[llm]

# Install with all optional dependencies
uv pip install struktx[all]`;

const llmCode = `from strukt import LLMClientConfig

llm = LLMClientConfig(
  "langchain_openai:ChatOpenAI",
  dict(model="gpt-4o-mini"),
)`;

const classifierCode = `from strukt import ClassifierConfig
from strukt.classifiers.llm_classifier import DefaultLLMClassifier, DEFAULT_CLASSIFIER_TEMPLATE

classifier = ClassifierConfig(
  DefaultLLMClassifier,
  dict(
    prompt_template=DEFAULT_CLASSIFIER_TEMPLATE,
    allowed_types=["time_service", "general", "memory_extraction"],
    max_parts=4,
  ),
)`;

const memoryCode = `from strukt import MemoryConfig
from strukt.memory import UpstashVectorMemoryEngine

memory = MemoryConfig(
  factory=UpstashVectorMemoryEngine,
  params=dict(
    index_url="...",
    index_token="...",
  ),
  use_store=True,
  augment_llm=True,
)`;

const handlersCode = `from strukt import HandlersConfig
from strukt.examples.time_handler import TimeHandler

handlers = HandlersConfig(
  {"time_service": TimeHandler},
  default_route="general",
)`;

const middlewareCode = `from strukt import MiddlewareConfig
from strukt.examples.middleware import MemoryExtractionMiddleware

middleware = [
  MiddlewareConfig(MemoryExtractionMiddleware)
]`;

const appCreateCode = `from strukt import create, StruktConfig

app = create(StruktConfig(
  llm=llm,
  classifier=classifier,
  memory=memory,
  handlers=handlers,
  middleware=middleware,
))`;

const invokePrefix = 'app.invoke("I currently live in ';
const invokeSuffix = ', what is the time there?")';

const response = `{
  "response": "The current time in Beirut is 3:45 PM (UTC+3)",
  "status": "time_service",
  "context": {"user_id": null},
  "metadata": {
    "classification": {
      "query_type": "time_service",
      "parts": ["time", "Beirut"],
      "confidence": 0.95
    }
  }
}`;

// AnimatedLocation is now a separate component

export function CodeShowcase({ className }: { className?: string }) {
  const [isOpen, setIsOpen] = useState(false);
  // Shared cycling index to sync location and response
  const [cycleIndex, setCycleIndex] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setCycleIndex((i) => i + 1), 2000);
    return () => clearInterval(id);
  }, []);
  const contentRef = useRef<HTMLDivElement>(null);
  const chevronRef = useRef<SVGSVGElement>(null);
  // No DOM rewriting; we show the animated location inline in a separate code row

  useEffect(() => {
    if (contentRef.current) {
      gsap.set(contentRef.current, { height: 0, opacity: 0 });
    }
  }, []);

  useEffect(() => {
    // nothing to initialize for inline rendering
  }, []);

  const toggleDetails = () => {
    setIsOpen(!isOpen);
    
    if (contentRef.current && chevronRef.current) {
      if (!isOpen) {
        // Opening
        gsap.to(chevronRef.current, { rotation: 90, duration: 0.3, ease: 'power2.out' });
        gsap.to(contentRef.current, {
          height: 'auto',
          opacity: 1,
          duration: 0.4,
          ease: 'power2.out',
          onStart: () => {
            gsap.set(contentRef.current, { height: 'auto' });
          },
        });
      } else {
        // Closing
        gsap.to(chevronRef.current, { rotation: 0, duration: 0.3, ease: 'power2.out' });
        gsap.to(contentRef.current, {
          height: 0,
          opacity: 0,
          duration: 0.4,
          ease: 'power2.in',
        });
      }
    }
  };

  return (
    <section className={cn("relative", className)}>
      <div className="mx-auto max-w-5xl px-4 sm:px-6 lg:px-8">
        <div className="glass-panel gradient-border p-8 shine">
          <div className="text-center mb-8">
            <h3 className="text-2xl font-semibold text-dark-900 dark:text-white">Build in minutes</h3>
            <p className="text-dark-600 dark:text-white/70 mt-2">Install and ship with a few lines of code</p>
          </div>
          <div className="space-y-6">
            <CodeBlock language="bash" code={installCode} />
            
            <div className="border border-white dark:border-white/10 rounded-lg">
              <button 
                onClick={toggleDetails}
                className="w-full cursor-pointer text-lg font-medium text-dark-900 dark:text-white hover:text-blue-600 dark:hover:text-blue-400 transition-colors focus:outline-none rounded-lg p-4 text-left"
                style={{ listStyle: 'none' }}
              >
                <span className="flex items-center gap-2">
                  <svg 
                    ref={chevronRef}
                    className="w-4 h-4" 
                    fill="none" 
                    stroke="currentColor" 
                    viewBox="0 0 24 24"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                  Show complete example
                </span>
              </button>
              <div 
                ref={contentRef}
                className="overflow-hidden"
                style={{ listStyle: 'none' }}
              >
                <div className="p-4 pt-0 space-y-6">
                  <div>
                    <h4 className="text-sm font-medium text-dark-700 dark:text-white/80 mb-2">LLM:</h4>
                    <CodeBlock language="python" code={llmCode} />
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-dark-700 dark:text-white/80 mb-2">Classifier:</h4>
                    <CodeBlock language="python" code={classifierCode} />
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-dark-700 dark:text-white/80 mb-2">Memory:</h4>
                    <CodeBlock language="python" code={memoryCode} />
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-dark-700 dark:text-white/80 mb-2">Handlers:</h4>
                    <CodeBlock language="python" code={handlersCode} />
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-dark-700 dark:text-white/80 mb-2">Middleware:</h4>
                    <CodeBlock language="python" code={middlewareCode} />
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-dark-700 dark:text-white/80 mb-2">Create App & Invoke:</h4>
                    <CodeBlock language="python" code={appCreateCode} />
                    <div className="relative rounded-xl overflow-hidden border border-white/20 dark:border-white/10 bg-white/5 dark:bg-dark-800/50 backdrop-blur-sm shadow-lg mt-3">
                      <pre className="p-4 text-dark-900 dark:text-white font-mono text-sm leading-relaxed overflow-x-auto">
                        <code className="text-dark-900 dark:text-white">
                          {invokePrefix}
                          <span className="align-baseline inline-block">
                            <AnimatedLocation index={cycleIndex} autoplay={false} />
                          </span>
                          {invokeSuffix}
                        </code>
                      </pre>
                    </div>
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-dark-700 dark:text-white/80 mb-2">Response:</h4>
                    <div className="mt-3">
                      <TimeServicePreview index={cycleIndex} />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}