<div align="center">
  <img src="/docs/public/logo-blue.png" alt="StruktX Logo" width="120" height="120">
  <br>
  <img src="/docs/public/nobg-both-white.png" alt="StruktX" width="200">
  
  A configurable, typed AI framework with swappable LLM, classifier, handlers, and memory for Natural Language to Action apps
  
  [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>






## Quick Install

### Using uv (Recommended)
```bash
# Install with core dependencies only
uv pip install struktx-ai

# Install with LLM support (LangChain)
uv pip install struktx-ai[llm]

# Install with vector store support
uv pip install struktx-ai[vector]

# Install with all optional dependencies
uv pip install struktx-ai[all]

# Install for development
uv pip install struktx-ai[dev]
```

### Using pip
```bash
# Install with core dependencies only
pip install struktx-ai

# Install with LLM support (LangChain)
pip install struktx-ai[llm]

# Install with vector store support
pip install struktx-ai[vector]

# Install with all optional dependencies
pip install struktx-ai[all]
```

### Source
```bash
# Create virtual env
uv venv

# Activate environment
source .venv/bin/activate

# Sync and install pyproject dependencies
uv sync

# Install finall Strukt package
uv pip install struktx-ai[all]
```

## Quick Start with LangChain
```python
from strukt import create, StruktConfig, HandlersConfig, LLMClientConfig, ClassifierConfig
from strukt.classifiers.llm_classifier import DefaultLLMClassifier
from strukt.prompts import DEFAULT_CLASSIFIER_TEMPLATE
from strukt.examples.time_handler import TimeHandler

from langchain_openai import ChatOpenAI
from strukt.langchain_helpers import LangChainLLMClient

# Create LangChain LLM client
langchain_llm = ChatOpenAI(api_key="your-openai-api-key")
llm = LangChainLLMClient(langchain_llm)

app = create(StruktConfig(
    llm=LLMClientConfig(llm),
    classifier=ClassifierConfig(
        factory=DefaultLLMClassifier(
            llm=llm,
            prompt_template=DEFAULT_CLASSIFIER_TEMPLATE,
            allowed_types=["time_service", "weather", "general"],
            max_parts=4,
        )
    ),
    handlers=HandlersConfig(
        registry={"time_service": TimeHandler(llm)},
        default_route="general",
    ),
))

app.invoke("What is the time in Tokyo?")
```

## What is StruktX?

StruktX is a Python framework for building AI applications with focus on natural language -> actions

- **🔄 Swappable Components**: LLM clients, classifiers, handlers, and memory engines
- **📝 Type Safety**: Full type hints and Pydantic models
- **⚙️ Configurable**: Factory-based configuration system
- **🔌 Extensible**: Easy to add custom components
- **🧠 Memory Support**: Optional conversation memory
- **🔧 Middleware**: Pre/post processing hooks

## Features

- **LLM Integration**: Support for any LLM via the `LLMClient` interface
- **Query Classification**: Route requests to appropriate handlers
- **Structured Outputs**: Pydantic model integration
- **Memory Engines**: Conversation history and context
- **Middleware System**: Pre/post processing hooks
- **LangChain Helpers**: Easy integration with LangChain ecosystem

## Requirements

- Python 3.10+
- Core: `pydantic>=2.0.0`, `python-dotenv>=1.0.0`
- Optional: LangChain packages, vector stores, etc.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please report any issues, bugs, improvements, or features on GitHub.

## Support

- 📖 [Documentation](https://struktx.snowheap.ai/docs)
- 🐛 [Issue Tracker](https://github.com/snowheapllc/StruktX/issues)
- 💬 [Discussions](https://github.com/snowheapllc/StruktX/discussions)

<div align="right">
  <a href="https://github.com/snowheapllc">
    <img src="/docs/public/snowheap.png" alt="Snowheap LLC Logo" width="100">
  </a>
</div>

