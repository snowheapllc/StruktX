<div align="center">
  <img src="https://github.com/aymanHS-code/struktx-docs/blob/main/logo/logo.png" alt="StruktX Logo" width="120" height="120">
  <br>
  <img src="https://github.com/aymanHS-code/struktx-docs/blob/main/logo/nobg-both.png" alt="StruktX" width="200">
</div>

# StruktX

A configurable, typed AI framework with swappable LLM, classifier, handlers, and optional memory.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Quick Install

### Using uv (Recommended)
```bash
# Install with core dependencies only
uv pip install struktx

# Install with LLM support (LangChain)
uv pip install struktx[llm]

# Install with vector store support
uv pip install struktx[vector]

# Install with all optional dependencies
uv pip install struktx[all]

# Install for development
uv pip install struktx[dev]
```

### Using pip
```bash
# Install with core dependencies only
pip install struktx

# Install with LLM support (LangChain)
pip install struktx[llm]

# Install with vector store support
pip install struktx[vector]

# Install with all optional dependencies
pip install struktx[all]
```

### Source
```bash
uv venv

source .venv/bin/activate

uv sync

uv pip install struktx[all]
```

## Quick Start with LangChain
```python
from strukt import create, StruktConfig, HandlersConfig, LLMClientConfig, ClassifierConfig
from strukt.classifiers.llm_classifier import DefaultLLMClassifier, DEFAULT_CLASSIFIER_TEMPLATE
from strukt.examples.time_handler import TimeHandler

# Example with LangChain OpenAI integration
from langchain_openai import ChatOpenAI
from strukt.langchain_helpers import LangChainLLMClient

# Create LangChain LLM client
langchain_llm = ChatOpenAI(api_key="your-openai-api-key")
llm_client = LangChainLLMClient(langchain_llm)

app = create(StruktConfig(
    llm=LLMClientConfig(factory=lambda **_: llm_client),
    classifier=ClassifierConfig(
        factory=lambda llm, **_: DefaultLLMClassifier(
            llm=llm,
            prompt_template=DEFAULT_CLASSIFIER_TEMPLATE,
            allowed_types=["time_service", "weather", "general"],
            max_parts=4,
        )
    ),
    handlers=HandlersConfig(
        registry={"time_service": lambda llm, **_: TimeHandler(llm)},
        default_route="general",
    ),
))
```

## What is StruktX?

StruktX is a Python framework for building AI applications with:

- **🔄 Swappable Components**: LLM clients, classifiers, handlers, and memory engines
- **📝 Type Safety**: Full type hints and Pydantic models
- **⚙️ Configurable**: Factory-based configuration system
- **🔌 Extensible**: Easy to add custom components
- **🧠 Memory Support**: Optional conversation memory
- **🔧 Middleware**: Pre/post processing hooks

## Documentation

📖 **Full Documentation**: [docs/README.md](docs/README.md)

The documentation includes:
- Component architecture overview
- Configuration examples
- Custom handler/classifier development
- LangChain integration
- Middleware examples
- Best practices

## Features

- **LLM Integration**: Support for any LLM via the `LLMClient` interface
- **Query Classification**: Route requests to appropriate handlers
- **Structured Outputs**: Pydantic model integration
- **Memory Engines**: Conversation history and context
- **Middleware System**: Pre/post processing hooks
- **LangChain Helpers**: Easy integration with LangChain ecosystem

## Requirements

- Python 3.8.1+
- Core: `pydantic>=2.0.0`, `python-dotenv>=1.0.0`
- Optional: LangChain packages, vector stores, etc.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## Support

- 📖 [Documentation](docs/README.md)
- 🐛 [Issue Tracker](https://github.com/struktx/struktx/issues)
- 💬 [Discussions](https://github.com/struktx/struktx/discussions)
