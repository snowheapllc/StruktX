"""
Basic tests for StruktX AI framework
"""

import pytest
from strukt.config import StruktConfig
from strukt.ai import create


def test_config_creation():
    """Test that we can create a basic configuration"""
    config = StruktConfig()
    assert config is not None
    assert hasattr(config, 'classifier')


def test_ai_initialization():
    """Test that we can initialize the AI framework"""
    config = StruktConfig()
    ai = create(config)
    assert ai is not None
    assert hasattr(ai, '_engine')


@pytest.mark.asyncio
async def test_ai_chat_method():
    """Test that the chat method exists and is callable"""
    config = StruktConfig()
    ai = create(config)
    
    # Test that the method exists and is callable
    assert hasattr(ai, 'invoke')
    assert callable(ai.invoke)


def test_package_imports():
    """Test that all main modules can be imported"""
    try:
        from strukt import ai, config, engine, interfaces, logging, memory, middleware, types, utils
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import strukt modules: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
