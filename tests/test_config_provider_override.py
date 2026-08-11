"""Test provider override in AgentConfig.auto_load()"""
import os
import pytest
from flagscale_agent.react.config import AgentConfig


def test_provider_override_openai(monkeypatch, tmp_path):
    """Test that provider='openai' override correctly switches to OpenAI config."""
    # Set up environment
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "sk-test-anthropic-key")
    monkeypatch.delenv("FLAGSCALE_AGENT_CONFIG", raising=False)
    
    # Mock get_config_path to return non-existent path
    import flagscale_agent.react.paths as paths_module
    monkeypatch.setattr(
        paths_module, 
        "get_config_path", 
        lambda: str(tmp_path / "nonexistent.yaml")
    )
    
    # Test default (should be anthropic)
    cfg_default = AgentConfig.auto_load()
    assert cfg_default.provider == "anthropic"
    assert cfg_default.api_key == "sk-test-anthropic-key"
    
    # Test with provider='openai' override
    cfg_openai = AgentConfig.auto_load(provider="openai")
    assert cfg_openai.provider == "openai"
    assert cfg_openai.api_key == "sk-test-openai-key"
    assert cfg_openai.base_url == "https://api.openai.com/v1"
    assert cfg_openai.model == "gpt-4o"  # Default OpenAI model


def test_provider_override_with_model(monkeypatch, tmp_path):
    """Test that model override also works with provider override."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    monkeypatch.delenv("FLAGSCALE_AGENT_CONFIG", raising=False)
    
    import flagscale_agent.react.paths as paths_module
    monkeypatch.setattr(
        paths_module,
        "get_config_path",
        lambda: str(tmp_path / "nonexistent.yaml")
    )
    
    cfg = AgentConfig.auto_load(provider="openai", model="gpt-4o-mini")
    assert cfg.provider == "openai"
    assert cfg.model == "gpt-4o-mini"
    assert cfg.api_key == "sk-test-key"


def test_provider_override_without_api_key(monkeypatch, tmp_path):
    """Test that provider override without API key sets api_key to None."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("FLAGSCALE_AGENT_CONFIG", raising=False)
    
    import flagscale_agent.react.paths as paths_module
    monkeypatch.setattr(
        paths_module,
        "get_config_path",
        lambda: str(tmp_path / "nonexistent.yaml")
    )
    
    cfg = AgentConfig.auto_load(provider="openai")
    assert cfg.provider == "openai"
    assert cfg.api_key is None
