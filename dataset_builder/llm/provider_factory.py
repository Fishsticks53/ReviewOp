from __future__ import annotations
from ..config import BuilderConfig
from .base_client import BaseLLMClient
from .openai_client import OpenAIClient
from .anthropic_client import AnthropicClient
from .groq_client import GroqClient
from .openrouter_client import OpenRouterClient
from .huggingface_client import HuggingFaceClient
from .ollama_client import OllamaClient
from .lightning_client import LightningClient
from .gemini_client import GeminiClient

def get_llm_client(cfg: BuilderConfig, wrap_fallback: bool = True) -> BaseLLMClient:
    """Factory method to get the appropriate LLM client."""
    provider = str(cfg.llm_provider).lower()
    client = None
    
    if provider == "openai":
        client = OpenAIClient(cfg)
    elif provider in ("anthropic", "claude"):
        client = AnthropicClient(cfg)
    elif provider == "gemini":
        client = GeminiClient(cfg)
    elif provider == "groq":
        client = GroqClient(cfg)
    elif provider == "openrouter":
        client = OpenRouterClient(cfg)
    elif provider == "huggingface":
        client = HuggingFaceClient(cfg)
    elif provider == "ollama":
        client = OllamaClient(cfg)
    elif provider == "lightning":
        client = LightningClient(cfg)
    elif provider == "none":
        raise ValueError("LLM provider is set to 'none'")
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    if wrap_fallback and provider != "none":
        from .fallback_client import FallbackLLMClient
        return FallbackLLMClient(cfg, client)
    
    return client
