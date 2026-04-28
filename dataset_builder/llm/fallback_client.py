from __future__ import annotations
import logging
from .base_client import BaseLLMClient
from ..config import BuilderConfig

logger = logging.getLogger(__name__)

class FallbackLLMClient(BaseLLMClient):
    """
    A client that wraps multiple providers and falls back to the next one
    if the primary one fails with a rate limit or quota error.
    """
    def __init__(self, cfg: BuilderConfig, primary_client: BaseLLMClient):
        super().__init__(cfg)
        self.primary = primary_client
        self.fallback_client = None
        self._initialized_fallback = False

    def _ensure_fallback(self):
        if self._initialized_fallback:
            return
        self._initialized_fallback = True
        
        # Determine fallback provider
        import os
        primary_provider = str(self.cfg.llm_provider).lower()
        
        # If primary is OpenAI, fallback to Groq or Anthropic
        fallback_provider = None
        if primary_provider == "openai":
            if os.environ.get("GROQ_API_KEY"):
                fallback_provider = "groq"
            elif os.environ.get("CLAUDE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"):
                fallback_provider = "anthropic"
        elif primary_provider == "groq":
            if os.environ.get("OPENAI_API_KEY"):
                fallback_provider = "openai"
        
        if fallback_provider:
            from .provider_factory import get_llm_client
            from dataclasses import replace
            from ..config import get_env_model
            
            fallback_cfg = replace(self.cfg, 
                llm_provider=fallback_provider,
                llm_model=get_env_model(fallback_provider)
            )
            try:
                self.fallback_client = get_llm_client(fallback_cfg, wrap_fallback=False)
                logger.info(f"Initialized fallback provider: {fallback_provider} (model: {fallback_cfg.llm_model})")
            except Exception as e:
                logger.warning(f"Failed to initialize fallback provider {fallback_provider}: {e}")

    def generate(self, prompt: str, system_prompt: str | None = None, **kwargs) -> str:
        from ..orchestrator.telemetry import GLOBAL_STATS
        try:
            return self.primary.generate(prompt, system_prompt=system_prompt, **kwargs)
        except Exception as e:
            error_str = str(e).lower()
            is_transient = (
                "rate limit" in error_str or 
                "429" in error_str or 
                "quota" in error_str or 
                "too many requests" in error_str
            )
            
            if is_transient:
                self._ensure_fallback()
                if self.fallback_client:
                    logger.warning(f"Primary provider {self.cfg.llm_provider} failed ({e}). Falling back to {self.fallback_client.cfg.llm_provider}...")
                    GLOBAL_STATS.record_llm_call(fallback=True)
                    return self.fallback_client.generate(prompt, system_prompt=system_prompt, **kwargs)
            
            raise e

    def _generate_inner(self, prompt: str, system_prompt: str | None = None, **kwargs) -> str:
        return ""
