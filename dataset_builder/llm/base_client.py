from __future__ import annotations
from abc import ABC, abstractmethod
from ..config import BuilderConfig
from .disk_cache import LLMDiskCache

class BaseLLMClient(ABC):
    def __init__(self, cfg: BuilderConfig):
        self.cfg = cfg
        self.model = cfg.llm_model
        self.cache = LLMDiskCache() if getattr(cfg, "use_cache", True) else None

    def generate(self, prompt: str, system_prompt: str | None = None, **kwargs) -> str:
        from ..orchestrator.telemetry import GLOBAL_STATS
        if self.cache:
            cached = self.cache.get(prompt, self.model, system_prompt=system_prompt)
            if cached:
                GLOBAL_STATS.record_llm_call(cached=True)
                return cached
        
        import time
        import random
        max_retries = 3
        base_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                response = self._generate_inner(prompt, system_prompt=system_prompt, **kwargs)
                GLOBAL_STATS.record_llm_call(cached=False)
                if self.cache:
                    self.cache.set(prompt, self.model, response, system_prompt=system_prompt)
                return response
            except Exception as e:
                error_str = str(e).lower()
                is_transient = (
                    "rate limit" in error_str or 
                    "429" in error_str or 
                    "quota" in error_str or 
                    "too many requests" in error_str
                )
                
                if is_transient and attempt < max_retries - 1:
                    # For 429s, we want a more aggressive backoff
                    wait_factor = 4.0 if "429" in error_str else 2.0
                    delay = (base_delay * (wait_factor ** attempt)) + (random.random() * 2.0)
                    print(f"Retrying LLM call (attempt {attempt + 1}/{max_retries}) after {delay:.1f}s due to error: {e}")
                    time.sleep(delay)
                    continue
                
                GLOBAL_STATS.record_llm_call(failed=True)
                raise e
        
        return ""

    @abstractmethod
    def _generate_inner(self, prompt: str, system_prompt: str | None = None, **kwargs) -> str:
        """Actual generation logic."""
        pass
