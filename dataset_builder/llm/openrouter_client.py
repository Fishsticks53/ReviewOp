import os
import threading
import time
from openai import OpenAI
from .base_client import BaseLLMClient
from .openai_client import OpenAIClient
from ..config import BuilderConfig

class OpenRouterClient(OpenAIClient):
    _lock = threading.Lock()
    _last_call = 0.0
    _min_interval = 2.0  # ~30 RPM for free tier safety
    
    def __init__(self, cfg: BuilderConfig):
        # OpenRouter is OpenAI-compatible but requires custom base_url and headers
        BaseLLMClient.__init__(self, cfg)
        
        raw_key = os.environ.get("OPENROUTER_API_KEY")
        api_key = raw_key.strip() if raw_key else None
        
        # OpenRouter specific headers
        extra_headers = {}
        referer = os.environ.get("OPENROUTER_REFERER")
        if referer:
            extra_headers["HTTP-Referer"] = referer.strip()
        
        title = os.environ.get("OPENROUTER_TITLE")
        if title:
            extra_headers["X-Title"] = title.strip()
        else:
            extra_headers["X-Title"] = "ReviewOp"
            
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers=extra_headers
        )

    def _generate_inner(self, prompt: str, system_prompt: str | None = None, **kwargs) -> str:
        with self._lock:
            now = time.time()
            elapsed = now - self._last_call
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self.__class__._last_call = time.time()
            
        return super()._generate_inner(prompt, system_prompt=system_prompt, **kwargs)
