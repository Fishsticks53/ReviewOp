import os
import threading
import time
from openai import OpenAI
from .base_client import BaseLLMClient
from .openai_client import OpenAIClient
from ..config import BuilderConfig

class OllamaClient(OpenAIClient):
    """
    Ollama client using the OpenAI-compatible endpoint.
    Optimized for local execution with configurable concurrency control.
    """
    _lock = threading.Lock()
    _last_call = 0.0
    
    # Local GPUs usually prefer sequential or low-concurrency access.
    # We default to a small interval to prevent VRAM spikes.
    _min_interval = float(os.environ.get("OLLAMA_MIN_INTERVAL", "0.5"))

    def __init__(self, cfg: BuilderConfig):
        # Initialize BaseLLMClient to set self.model
        BaseLLMClient.__init__(self, cfg)
        
        # Ollama local default endpoint
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        
        # Ollama doesn't require an API key by default, but the client needs a string
        api_key = os.environ.get("OLLAMA_API_KEY", "ollama")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def _generate_inner(self, prompt: str, system_prompt: str | None = None, **kwargs) -> str:
        # Enforce rate limiting to protect local hardware
        with self._lock:
            now = time.time()
            elapsed = now - self._last_call
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self.__class__._last_call = time.time()
            
        return super()._generate_inner(prompt, system_prompt=system_prompt, **kwargs)
