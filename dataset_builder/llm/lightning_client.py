import os
import threading
import time
from litai import LLM
from .base_client import BaseLLMClient
from ..config import BuilderConfig

class LightningClient(BaseLLMClient):
    """
    Lightning AI client using the litai library.
    """
    _lock = threading.Lock()
    _last_call = 0.0
    _min_interval = 1.0  # 60 RPM default safety

    def __init__(self, cfg: BuilderConfig):
        super().__init__(cfg)
        raw_key = os.environ.get("LIGHTNING_API_KEY")
        self.api_key = raw_key.strip() if raw_key else None
        
        # Initialize the LitAI LLM
        # api_key is required by LLM constructor if not in env
        self.client = LLM(model=self.model, api_key=self.api_key)

    def _generate_inner(self, prompt: str, system_prompt: str | None = None, **kwargs) -> str:
        # Rate limiting to ensure stability
        with self._lock:
            now = time.time()
            elapsed = now - self._last_call
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self.__class__._last_call = time.time()
            
        # litai.LLM.chat returns the string response
        full_input = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        return self.client.chat(full_input, **kwargs)
