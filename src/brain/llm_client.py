"""
OpenAI API client wrapper with retry logic and caching.
"""

import os
import json
import time
import hashlib
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from src.config import get_config


@dataclass
class LLMResponse:
    """Response from the LLM."""
    content: str
    model: str
    usage: Dict[str, int]
    cached: bool = False
    latency_ms: float = 0


class LLMClient:
    """OpenAI API client with retry logic and response caching."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        """Initialize the LLM client.

        Args:
            api_key: OpenAI API key (or from OPENAI_API_KEY env var)
            model: Model to use (default from config)
            cache_dir: Directory for response cache
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed. Run: pip install openai")

        self.config = get_config()

        # Get API key
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY env var or pass api_key.")

        # Initialize client
        self.client = OpenAI(api_key=self.api_key)

        # Model
        self.model = model or self.config.llm.model

        # Caching
        self.cache_enabled = self.config.llm.cache_evaluations
        self.cache_dir = cache_dir or Path(__file__).parent.parent.parent / ".cache" / "llm"
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, LLMResponse] = {}

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests

        # Stats
        self.total_requests = 0
        self.cached_requests = 0
        self.total_tokens = 0

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.3,
        use_cache: bool = True,
    ) -> LLMResponse:
        """Send a completion request to the LLM.

        Args:
            prompt: User prompt
            system_prompt: System prompt (instructions)
            max_tokens: Maximum response tokens
            temperature: Sampling temperature (0-1)
            use_cache: Whether to use response cache

        Returns:
            LLMResponse with content and metadata
        """
        # Check cache
        cache_key = self._cache_key(prompt, system_prompt)
        if use_cache and self.cache_enabled:
            cached = self._get_cached(cache_key)
            if cached:
                self.cached_requests += 1
                return cached

        # Rate limiting
        self._rate_limit()

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Make request
        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:
            # Retry once on failure
            time.sleep(1)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        latency = (time.time() - start_time) * 1000

        # Parse response
        content = response.choices[0].message.content
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        result = LLMResponse(
            content=content,
            model=response.model,
            usage=usage,
            cached=False,
            latency_ms=latency,
        )

        # Update stats
        self.total_requests += 1
        self.total_tokens += usage["total_tokens"]

        # Cache response
        if use_cache and self.cache_enabled:
            self._set_cached(cache_key, result)

        return result

    def complete_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Get a JSON response from the LLM.

        Args:
            prompt: User prompt (should request JSON output)
            system_prompt: System prompt
            max_tokens: Maximum response tokens
            use_cache: Whether to use cache

        Returns:
            Parsed JSON dict (empty dict on parse failure)
        """
        response = self.complete(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=0.1,  # Lower temp for JSON
            use_cache=use_cache,
        )

        # Parse JSON from response
        content = response.content.strip()

        # Try to extract JSON from markdown code blocks
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            if end > start:
                content = content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            if end > start:
                content = content[start:end].strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to find JSON object in response
            try:
                start = content.find("{")
                end = content.rfind("}") + 1
                if start >= 0 and end > start:
                    return json.loads(content[start:end])
            except json.JSONDecodeError:
                pass

            return {}

    def _cache_key(self, prompt: str, system_prompt: Optional[str]) -> str:
        """Generate cache key for a request."""
        combined = f"{system_prompt or ''}\n---\n{prompt}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _get_cached(self, key: str) -> Optional[LLMResponse]:
        """Get cached response."""
        # Check memory cache first
        if key in self._cache:
            response = self._cache[key]
            response.cached = True
            return response

        # Check disk cache
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                response = LLMResponse(
                    content=data["content"],
                    model=data["model"],
                    usage=data["usage"],
                    cached=True,
                )
                self._cache[key] = response
                return response
            except Exception:
                pass

        return None

    def _set_cached(self, key: str, response: LLMResponse) -> None:
        """Cache a response."""
        # Memory cache
        self._cache[key] = response

        # Disk cache
        cache_file = self.cache_dir / f"{key}.json"
        try:
            with open(cache_file, "w") as f:
                json.dump({
                    "content": response.content,
                    "model": response.model,
                    "usage": response.usage,
                }, f)
        except Exception:
            pass

    def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_requests": self.total_requests,
            "cached_requests": self.cached_requests,
            "cache_hit_rate": f"{self.cached_requests / max(1, self.total_requests):.1%}",
            "total_tokens": self.total_tokens,
            "model": self.model,
        }


# Singleton instance
_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get the global LLM client instance."""
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
