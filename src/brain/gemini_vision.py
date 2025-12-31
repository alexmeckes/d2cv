"""
Gemini Vision client for screenshot understanding.

Uses Gemini Flash for fast, affordable image analysis:
- Item evaluation from screenshots
- Game state understanding
- Stuck/error detection
"""

import base64
import time
import json
import hashlib
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import cv2

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from src.config import get_config
from src.state.session_logger import get_logger


@dataclass
class VisionResponse:
    """Response from vision model."""
    content: Dict[str, Any]
    raw_text: str
    latency_ms: float
    cached: bool = False
    tokens_used: int = 0


@dataclass
class ItemAnalysis:
    """Analysis of an item from screenshot."""
    name: str
    base_type: str
    rarity: str
    identified: bool
    stats: List[str]
    value_assessment: str  # "trash", "keep", "valuable", "gg"
    reason: str
    pickup: bool
    priority: int  # 1-10


@dataclass
class ScreenAnalysis:
    """Analysis of full game screen."""
    location: str  # "town", "durance_2", "unknown", etc.
    health_percent: float
    mana_percent: float
    enemies_visible: int
    items_on_ground: List[Dict[str, Any]]
    portals_visible: bool
    is_dead: bool
    suggested_action: str


class GeminiVisionClient:
    """Gemini-powered vision for game understanding."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-1.5-flash",
        cache_responses: bool = True,
    ):
        """Initialize Gemini vision client.

        Args:
            api_key: Gemini API key (or uses GEMINI_API_KEY env var)
            model: Model to use (gemini-1.5-flash recommended)
            cache_responses: Whether to cache identical screenshots
        """
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "google-generativeai not installed. "
                "Run: pip install google-generativeai"
            )

        self.config = get_config()
        self.logger = get_logger("gemini_vision")

        # Get API key
        api_key = api_key or self.config.get("gemini.api_key") or self._get_env_key()
        if not api_key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY env var "
                "or add gemini.api_key to config."
            )

        # Configure Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.model_name = model

        # Response cache
        self.cache_responses = cache_responses
        self._cache: Dict[str, VisionResponse] = {}
        self._cache_ttl = 60  # Cache for 60 seconds
        self._cache_timestamps: Dict[str, float] = {}

        # Stats
        self.total_calls = 0
        self.cache_hits = 0
        self.total_tokens = 0
        self.total_latency_ms = 0

    def _get_env_key(self) -> Optional[str]:
        """Get API key from environment."""
        import os
        return os.environ.get("GEMINI_API_KEY")

    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert OpenCV image to base64."""
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')

    def _get_cache_key(self, image: np.ndarray, prompt: str) -> str:
        """Generate cache key from image and prompt."""
        # Downsample image for faster hashing
        small = cv2.resize(image, (64, 36))
        img_hash = hashlib.md5(small.tobytes()).hexdigest()
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
        return f"{img_hash}_{prompt_hash}"

    def _check_cache(self, key: str) -> Optional[VisionResponse]:
        """Check if response is cached and valid."""
        if not self.cache_responses:
            return None

        if key in self._cache:
            timestamp = self._cache_timestamps.get(key, 0)
            if time.time() - timestamp < self._cache_ttl:
                self.cache_hits += 1
                response = self._cache[key]
                response.cached = True
                return response
            else:
                # Expired
                del self._cache[key]
                del self._cache_timestamps[key]
        return None

    def _store_cache(self, key: str, response: VisionResponse) -> None:
        """Store response in cache."""
        if self.cache_responses:
            self._cache[key] = response
            self._cache_timestamps[key] = time.time()

            # Limit cache size
            if len(self._cache) > 100:
                oldest_key = min(self._cache_timestamps, key=self._cache_timestamps.get)
                del self._cache[oldest_key]
                del self._cache_timestamps[oldest_key]

    def _call_gemini(
        self,
        image: np.ndarray,
        prompt: str,
        use_cache: bool = True,
    ) -> VisionResponse:
        """Make a call to Gemini with image.

        Args:
            image: OpenCV BGR image
            prompt: Text prompt
            use_cache: Whether to use caching

        Returns:
            VisionResponse with parsed content
        """
        # Check cache
        cache_key = self._get_cache_key(image, prompt)
        if use_cache:
            cached = self._check_cache(cache_key)
            if cached:
                return cached

        self.total_calls += 1
        start_time = time.time()

        try:
            # Convert image to PIL format for Gemini
            from PIL import Image
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)

            # Make request
            response = self.model.generate_content(
                [prompt, pil_image],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Low temp for consistent outputs
                    max_output_tokens=1024,
                ),
            )

            latency = (time.time() - start_time) * 1000
            self.total_latency_ms += latency

            raw_text = response.text

            # Try to parse as JSON
            content = self._parse_json_response(raw_text)

            # Estimate tokens (rough)
            tokens = len(raw_text.split()) + 258  # ~258 tokens for image
            self.total_tokens += tokens

            result = VisionResponse(
                content=content,
                raw_text=raw_text,
                latency_ms=latency,
                tokens_used=tokens,
            )

            # Cache successful response
            self._store_cache(cache_key, result)

            self.logger.debug(f"Gemini call: {latency:.0f}ms, ~{tokens} tokens")

            return result

        except Exception as e:
            self.logger.error(f"Gemini call failed: {e}")
            return VisionResponse(
                content={"error": str(e)},
                raw_text=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Parse JSON from response, handling markdown code blocks."""
        # Remove markdown code blocks if present
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Return as raw text if not valid JSON
            return {"raw": text}

    def analyze_item(
        self,
        screenshot: np.ndarray,
        item_region: Optional[Tuple[int, int, int, int]] = None,
        character_class: str = "sorceress",
        build: str = "blizzard",
    ) -> ItemAnalysis:
        """Analyze an item from screenshot.

        Args:
            screenshot: Full game screenshot
            item_region: Optional (x, y, w, h) of item label region
            character_class: Character class for value assessment
            build: Build type for value assessment

        Returns:
            ItemAnalysis with pickup decision
        """
        # Crop to item region if specified
        if item_region:
            x, y, w, h = item_region
            # Add padding for context
            pad = 20
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(screenshot.shape[1], x + w + pad)
            y2 = min(screenshot.shape[0], y + h + pad)
            image = screenshot[y1:y2, x1:x2]
        else:
            image = screenshot

        prompt = f"""Analyze this Diablo 2 item screenshot. The player is a {build} {character_class}.

Return JSON with:
{{
    "name": "item name if visible",
    "base_type": "armor/weapon/ring/amulet/charm/rune/etc",
    "rarity": "normal/magic/rare/set/unique/rune",
    "identified": true/false,
    "stats": ["list of visible stats"],
    "value_assessment": "trash/keep/valuable/gg",
    "reason": "brief explanation",
    "pickup": true/false,
    "priority": 1-10
}}

Value guide for {build} {character_class}:
- GG (10): Perfect rolled BiS items, high runes (Ber, Jah, etc)
- Valuable (7-9): Good uniques, useful sets, mid runes
- Keep (4-6): Self-use upgrades, decent rares
- Trash (1-3): Vendor or ignore

Be concise. Focus on whether to pick up."""

        response = self._call_gemini(image, prompt)

        # Parse response
        content = response.content

        return ItemAnalysis(
            name=content.get("name", "Unknown"),
            base_type=content.get("base_type", "unknown"),
            rarity=content.get("rarity", "normal"),
            identified=content.get("identified", False),
            stats=content.get("stats", []),
            value_assessment=content.get("value_assessment", "trash"),
            reason=content.get("reason", "Could not analyze"),
            pickup=content.get("pickup", False),
            priority=content.get("priority", 1),
        )

    def analyze_items_batch(
        self,
        screenshot: np.ndarray,
        character_class: str = "sorceress",
        build: str = "blizzard",
    ) -> List[ItemAnalysis]:
        """Analyze all visible items in one call (more efficient).

        Args:
            screenshot: Full game screenshot with items visible
            character_class: Character class
            build: Build type

        Returns:
            List of ItemAnalysis for each visible item
        """
        prompt = f"""Analyze ALL items visible on the ground in this Diablo 2 screenshot.
The player is a {build} {character_class}.

Return JSON array:
[
    {{
        "name": "item name",
        "rarity": "normal/magic/rare/set/unique/rune",
        "value_assessment": "trash/keep/valuable/gg",
        "reason": "brief reason",
        "pickup": true/false,
        "priority": 1-10
    }}
]

Only include items that are clearly visible. Sort by priority (highest first).
If no items visible, return empty array [].

Value guide for {build} {character_class}:
- Always pickup: Uniques, Sets, Runes, high-value bases
- Evaluate: Rares with good bases (diadem, monarch, etc)
- Skip: Normal items, low magic items, gold under 5k"""

        response = self._call_gemini(screenshot, prompt)
        content = response.content

        # Handle array or object response
        items_data = content if isinstance(content, list) else content.get("items", [])

        results = []
        for item in items_data:
            results.append(ItemAnalysis(
                name=item.get("name", "Unknown"),
                base_type=item.get("base_type", "unknown"),
                rarity=item.get("rarity", "normal"),
                identified=item.get("identified", True),
                stats=item.get("stats", []),
                value_assessment=item.get("value_assessment", "trash"),
                reason=item.get("reason", ""),
                pickup=item.get("pickup", False),
                priority=item.get("priority", 1),
            ))

        return results

    def analyze_screen(self, screenshot: np.ndarray) -> ScreenAnalysis:
        """Analyze full game screen for state understanding.

        Args:
            screenshot: Full game screenshot

        Returns:
            ScreenAnalysis with game state
        """
        prompt = """Analyze this Diablo 2 game screenshot.

Return JSON:
{
    "location": "town/act1_wild/catacombs/durance/unknown/etc",
    "health_percent": 0.0-1.0,
    "mana_percent": 0.0-1.0,
    "enemies_visible": count,
    "items_on_ground": [{"name": "...", "rarity": "..."}],
    "portals_visible": true/false,
    "is_dead": true/false,
    "is_in_menu": true/false,
    "suggested_action": "continue/heal/retreat/pickup/unclear"
}

Estimate health/mana from the orbs (red=health, blue=mana).
Be concise."""

        response = self._call_gemini(screenshot, prompt)
        content = response.content

        return ScreenAnalysis(
            location=content.get("location", "unknown"),
            health_percent=float(content.get("health_percent", 1.0)),
            mana_percent=float(content.get("mana_percent", 1.0)),
            enemies_visible=int(content.get("enemies_visible", 0)),
            items_on_ground=content.get("items_on_ground", []),
            portals_visible=content.get("portals_visible", False),
            is_dead=content.get("is_dead", False),
            suggested_action=content.get("suggested_action", "continue"),
        )

    def detect_stuck(
        self,
        screenshot: np.ndarray,
        last_actions: List[str],
        error_context: str = "",
    ) -> Dict[str, Any]:
        """Use vision to diagnose stuck/error state.

        Args:
            screenshot: Current game screenshot
            last_actions: Recent actions taken
            error_context: Description of the problem

        Returns:
            Diagnosis and recovery suggestion
        """
        actions_str = ", ".join(last_actions[-5:]) if last_actions else "none"

        prompt = f"""The bot appears stuck or in an error state.

Recent actions: {actions_str}
Error context: {error_context or "Unknown"}

Analyze this Diablo 2 screenshot and diagnose the problem.

Return JSON:
{{
    "diagnosis": "what's wrong",
    "screen_shows": "description of what's visible",
    "severity": "low/medium/high/critical",
    "recovery_action": "teleport_random/return_town/use_waypoint/wait/exit_game",
    "explanation": "why this action"
}}"""

        response = self._call_gemini(screenshot, prompt, use_cache=False)
        return response.content

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        avg_latency = (
            self.total_latency_ms / self.total_calls
            if self.total_calls > 0 else 0
        )
        cache_rate = (
            self.cache_hits / (self.total_calls + self.cache_hits)
            if (self.total_calls + self.cache_hits) > 0 else 0
        )

        # Estimate cost (Flash pricing)
        estimated_cost = self.total_tokens * 0.000000075  # $0.075 per 1M tokens

        return {
            "total_calls": self.total_calls,
            "cache_hits": self.cache_hits,
            "cache_rate": f"{cache_rate:.1%}",
            "total_tokens": self.total_tokens,
            "avg_latency_ms": f"{avg_latency:.0f}",
            "estimated_cost": f"${estimated_cost:.4f}",
        }


# Global instance
_gemini_client: Optional[GeminiVisionClient] = None


def get_gemini_vision() -> GeminiVisionClient:
    """Get or create the global Gemini vision client."""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = GeminiVisionClient()
    return _gemini_client
