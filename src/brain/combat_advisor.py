"""
Combat Advisor - Gemini-powered strategic combat decisions.

Called periodically (every 1-2 seconds) to provide high-level guidance
while the reactive brain handles frame-by-frame actions.

Use cases:
- Should I engage this pack or skip?
- Am I in danger? What's the threat level?
- Where should I position?
- Is the boss dead? Should I start looting?
- Am I stuck? What should I do?
"""

import time
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum, auto
import numpy as np

from src.config import get_config
from src.state.session_logger import get_logger

# Lazy import
_gemini_client = None


def _get_gemini():
    """Lazy load Gemini client."""
    global _gemini_client
    if _gemini_client is None:
        try:
            from .gemini_vision import get_gemini_vision
            _gemini_client = get_gemini_vision()
        except Exception as e:
            get_logger("combat_advisor").warning(f"Gemini not available: {e}")
            _gemini_client = False
    return _gemini_client if _gemini_client else None


class ThreatLevel(Enum):
    """Current threat assessment."""
    SAFE = auto()       # No enemies, can loot/explore
    LOW = auto()        # Few weak enemies
    MEDIUM = auto()     # Normal pack
    HIGH = auto()       # Dangerous situation
    CRITICAL = auto()   # Need to retreat/chicken


class CombatAction(Enum):
    """Recommended combat action."""
    ENGAGE = auto()         # Attack the enemies
    KITE = auto()           # Attack while moving back
    RETREAT = auto()        # Get distance, don't engage
    SKIP = auto()           # Teleport past, not worth fighting
    HOLD_POSITION = auto()  # Stay and cast
    LOOT = auto()           # Safe to pick up items
    HEAL = auto()           # Use potions, find safety
    CHICKEN = auto()        # Emergency exit


@dataclass
class CombatAdvice:
    """Strategic advice from Gemini."""
    threat_level: ThreatLevel
    recommended_action: CombatAction
    reasoning: str
    enemy_count: int
    boss_present: bool
    immunities_detected: List[str]  # ["cold", "fire", etc.]
    positioning_tip: str
    priority_target: Optional[str]  # "focus the boss", "kill archers first"
    confidence: float  # 0-1
    latency_ms: float


@dataclass
class SituationAnalysis:
    """Full situation analysis."""
    location_type: str  # "open_area", "narrow_corridor", "boss_room"
    health_estimate: float  # 0-1
    mana_estimate: float
    enemies: List[Dict[str, Any]]
    items_on_ground: bool
    portal_visible: bool
    is_town: bool
    suggested_action: str


class CombatAdvisor:
    """Gemini-powered combat decision maker."""

    def __init__(
        self,
        character_class: str = "sorceress",
        build: str = "blizzard",
        min_call_interval: float = 1.0,  # Minimum seconds between calls
    ):
        """Initialize combat advisor.

        Args:
            character_class: Player's class
            build: Build type (affects advice)
            min_call_interval: Rate limiting for API calls
        """
        self.character_class = character_class
        self.build = build
        self.min_call_interval = min_call_interval
        self.logger = get_logger("combat_advisor")

        # Rate limiting
        self.last_call_time = 0

        # Cache last advice for quick access
        self.last_advice: Optional[CombatAdvice] = None
        self.last_analysis: Optional[SituationAnalysis] = None

        # Stats
        self.total_calls = 0
        self.total_latency_ms = 0

    def should_call(self) -> bool:
        """Check if enough time has passed for another call."""
        return time.time() - self.last_call_time >= self.min_call_interval

    def get_combat_advice(
        self,
        screenshot: np.ndarray,
        health_percent: Optional[float] = None,
        mana_percent: Optional[float] = None,
        current_action: str = "exploring",
        force: bool = False,
    ) -> CombatAdvice:
        """Get strategic combat advice from Gemini.

        Args:
            screenshot: Current game screenshot
            health_percent: Known health (if available from CV)
            mana_percent: Known mana (if available from CV)
            current_action: What the bot is currently doing
            force: Bypass rate limiting

        Returns:
            CombatAdvice with recommended action
        """
        # Rate limiting
        if not force and not self.should_call():
            if self.last_advice:
                return self.last_advice
            # Return safe default
            return CombatAdvice(
                threat_level=ThreatLevel.LOW,
                recommended_action=CombatAction.ENGAGE,
                reasoning="Rate limited, using default",
                enemy_count=0,
                boss_present=False,
                immunities_detected=[],
                positioning_tip="Continue current action",
                priority_target=None,
                confidence=0.5,
                latency_ms=0,
            )

        gemini = _get_gemini()
        if not gemini:
            return self._default_advice()

        self.last_call_time = time.time()
        self.total_calls += 1

        # Build context string
        context = f"Currently: {current_action}. "
        if health_percent is not None:
            context += f"Health: {health_percent:.0%}. "
        if mana_percent is not None:
            context += f"Mana: {mana_percent:.0%}. "

        prompt = f"""Analyze this Diablo 2 combat situation for a {self.build} {self.character_class}.

{context}

Return JSON:
{{
    "threat_level": "safe/low/medium/high/critical",
    "recommended_action": "engage/kite/retreat/skip/hold_position/loot/heal/chicken",
    "reasoning": "brief explanation",
    "enemy_count": number,
    "boss_present": true/false,
    "immunities_detected": ["cold", "lightning", etc] or [],
    "positioning_tip": "stay at range / teleport behind / etc",
    "priority_target": "target description or null",
    "confidence": 0.0-1.0
}}

For {self.build} {self.character_class}:
- Cold immunes are dangerous (can't kill them)
- Maintain distance, kite if enemies get close
- Use terrain/moat trick for bosses if possible
- Chicken (emergency exit) if health critically low

Be concise and actionable."""

        start_time = time.time()

        try:
            from PIL import Image
            rgb_image = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)

            response = gemini.model.generate_content(
                [prompt, pil_image],
                generation_config={
                    "temperature": 0.1,
                    "max_output_tokens": 512,
                },
            )

            latency = (time.time() - start_time) * 1000
            self.total_latency_ms += latency

            # Parse response
            content = self._parse_json(response.text)

            advice = CombatAdvice(
                threat_level=self._parse_threat(content.get("threat_level", "low")),
                recommended_action=self._parse_action(content.get("recommended_action", "engage")),
                reasoning=content.get("reasoning", "No reasoning provided"),
                enemy_count=int(content.get("enemy_count", 0)),
                boss_present=content.get("boss_present", False),
                immunities_detected=content.get("immunities_detected", []),
                positioning_tip=content.get("positioning_tip", ""),
                priority_target=content.get("priority_target"),
                confidence=float(content.get("confidence", 0.7)),
                latency_ms=latency,
            )

            self.last_advice = advice
            self.logger.debug(
                f"Combat advice: {advice.recommended_action.name} "
                f"(threat={advice.threat_level.name}, {latency:.0f}ms)"
            )

            return advice

        except Exception as e:
            self.logger.error(f"Combat advice failed: {e}")
            return self._default_advice()

    def analyze_situation(
        self,
        screenshot: np.ndarray,
        force: bool = False,
    ) -> SituationAnalysis:
        """Get detailed situation analysis (more comprehensive than combat advice).

        Args:
            screenshot: Current game screenshot
            force: Bypass rate limiting

        Returns:
            SituationAnalysis with full breakdown
        """
        if not force and not self.should_call():
            if self.last_analysis:
                return self.last_analysis
            return self._default_analysis()

        gemini = _get_gemini()
        if not gemini:
            return self._default_analysis()

        self.last_call_time = time.time()
        self.total_calls += 1

        prompt = """Analyze this Diablo 2 screenshot comprehensively.

Return JSON:
{
    "location_type": "town/open_area/narrow_corridor/boss_room/unknown",
    "health_estimate": 0.0-1.0 (from red orb),
    "mana_estimate": 0.0-1.0 (from blue orb),
    "enemies": [
        {"type": "skeleton/demon/boss/etc", "threat": "low/medium/high", "count": 1}
    ],
    "items_on_ground": true/false,
    "portal_visible": true/false,
    "is_town": true/false,
    "suggested_action": "what should the bot do next"
}

Be accurate about health/mana orb fill levels."""

        try:
            import cv2
            from PIL import Image
            rgb_image = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)

            response = gemini.model.generate_content(
                [prompt, pil_image],
                generation_config={
                    "temperature": 0.1,
                    "max_output_tokens": 512,
                },
            )

            content = self._parse_json(response.text)

            analysis = SituationAnalysis(
                location_type=content.get("location_type", "unknown"),
                health_estimate=float(content.get("health_estimate", 1.0)),
                mana_estimate=float(content.get("mana_estimate", 1.0)),
                enemies=content.get("enemies", []),
                items_on_ground=content.get("items_on_ground", False),
                portal_visible=content.get("portal_visible", False),
                is_town=content.get("is_town", False),
                suggested_action=content.get("suggested_action", "continue"),
            )

            self.last_analysis = analysis
            return analysis

        except Exception as e:
            self.logger.error(f"Situation analysis failed: {e}")
            return self._default_analysis()

    def should_engage(self, screenshot: np.ndarray) -> Tuple[bool, str]:
        """Quick check: should we fight or skip this pack?

        Args:
            screenshot: Current screenshot

        Returns:
            (should_engage, reason)
        """
        advice = self.get_combat_advice(screenshot, current_action="evaluating_pack")

        if advice.recommended_action in (CombatAction.ENGAGE, CombatAction.HOLD_POSITION):
            return True, advice.reasoning
        elif advice.recommended_action == CombatAction.SKIP:
            return False, advice.reasoning
        elif "cold" in advice.immunities_detected:
            return False, "Cold immune enemies detected"
        else:
            return True, "Default: engage"

    def should_retreat(self, screenshot: np.ndarray, health_percent: float) -> Tuple[bool, str]:
        """Quick check: should we retreat?

        Args:
            screenshot: Current screenshot
            health_percent: Current health

        Returns:
            (should_retreat, reason)
        """
        # Immediate check without API
        if health_percent < 0.25:
            return True, "Health critical"

        advice = self.get_combat_advice(
            screenshot,
            health_percent=health_percent,
            current_action="in_combat",
        )

        if advice.recommended_action in (CombatAction.RETREAT, CombatAction.CHICKEN):
            return True, advice.reasoning
        if advice.threat_level == ThreatLevel.CRITICAL:
            return True, "Critical threat level"

        return False, "Safe to continue"

    def is_boss_dead(self, screenshot: np.ndarray) -> Tuple[bool, str]:
        """Check if the boss is dead (safe to loot).

        Args:
            screenshot: Current screenshot

        Returns:
            (is_dead, description)
        """
        advice = self.get_combat_advice(screenshot, current_action="checking_boss", force=True)

        if advice.threat_level == ThreatLevel.SAFE and not advice.boss_present:
            return True, "No boss visible, safe to loot"
        elif advice.boss_present:
            return False, f"Boss still alive, {advice.enemy_count} enemies"
        else:
            return True, advice.reasoning

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Parse JSON from response."""
        import json
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        try:
            return json.loads(text.strip())
        except:
            return {}

    def _parse_threat(self, threat_str: str) -> ThreatLevel:
        """Parse threat level string."""
        mapping = {
            "safe": ThreatLevel.SAFE,
            "low": ThreatLevel.LOW,
            "medium": ThreatLevel.MEDIUM,
            "high": ThreatLevel.HIGH,
            "critical": ThreatLevel.CRITICAL,
        }
        return mapping.get(threat_str.lower(), ThreatLevel.MEDIUM)

    def _parse_action(self, action_str: str) -> CombatAction:
        """Parse action string."""
        mapping = {
            "engage": CombatAction.ENGAGE,
            "kite": CombatAction.KITE,
            "retreat": CombatAction.RETREAT,
            "skip": CombatAction.SKIP,
            "hold_position": CombatAction.HOLD_POSITION,
            "hold": CombatAction.HOLD_POSITION,
            "loot": CombatAction.LOOT,
            "heal": CombatAction.HEAL,
            "chicken": CombatAction.CHICKEN,
        }
        return mapping.get(action_str.lower(), CombatAction.ENGAGE)

    def _default_advice(self) -> CombatAdvice:
        """Return safe default advice."""
        return CombatAdvice(
            threat_level=ThreatLevel.MEDIUM,
            recommended_action=CombatAction.ENGAGE,
            reasoning="Default advice (Gemini unavailable)",
            enemy_count=0,
            boss_present=False,
            immunities_detected=[],
            positioning_tip="Maintain distance",
            priority_target=None,
            confidence=0.3,
            latency_ms=0,
        )

    def _default_analysis(self) -> SituationAnalysis:
        """Return safe default analysis."""
        return SituationAnalysis(
            location_type="unknown",
            health_estimate=1.0,
            mana_estimate=1.0,
            enemies=[],
            items_on_ground=False,
            portal_visible=False,
            is_town=False,
            suggested_action="continue",
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get advisor statistics."""
        avg_latency = (
            self.total_latency_ms / self.total_calls
            if self.total_calls > 0 else 0
        )
        return {
            "total_calls": self.total_calls,
            "avg_latency_ms": f"{avg_latency:.0f}",
            "last_threat": self.last_advice.threat_level.name if self.last_advice else "N/A",
            "last_action": self.last_advice.recommended_action.name if self.last_advice else "N/A",
        }


# Need cv2 for image conversion
import cv2


# Global instance
_combat_advisor: Optional[CombatAdvisor] = None


def get_combat_advisor() -> CombatAdvisor:
    """Get or create global combat advisor."""
    global _combat_advisor
    if _combat_advisor is None:
        config = get_config()
        _combat_advisor = CombatAdvisor(
            character_class=config.get("character.class", "sorceress"),
            build=config.get("character.build", "blizzard"),
        )
    return _combat_advisor
