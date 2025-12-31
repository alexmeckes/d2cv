"""
Navigation Advisor - Gemini-powered pathfinding and exploration.

Uses visual understanding to:
- Find stairs, portals, waypoints
- Navigate random map layouts
- Detect if stuck or going in circles
- Understand minimap for direction
"""

import time
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum, auto
import numpy as np
import cv2

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
            get_logger("navigation_advisor").warning(f"Gemini not available: {e}")
            _gemini_client = False
    return _gemini_client if _gemini_client else None


class NavigationTarget(Enum):
    """What we're looking for."""
    STAIRS_DOWN = auto()    # Next level
    STAIRS_UP = auto()      # Previous level
    WAYPOINT = auto()       # Waypoint
    PORTAL = auto()         # Town portal or area portal
    BOSS = auto()           # Boss monster
    EXIT = auto()           # Area exit
    NPC = auto()            # Specific NPC
    ITEM = auto()           # Item on ground
    UNKNOWN = auto()        # General exploration


class Direction(Enum):
    """Cardinal directions for teleporting."""
    NORTH = auto()
    SOUTH = auto()
    EAST = auto()
    WEST = auto()
    NORTHEAST = auto()
    NORTHWEST = auto()
    SOUTHEAST = auto()
    SOUTHWEST = auto()
    HERE = auto()  # Target is on screen
    UNKNOWN = auto()


@dataclass
class NavigationAdvice:
    """Navigation guidance from Gemini."""
    target_found: bool
    target_location: Optional[Tuple[int, int]]  # Screen coordinates if visible
    suggested_direction: Direction
    reasoning: str
    confidence: float
    landmark_description: str  # What Gemini sees
    is_stuck: bool
    alternate_suggestion: Optional[str]
    latency_ms: float


@dataclass
class MapAnalysis:
    """Analysis of the current map/area."""
    area_name: str  # "Durance Level 2", "Lost City", etc.
    area_type: str  # "dungeon", "outdoor", "town"
    explored_directions: List[str]  # Directions that look explored on minimap
    unexplored_directions: List[str]  # Directions that look unexplored
    visible_landmarks: List[Dict[str, Any]]  # Stairs, waypoints, etc.
    suggested_exploration_dir: Direction
    minimap_analysis: str  # Description of minimap state


class NavigationAdvisor:
    """Gemini-powered navigation and pathfinding."""

    def __init__(
        self,
        min_call_interval: float = 0.5,  # Can call more frequently for navigation
    ):
        """Initialize navigation advisor.

        Args:
            min_call_interval: Minimum seconds between calls
        """
        self.min_call_interval = min_call_interval
        self.logger = get_logger("navigation_advisor")

        # Rate limiting
        self.last_call_time = 0

        # Position tracking (for stuck detection)
        self.position_history: List[Tuple[int, int]] = []
        self.max_history = 20

        # Cache
        self.last_advice: Optional[NavigationAdvice] = None
        self.last_analysis: Optional[MapAnalysis] = None

        # Stats
        self.total_calls = 0
        self.targets_found = 0

    def find_target(
        self,
        screenshot: np.ndarray,
        target: NavigationTarget,
        target_name: Optional[str] = None,
    ) -> NavigationAdvice:
        """Find a navigation target on screen or get direction to it.

        Args:
            screenshot: Current game screenshot
            target: What we're looking for
            target_name: Specific name (e.g., "Durance Level 3" for stairs)

        Returns:
            NavigationAdvice with location or direction
        """
        gemini = _get_gemini()
        if not gemini:
            return self._default_advice()

        self.last_call_time = time.time()
        self.total_calls += 1

        target_desc = target_name or target.name.lower().replace("_", " ")

        prompt = f"""Find the {target_desc} in this Diablo 2 screenshot.

Return JSON:
{{
    "target_found": true/false,
    "target_x": pixel X coordinate if visible (0 = left edge),
    "target_y": pixel Y coordinate if visible (0 = top edge),
    "suggested_direction": "north/south/east/west/northeast/northwest/southeast/southwest/here/unknown",
    "reasoning": "why you suggest this direction",
    "confidence": 0.0-1.0,
    "landmark_description": "what you see that helps identify location",
    "is_stuck": true/false (does it look like we're going in circles?),
    "alternate_suggestion": "backup plan if primary doesn't work"
}}

Target: {target_desc}

Tips for Diablo 2 navigation:
- Stairs/portals often have a distinctive glow or color
- Minimap (top right) shows explored areas in lighter color
- Waypoints have a blue/teal glow
- Town portals are orange/red
- Boss rooms often have unique architecture

If target is visible, provide exact pixel coordinates.
If not visible, suggest which direction to teleport based on map layout."""

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
            content = self._parse_json(response.text)

            target_found = content.get("target_found", False)
            target_location = None

            if target_found and "target_x" in content and "target_y" in content:
                target_location = (
                    int(content.get("target_x", 0)),
                    int(content.get("target_y", 0))
                )
                self.targets_found += 1

            advice = NavigationAdvice(
                target_found=target_found,
                target_location=target_location,
                suggested_direction=self._parse_direction(content.get("suggested_direction", "unknown")),
                reasoning=content.get("reasoning", ""),
                confidence=float(content.get("confidence", 0.5)),
                landmark_description=content.get("landmark_description", ""),
                is_stuck=content.get("is_stuck", False),
                alternate_suggestion=content.get("alternate_suggestion"),
                latency_ms=latency,
            )

            self.last_advice = advice

            if target_found:
                self.logger.info(f"Found {target_desc} at {target_location}")
            else:
                self.logger.debug(f"Navigate {advice.suggested_direction.name} to find {target_desc}")

            return advice

        except Exception as e:
            self.logger.error(f"Navigation failed: {e}")
            return self._default_advice()

    def find_stairs(self, screenshot: np.ndarray, level_name: str = "") -> NavigationAdvice:
        """Find stairs to next level.

        Args:
            screenshot: Current game screenshot
            level_name: Target level name (e.g., "Durance Level 3")

        Returns:
            NavigationAdvice
        """
        target_name = f"stairs to {level_name}" if level_name else "stairs to next level"
        return self.find_target(screenshot, NavigationTarget.STAIRS_DOWN, target_name)

    def find_waypoint(self, screenshot: np.ndarray) -> NavigationAdvice:
        """Find waypoint in current area."""
        return self.find_target(screenshot, NavigationTarget.WAYPOINT, "waypoint")

    def find_portal(self, screenshot: np.ndarray, portal_type: str = "town") -> NavigationAdvice:
        """Find a portal.

        Args:
            screenshot: Current game screenshot
            portal_type: "town" for town portal, "red" for area portal

        Returns:
            NavigationAdvice
        """
        return self.find_target(screenshot, NavigationTarget.PORTAL, f"{portal_type} portal")

    def find_boss(self, screenshot: np.ndarray, boss_name: str) -> NavigationAdvice:
        """Find a specific boss.

        Args:
            screenshot: Current game screenshot
            boss_name: Boss name (e.g., "Mephisto", "Andariel")

        Returns:
            NavigationAdvice
        """
        return self.find_target(screenshot, NavigationTarget.BOSS, boss_name)

    def analyze_map(self, screenshot: np.ndarray) -> MapAnalysis:
        """Get comprehensive map analysis for exploration.

        Args:
            screenshot: Current game screenshot

        Returns:
            MapAnalysis with exploration suggestions
        """
        gemini = _get_gemini()
        if not gemini:
            return self._default_analysis()

        self.total_calls += 1

        prompt = """Analyze the map/minimap in this Diablo 2 screenshot for navigation.

Return JSON:
{
    "area_name": "best guess of area name (Durance, Catacombs, etc.)",
    "area_type": "dungeon/outdoor/town",
    "explored_directions": ["directions that appear already explored on minimap"],
    "unexplored_directions": ["directions that appear unexplored"],
    "visible_landmarks": [
        {"type": "stairs/waypoint/portal/boss", "direction": "north/south/etc", "distance": "close/medium/far"}
    ],
    "suggested_exploration_dir": "north/south/east/west/etc",
    "minimap_analysis": "description of what the minimap shows"
}

The minimap is in the top-right corner. Lighter areas = explored, darker = unexplored.
Suggest exploring unexplored areas to find objectives."""

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

            content = self._parse_json(response.text)

            analysis = MapAnalysis(
                area_name=content.get("area_name", "Unknown"),
                area_type=content.get("area_type", "dungeon"),
                explored_directions=content.get("explored_directions", []),
                unexplored_directions=content.get("unexplored_directions", []),
                visible_landmarks=content.get("visible_landmarks", []),
                suggested_exploration_dir=self._parse_direction(
                    content.get("suggested_exploration_dir", "unknown")
                ),
                minimap_analysis=content.get("minimap_analysis", ""),
            )

            self.last_analysis = analysis
            return analysis

        except Exception as e:
            self.logger.error(f"Map analysis failed: {e}")
            return self._default_analysis()

    def get_teleport_direction(
        self,
        screenshot: np.ndarray,
        target: NavigationTarget,
        screen_width: int = 1280,
        screen_height: int = 720,
    ) -> Tuple[int, int]:
        """Get screen coordinates to teleport towards target.

        Args:
            screenshot: Current game screenshot
            target: What we're navigating to
            screen_width: Game window width
            screen_height: Game window height

        Returns:
            (x, y) screen coordinates to teleport to
        """
        advice = self.find_target(screenshot, target)

        # If target is on screen, teleport to it
        if advice.target_found and advice.target_location:
            return advice.target_location

        # Otherwise, teleport in suggested direction
        center_x = screen_width // 2
        center_y = screen_height // 2
        offset = 200  # Teleport distance

        direction_offsets = {
            Direction.NORTH: (0, -offset),
            Direction.SOUTH: (0, offset),
            Direction.EAST: (offset, 0),
            Direction.WEST: (-offset, 0),
            Direction.NORTHEAST: (offset, -offset),
            Direction.NORTHWEST: (-offset, -offset),
            Direction.SOUTHEAST: (offset, offset),
            Direction.SOUTHWEST: (-offset, offset),
            Direction.HERE: (0, 0),
            Direction.UNKNOWN: (offset, 0),  # Default: go east
        }

        dx, dy = direction_offsets.get(advice.suggested_direction, (0, 0))
        return (center_x + dx, center_y + dy)

    def update_position(self, x: int, y: int) -> bool:
        """Update position history for stuck detection.

        Args:
            x: Current X position (from minimap or estimation)
            y: Current Y position

        Returns:
            True if appears stuck (same area repeatedly)
        """
        self.position_history.append((x, y))

        # Keep history limited
        if len(self.position_history) > self.max_history:
            self.position_history.pop(0)

        # Check for repeated positions (stuck)
        if len(self.position_history) >= 10:
            recent = self.position_history[-10:]
            unique_positions = len(set(recent))
            if unique_positions <= 3:
                self.logger.warning("Stuck detected - revisiting same positions")
                return True

        return False

    def suggest_unstuck_direction(self, screenshot: np.ndarray) -> Direction:
        """When stuck, suggest a new direction to try.

        Args:
            screenshot: Current game screenshot

        Returns:
            Direction to try
        """
        analysis = self.analyze_map(screenshot)

        # Try unexplored directions first
        if analysis.unexplored_directions:
            dir_str = analysis.unexplored_directions[0]
            return self._parse_direction(dir_str)

        # Random direction as fallback
        import random
        directions = [Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST]
        return random.choice(directions)

    def _parse_direction(self, dir_str: str) -> Direction:
        """Parse direction string."""
        mapping = {
            "north": Direction.NORTH,
            "south": Direction.SOUTH,
            "east": Direction.EAST,
            "west": Direction.WEST,
            "northeast": Direction.NORTHEAST,
            "northwest": Direction.NORTHWEST,
            "southeast": Direction.SOUTHEAST,
            "southwest": Direction.SOUTHWEST,
            "here": Direction.HERE,
            "unknown": Direction.UNKNOWN,
            "n": Direction.NORTH,
            "s": Direction.SOUTH,
            "e": Direction.EAST,
            "w": Direction.WEST,
        }
        return mapping.get(dir_str.lower(), Direction.UNKNOWN)

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

    def _default_advice(self) -> NavigationAdvice:
        """Return default advice when Gemini unavailable."""
        return NavigationAdvice(
            target_found=False,
            target_location=None,
            suggested_direction=Direction.UNKNOWN,
            reasoning="Gemini unavailable",
            confidence=0.0,
            landmark_description="",
            is_stuck=False,
            alternate_suggestion="Try random direction",
            latency_ms=0,
        )

    def _default_analysis(self) -> MapAnalysis:
        """Return default analysis."""
        return MapAnalysis(
            area_name="Unknown",
            area_type="dungeon",
            explored_directions=[],
            unexplored_directions=["north", "east", "south", "west"],
            visible_landmarks=[],
            suggested_exploration_dir=Direction.EAST,
            minimap_analysis="Unable to analyze",
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get advisor statistics."""
        return {
            "total_calls": self.total_calls,
            "targets_found": self.targets_found,
            "find_rate": f"{self.targets_found / max(1, self.total_calls):.1%}",
            "position_history_size": len(self.position_history),
        }


# Global instance
_navigation_advisor: Optional[NavigationAdvisor] = None


def get_navigation_advisor() -> NavigationAdvisor:
    """Get or create global navigation advisor."""
    global _navigation_advisor
    if _navigation_advisor is None:
        _navigation_advisor = NavigationAdvisor()
    return _navigation_advisor
