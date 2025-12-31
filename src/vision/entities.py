"""
Entity and item detection - monsters, items on ground, NPCs, portals.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .detector import ColorDetector, TemplateMatcher, MatchResult


class ItemRarity(Enum):
    """Item rarity/quality based on label color."""
    NORMAL = "normal"       # White
    SUPERIOR = "superior"   # White (same as normal visually)
    MAGIC = "magic"         # Blue
    RARE = "rare"           # Yellow
    SET = "set"             # Green
    UNIQUE = "unique"       # Gold/tan
    CRAFTED = "crafted"     # Orange
    RUNEWORD = "runeword"   # Gold (same as unique)
    RUNE = "rune"           # Orange text
    GEM = "gem"             # Varies
    GOLD = "gold"           # Gold pile


@dataclass
class DetectedItem:
    """An item detected on the ground."""
    x: int
    y: int
    width: int
    height: int
    rarity: ItemRarity
    name: Optional[str] = None  # From OCR
    confidence: float = 1.0

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def click_point(self) -> Tuple[int, int]:
        """Point to click to pick up item."""
        return self.center


@dataclass
class DetectedEnemy:
    """A detected enemy/monster."""
    x: int
    y: int
    width: int
    height: int
    name: Optional[str] = None
    is_boss: bool = False
    is_unique: bool = False  # Unique/champion monster

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)


@dataclass
class DetectedPortal:
    """A detected portal (town portal, waypoint, area exit)."""
    x: int
    y: int
    width: int
    height: int
    portal_type: str  # "town_portal", "waypoint", "stairs", "area_exit"

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)


class ItemDetector:
    """Detects items on the ground when Alt is held."""

    # Item label colors in BGR (OpenCV format)
    # These are the background colors of item labels
    RARITY_COLORS = {
        ItemRarity.UNIQUE: {
            "hsv_low": np.array([15, 100, 150]),
            "hsv_high": np.array([30, 255, 255]),
        },
        ItemRarity.SET: {
            "hsv_low": np.array([35, 100, 100]),
            "hsv_high": np.array([85, 255, 255]),
        },
        ItemRarity.RARE: {
            "hsv_low": np.array([20, 150, 150]),
            "hsv_high": np.array([35, 255, 255]),
        },
        ItemRarity.MAGIC: {
            "hsv_low": np.array([100, 100, 100]),
            "hsv_high": np.array([130, 255, 255]),
        },
        ItemRarity.RUNE: {
            "hsv_low": np.array([5, 150, 150]),
            "hsv_high": np.array([20, 255, 255]),
        },
    }

    def __init__(self):
        self.color_detector = ColorDetector()

    def detect_items(
        self,
        screenshot: np.ndarray,
        min_area: int = 200,
        max_area: int = 50000
    ) -> List[DetectedItem]:
        """Detect item labels on screen (when Alt is held).

        Item labels appear as colored rectangles with text.

        Args:
            screenshot: Game screenshot (BGR)
            min_area: Minimum label area in pixels
            max_area: Maximum label area in pixels

        Returns:
            List of detected items
        """
        items = []
        hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)

        # Detect each rarity type
        for rarity, color_range in self.RARITY_COLORS.items():
            mask = cv2.inRange(hsv, color_range["hsv_low"], color_range["hsv_high"])

            # Morphological operations to clean up
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if min_area <= area <= max_area:
                    x, y, w, h = cv2.boundingRect(contour)

                    # Item labels are typically wider than tall
                    aspect_ratio = w / h if h > 0 else 0
                    if 1.5 < aspect_ratio < 20:  # Filter non-label shapes
                        items.append(DetectedItem(
                            x=x,
                            y=y,
                            width=w,
                            height=h,
                            rarity=rarity,
                        ))

        # Remove duplicates/overlapping detections
        items = self._remove_overlapping(items)

        return items

    def detect_gold(self, screenshot: np.ndarray) -> List[DetectedItem]:
        """Detect gold piles on ground.

        Gold has a distinctive yellow/gold color and sparkle effect.
        """
        hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)

        # Gold color range
        mask = cv2.inRange(
            hsv,
            np.array([20, 150, 200]),
            np.array([35, 255, 255])
        )

        # Find bright spots (gold sparkles)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        items = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 5000:  # Gold piles are small
                x, y, w, h = cv2.boundingRect(contour)
                items.append(DetectedItem(
                    x=x, y=y, width=w, height=h,
                    rarity=ItemRarity.GOLD
                ))

        return items

    def _remove_overlapping(
        self,
        items: List[DetectedItem],
        iou_threshold: float = 0.3
    ) -> List[DetectedItem]:
        """Remove overlapping item detections."""
        if not items:
            return []

        # Sort by area (larger first)
        items = sorted(items, key=lambda i: i.width * i.height, reverse=True)

        keep = []
        for item in items:
            overlapping = False
            for kept in keep:
                iou = self._compute_iou(item, kept)
                if iou > iou_threshold:
                    overlapping = True
                    break
            if not overlapping:
                keep.append(item)

        return keep

    def _compute_iou(self, item1: DetectedItem, item2: DetectedItem) -> float:
        """Compute intersection over union."""
        x1 = max(item1.x, item2.x)
        y1 = max(item1.y, item2.y)
        x2 = min(item1.x + item1.width, item2.x + item2.width)
        y2 = min(item1.y + item1.height, item2.y + item2.height)

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = item1.width * item1.height
        area2 = item2.width * item2.height
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0


class EnemyDetector:
    """Detects enemies/monsters on screen."""

    # Enemy health bar colors (red bar above enemies)
    HEALTH_BAR_HSV = {
        "low": np.array([0, 100, 100]),
        "high": np.array([10, 255, 255]),
    }

    def __init__(self, template_matcher: Optional[TemplateMatcher] = None):
        self.template_matcher = template_matcher

    def detect_by_health_bars(
        self,
        screenshot: np.ndarray,
        min_width: int = 30
    ) -> List[DetectedEnemy]:
        """Detect enemies by their health bars.

        Enemy health bars are thin red rectangles above the enemy.
        """
        hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)

        # Find red health bars
        mask = cv2.inRange(
            hsv,
            self.HEALTH_BAR_HSV["low"],
            self.HEALTH_BAR_HSV["high"]
        )

        # Health bars are thin horizontal lines
        kernel = np.ones((1, 10), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        enemies = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Health bars are wide and thin
            if w >= min_width and h < 15 and w / h > 3:
                # Enemy is below the health bar
                enemies.append(DetectedEnemy(
                    x=x,
                    y=y,
                    width=w,
                    height=60,  # Approximate enemy height below bar
                    is_boss=w > 100,  # Boss health bars are wider
                ))

        return enemies

    def detect_mephisto(self, screenshot: np.ndarray) -> Optional[DetectedEnemy]:
        """Detect Mephisto specifically.

        Uses template matching if available, otherwise heuristics.
        """
        if self.template_matcher and "bosses/mephisto" in self.template_matcher.templates:
            match = self.template_matcher.find_by_name(screenshot, "bosses/mephisto")
            if match:
                return DetectedEnemy(
                    x=match.x,
                    y=match.y,
                    width=match.width,
                    height=match.height,
                    name="Mephisto",
                    is_boss=True,
                )

        # Fallback: Look for large health bar in upper portion of screen
        enemies = self.detect_by_health_bars(screenshot, min_width=80)
        for enemy in enemies:
            if enemy.y < screenshot.shape[0] // 2 and enemy.is_boss:
                enemy.name = "Boss"
                return enemy

        return None


class PortalDetector:
    """Detects portals, waypoints, and area transitions."""

    # Town portal - blue/cyan glow
    TOWN_PORTAL_HSV = {
        "low": np.array([85, 100, 100]),
        "high": np.array([110, 255, 255]),
    }

    # Waypoint - yellowish glow when active
    WAYPOINT_HSV = {
        "low": np.array([15, 100, 100]),
        "high": np.array([35, 255, 255]),
    }

    def __init__(self, template_matcher: Optional[TemplateMatcher] = None):
        self.template_matcher = template_matcher

    def detect_town_portal(self, screenshot: np.ndarray) -> Optional[DetectedPortal]:
        """Detect an open town portal."""
        hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(
            hsv,
            self.TOWN_PORTAL_HSV["low"],
            self.TOWN_PORTAL_HSV["high"]
        )

        # Town portals have a significant cyan glow
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 2000:  # Town portal is fairly large
                x, y, w, h = cv2.boundingRect(contour)
                # Portal is roughly circular/oval
                aspect_ratio = w / h if h > 0 else 0
                if 0.5 < aspect_ratio < 2.0:
                    return DetectedPortal(
                        x=x, y=y, width=w, height=h,
                        portal_type="town_portal"
                    )

        return None

    def detect_waypoint(self, screenshot: np.ndarray) -> Optional[DetectedPortal]:
        """Detect a waypoint."""
        # Try template matching first
        if self.template_matcher and "ui/waypoint" in self.template_matcher.templates:
            match = self.template_matcher.find_by_name(screenshot, "ui/waypoint")
            if match:
                return DetectedPortal(
                    x=match.x, y=match.y,
                    width=match.width, height=match.height,
                    portal_type="waypoint"
                )

        # Fallback to color detection
        hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            self.WAYPOINT_HSV["low"],
            self.WAYPOINT_HSV["high"]
        )

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if 1000 < area < 10000:
                x, y, w, h = cv2.boundingRect(contour)
                return DetectedPortal(
                    x=x, y=y, width=w, height=h,
                    portal_type="waypoint"
                )

        return None
