"""
Game state detection - health, mana, belt, skills, UI elements.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

from .detector import ColorDetector


class GameLocation(Enum):
    """Detected game location/state."""
    UNKNOWN = "unknown"
    LOADING = "loading"
    MAIN_MENU = "main_menu"
    CHARACTER_SELECT = "character_select"
    IN_GAME_TOWN = "in_game_town"
    IN_GAME_FIELD = "in_game_field"
    DEAD = "dead"
    INVENTORY_OPEN = "inventory_open"
    STASH_OPEN = "stash_open"
    WAYPOINT_OPEN = "waypoint_open"
    NPC_DIALOG = "npc_dialog"


@dataclass
class VitalsState:
    """Current health and mana state."""
    health_percent: float
    mana_percent: float
    health_full: bool
    mana_full: bool
    is_poisoned: bool = False
    merc_health_percent: Optional[float] = None


@dataclass
class BeltState:
    """State of potion belt."""
    slot_1_count: int  # Usually 4 potions per column
    slot_2_count: int
    slot_3_count: int
    slot_4_count: int

    @property
    def total_health_potions(self) -> int:
        # Assuming slots 1-2 are health
        return self.slot_1_count + self.slot_2_count

    @property
    def total_mana_potions(self) -> int:
        # Assuming slots 3-4 are mana
        return self.slot_3_count + self.slot_4_count


@dataclass
class ScreenRegions:
    """Screen regions for a given resolution.

    All coordinates are (x, y, width, height) relative to game window.
    """
    # Resolution this config is for
    width: int
    height: int

    # Health/Mana orbs (the circular orbs at bottom corners)
    health_orb: Tuple[int, int, int, int]
    mana_orb: Tuple[int, int, int, int]

    # Belt (bottom center)
    belt: Tuple[int, int, int, int]
    belt_slots: List[Tuple[int, int, int, int]]  # Individual slot regions

    # Skill bar
    left_skill: Tuple[int, int, int, int]
    right_skill: Tuple[int, int, int, int]

    # Minimap (top right)
    minimap: Tuple[int, int, int, int]

    # Inventory (when open, right side)
    inventory: Tuple[int, int, int, int]

    # Experience bar (very bottom)
    exp_bar: Tuple[int, int, int, int]

    # Merc portrait (top left, below player)
    merc_portrait: Optional[Tuple[int, int, int, int]] = None


# Predefined regions for common resolutions
REGIONS_1280x720 = ScreenRegions(
    width=1280,
    height=720,

    # Health orb - bottom left corner
    health_orb=(30, 570, 100, 100),

    # Mana orb - bottom right corner
    mana_orb=(1150, 570, 100, 100),

    # Belt - bottom center
    belt=(506, 650, 268, 50),
    belt_slots=[
        (506, 650, 67, 50),   # Slot 1
        (573, 650, 67, 50),   # Slot 2
        (640, 650, 67, 50),   # Slot 3
        (707, 650, 67, 50),   # Slot 4
    ],

    # Skill icons
    left_skill=(200, 650, 48, 48),
    right_skill=(1032, 650, 48, 48),

    # Minimap - top right
    minimap=(1050, 10, 220, 180),

    # Inventory - right side when open
    inventory=(880, 340, 390, 340),

    # Exp bar - very bottom
    exp_bar=(190, 708, 900, 12),

    # Merc portrait
    merc_portrait=(10, 100, 80, 80),
)


class GameStateDetector:
    """Detects current game state from screenshots."""

    # HSV ranges for health orb (red tones)
    HEALTH_HSV_RANGE = {
        "hue": (0, 15),       # Red hue (wraps around, also check 165-180)
        "sat": (100, 255),
        "val": (80, 255),
    }

    # HSV ranges for mana orb (blue tones)
    MANA_HSV_RANGE = {
        "hue": (100, 130),    # Blue hue
        "sat": (100, 255),
        "val": (80, 255),
    }

    # Poison green overlay
    POISON_HSV_RANGE = {
        "hue": (35, 85),
        "sat": (100, 255),
        "val": (50, 255),
    }

    def __init__(self, regions: Optional[ScreenRegions] = None):
        self.regions = regions or REGIONS_1280x720

    def get_vitals(self, screenshot: np.ndarray) -> VitalsState:
        """Detect current health and mana levels.

        Args:
            screenshot: Full game screenshot (BGR)

        Returns:
            VitalsState with health/mana percentages
        """
        # Extract orb regions
        hx, hy, hw, hh = self.regions.health_orb
        mx, my, mw, mh = self.regions.mana_orb

        health_region = screenshot[hy:hy+hh, hx:hx+hw]
        mana_region = screenshot[my:my+mh, mx:mx+mw]

        # Detect health (red pixels)
        health_pct = self._detect_orb_fill(health_region, is_health=True)

        # Detect mana (blue pixels)
        mana_pct = self._detect_orb_fill(mana_region, is_health=False)

        # Check for poison (green tint in health orb)
        is_poisoned = self._detect_poison(health_region)

        # Detect merc health if portrait visible
        merc_health = None
        if self.regions.merc_portrait:
            merc_health = self._detect_merc_health(screenshot)

        return VitalsState(
            health_percent=health_pct,
            mana_percent=mana_pct,
            health_full=health_pct > 0.95,
            mana_full=mana_pct > 0.95,
            is_poisoned=is_poisoned,
            merc_health_percent=merc_health,
        )

    def _detect_orb_fill(self, orb_region: np.ndarray, is_health: bool) -> float:
        """Detect how full an orb is based on colored pixel ratio.

        The orb fills from bottom to top, so we check vertical distribution.
        """
        hsv = cv2.cvtColor(orb_region, cv2.COLOR_BGR2HSV)

        if is_health:
            # Red wraps around in HSV, check both ends
            mask1 = cv2.inRange(
                hsv,
                np.array([0, 100, 80]),
                np.array([15, 255, 255])
            )
            mask2 = cv2.inRange(
                hsv,
                np.array([165, 100, 80]),
                np.array([180, 255, 255])
            )
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            # Blue/purple for mana
            mask = cv2.inRange(
                hsv,
                np.array([100, 80, 60]),
                np.array([140, 255, 255])
            )

        # Create circular mask (orb is roughly circular)
        h, w = orb_region.shape[:2]
        center = (w // 2, h // 2)
        radius = min(w, h) // 2 - 5
        circle_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(circle_mask, center, radius, 255, -1)

        # Apply circular mask
        masked = cv2.bitwise_and(mask, circle_mask)

        # Count filled pixels vs total orb pixels
        filled = np.count_nonzero(masked)
        total = np.count_nonzero(circle_mask)

        if total == 0:
            return 0.0

        # The ratio won't be exactly 0-100% due to orb decoration
        # Calibrate: empty orb ~5%, full orb ~60% colored
        raw_ratio = filled / total
        calibrated = (raw_ratio - 0.05) / 0.55
        return max(0.0, min(1.0, calibrated))

    def _detect_poison(self, health_region: np.ndarray) -> bool:
        """Detect if health orb has green poison tint."""
        hsv = cv2.cvtColor(health_region, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([35, 100, 50]),
            np.array([85, 255, 255])
        )
        ratio = np.count_nonzero(mask) / mask.size
        return ratio > 0.1  # More than 10% green = poisoned

    def _detect_merc_health(self, screenshot: np.ndarray) -> Optional[float]:
        """Detect mercenary health from portrait."""
        if not self.regions.merc_portrait:
            return None

        px, py, pw, ph = self.regions.merc_portrait
        portrait = screenshot[py:py+ph, px:px+pw]

        # Merc health bar is typically a thin red bar
        # This is approximate - needs calibration
        hsv = cv2.cvtColor(portrait, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([0, 100, 100]),
            np.array([10, 255, 255])
        )
        ratio = np.count_nonzero(mask) / mask.size

        # Very rough estimate
        return min(1.0, ratio * 10)

    def get_belt_state(self, screenshot: np.ndarray) -> BeltState:
        """Detect potion counts in belt slots.

        This is approximate - counts colored pixels in each slot.
        """
        counts = []

        for slot_region in self.regions.belt_slots:
            sx, sy, sw, sh = slot_region
            slot_img = screenshot[sy:sy+sh, sx:sx+sw]

            # Potions have distinct colors - red, blue, purple
            hsv = cv2.cvtColor(slot_img, cv2.COLOR_BGR2HSV)

            # Check for any potion colors
            red_mask = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([15, 255, 255]))
            blue_mask = cv2.inRange(hsv, np.array([100, 80, 80]), np.array([130, 255, 255]))
            purple_mask = cv2.inRange(hsv, np.array([130, 80, 80]), np.array([160, 255, 255]))

            combined = cv2.bitwise_or(red_mask, cv2.bitwise_or(blue_mask, purple_mask))
            ratio = np.count_nonzero(combined) / combined.size

            # Estimate count (4 potions max per column visible)
            if ratio < 0.05:
                count = 0
            elif ratio < 0.15:
                count = 1
            elif ratio < 0.3:
                count = 2
            elif ratio < 0.45:
                count = 3
            else:
                count = 4

            counts.append(count)

        return BeltState(
            slot_1_count=counts[0],
            slot_2_count=counts[1],
            slot_3_count=counts[2],
            slot_4_count=counts[3],
        )

    def detect_location(self, screenshot: np.ndarray) -> GameLocation:
        """Detect current game location/state.

        Uses various heuristics to determine where the player is.
        """
        h, w = screenshot.shape[:2]

        # Check for loading screen (mostly black)
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)

        if mean_brightness < 20:
            return GameLocation.LOADING

        # Check if orbs are visible (in-game indicator)
        vitals = self.get_vitals(screenshot)

        # If we can detect health/mana, we're in game
        if vitals.health_percent > 0 or vitals.mana_percent > 0:
            # TODO: Distinguish town vs field, check for open UIs
            return GameLocation.IN_GAME_FIELD

        # Check for death screen (specific red tint + text)
        # TODO: Implement death detection

        return GameLocation.UNKNOWN

    def is_inventory_open(self, screenshot: np.ndarray) -> bool:
        """Check if inventory panel is open."""
        ix, iy, iw, ih = self.regions.inventory
        inv_region = screenshot[iy:iy+ih, ix:ix+iw]

        # Inventory has a distinct brown/tan background
        hsv = cv2.cvtColor(inv_region, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([10, 50, 50]),
            np.array([30, 200, 200])
        )
        ratio = np.count_nonzero(mask) / mask.size

        return ratio > 0.3  # Significant brown = inventory open
