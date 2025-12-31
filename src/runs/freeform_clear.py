"""
Freeform Clear - Route-independent monster killing.

Uses Gemini vision to:
1. Detect enemies on screen
2. Kill them
3. Loot drops
4. Teleport to find more

No predefined route - works in any area. Just start it wherever
you are and it will clear everything it can see.

Best for:
- Chaos Sanctuary
- Worldstone Keep
- Any area you want to clear without a specific path
- Testing/farming random areas
"""

import time
import math
import random
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass

from .base_run import BaseRun, RunConfig, RunStatus
from src.vision.entities import DetectedEnemy
from src.state.session_logger import get_logger

# Lazy import Gemini
_nav_advisor = None
_combat_advisor = None


def _get_nav():
    global _nav_advisor
    if _nav_advisor is None:
        try:
            from src.brain.navigation_advisor import get_navigation_advisor
            _nav_advisor = get_navigation_advisor()
        except:
            pass
    return _nav_advisor


def _get_combat():
    global _combat_advisor
    if _combat_advisor is None:
        try:
            from src.brain.combat_advisor import get_combat_advisor
            _combat_advisor = get_combat_advisor()
        except:
            pass
    return _combat_advisor


@dataclass
class ClearStats:
    """Statistics for the clearing session."""
    monsters_killed: int = 0
    packs_cleared: int = 0
    teleports: int = 0
    items_picked: int = 0
    area_changes: int = 0


class FreeformClearRun(BaseRun):
    """Route-independent monster clearing using vision."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config = RunConfig(
            name="freeform_clear",
            waypoint_act=0,  # Doesn't use waypoint
            waypoint_name="",
            max_deaths=5,
            timeout_seconds=600,  # 10 minute sessions
            pickup_items=True,
            use_static_field=False,
        )

        self.logger = get_logger("freeform_clear")
        self.clear_stats = ClearStats()

        # Search pattern state
        self.search_direction = 0  # Angle in radians
        self.consecutive_empty_screens = 0
        self.last_enemy_positions: List[Tuple[int, int]] = []

    def execute(self) -> RunStatus:
        """Execute freeform clearing - just kill everything visible."""
        self.start()
        self.logger.info("Starting freeform clear - killing everything in sight")

        try:
            # Main loop - find and kill until timeout or stopped
            while not self.is_timed_out():
                # Check vitals
                vitals, should_continue = self.check_vitals()
                if not should_continue:
                    self.logger.info("Vitals critical - returning to town")
                    break

                # Get screenshot
                screenshot = self.get_screenshot()
                if screenshot is None:
                    time.sleep(0.1)
                    continue

                # Try to find and kill enemies
                enemies_found = self._scan_and_engage(screenshot.image)

                if enemies_found:
                    self.consecutive_empty_screens = 0
                else:
                    self.consecutive_empty_screens += 1

                    # If no enemies for a while, explore
                    if self.consecutive_empty_screens >= 3:
                        self._explore_for_enemies()
                        self.consecutive_empty_screens = 0

                time.sleep(0.1)

            # Return to town when done
            self.return_to_town()
            self.complete(success=True)
            self._log_session_stats()

        except Exception as e:
            self.logger.error(f"Freeform clear error: {e}")
            self.abort(f"Exception: {str(e)}")

        return self.status

    def _scan_and_engage(self, image) -> bool:
        """Scan for enemies and engage them.

        Returns:
            True if enemies were found and engaged
        """
        # First try CV detection (fast)
        enemies = self.enemy_detector.detect_by_health_bars(image)

        # If CV finds nothing, try Gemini for better detection
        if not enemies:
            enemies = self._gemini_detect_enemies(image)

        if not enemies:
            return False

        # Found enemies - engage!
        self.logger.debug(f"Found {len(enemies)} enemies")
        killed = self._kill_visible_enemies(enemies, image)

        if killed > 0:
            self.clear_stats.monsters_killed += killed
            self.clear_stats.packs_cleared += 1

            # Loot after killing
            self.pick_up_items(max_items=5)
            self.clear_stats.items_picked += len(self.items_found)

        return True

    def _gemini_detect_enemies(self, image) -> List[DetectedEnemy]:
        """Use Gemini to detect enemies when CV fails."""
        combat = _get_combat()
        if not combat:
            return []

        try:
            # Get situation analysis
            analysis = combat.analyze_situation(image, force=True)

            if not analysis.enemies:
                return []

            # Convert Gemini results to DetectedEnemy format
            # This is approximate since Gemini gives descriptions not coordinates
            # We'll use it to know enemies exist, then use CV or click center
            enemies = []
            window = self.input.window_manager.get_window()

            if window and analysis.enemies:
                # Gemini detected enemies - create approximate positions
                for i, enemy_info in enumerate(analysis.enemies):
                    # Spread enemies across screen center area
                    angle = (i / max(1, len(analysis.enemies))) * 2 * math.pi
                    dist = 150
                    x = window.width // 2 + int(math.cos(angle) * dist)
                    y = window.height // 2 + int(math.sin(angle) * dist)

                    enemies.append(DetectedEnemy(
                        x=x - 20,
                        y=y - 20,
                        width=40,
                        height=40,
                        health_percent=1.0,
                        is_boss=enemy_info.get("threat", "") == "high",
                    ))

            return enemies

        except Exception as e:
            self.logger.debug(f"Gemini detection failed: {e}")
            return []

    def _kill_visible_enemies(self, enemies: List[DetectedEnemy], image) -> int:
        """Kill all visible enemies.

        Returns:
            Number of enemies killed (estimated)
        """
        killed = 0
        pack_timeout = time.time() + 15  # 15 second timeout per pack

        initial_count = len(enemies)

        while enemies and time.time() < pack_timeout:
            # Check vitals
            vitals, should_continue = self.check_vitals()
            if not should_continue:
                break

            # Check for immunities we should skip
            if self._should_skip_pack(enemies):
                self.logger.info("Skipping immune pack")
                self._teleport_away()
                return 0

            # Get target - prefer center of pack for AoE
            target = self._get_best_target(enemies)

            # Position based on build
            self._position_for_combat(target)

            # Attack
            self.combat.build.cast_blizzard(target[0], target[1])
            self.combat.build.cast_glacial_spike(target[0], target[1])

            time.sleep(0.3)

            # Re-scan for remaining enemies
            screenshot = self.get_screenshot()
            if screenshot:
                new_enemies = self.enemy_detector.detect_by_health_bars(screenshot.image)
                if len(new_enemies) < len(enemies):
                    killed += len(enemies) - len(new_enemies)
                enemies = new_enemies

        return killed

    def _should_skip_pack(self, enemies: List[DetectedEnemy]) -> bool:
        """Check if we should skip this pack (immunities)."""
        combat = _get_combat()
        if not combat:
            return False

        # Use cached advice if recent
        if combat.last_advice:
            dominated = combat._get_dangerous_immunities()
            for immunity in dominated:
                if immunity in combat.last_advice.immunities_detected:
                    return True

        return False

    def _get_best_target(self, enemies: List[DetectedEnemy]) -> Tuple[int, int]:
        """Get the best target position (center of pack for AoE)."""
        if len(enemies) == 1:
            return enemies[0].center

        # Calculate center of all enemies
        avg_x = sum(e.center[0] for e in enemies) // len(enemies)
        avg_y = sum(e.center[1] for e in enemies) // len(enemies)

        return (avg_x, avg_y)

    def _position_for_combat(self, target: Tuple[int, int]) -> None:
        """Position character appropriately for build."""
        window = self.input.window_manager.get_window()
        if not window:
            return

        center_x = window.width // 2
        center_y = window.height // 2

        dist = math.sqrt(
            (target[0] - center_x) ** 2 +
            (target[1] - center_y) ** 2
        )

        build = self.brain.config.get("character.build", "blizzard")

        if build == "elemental_druid":
            # Druid wants to be close
            if dist > 200:
                approach_x = center_x + (target[0] - center_x) // 2
                approach_y = center_y + (target[1] - center_y) // 2
                self.movement.teleport_to(approach_x, approach_y)
                time.sleep(0.1)
        else:
            # Sorc wants distance
            if dist < 150:
                away_x = center_x - (target[0] - center_x)
                away_y = center_y - (target[1] - center_y)
                away_x = max(100, min(window.width - 100, away_x))
                away_y = max(100, min(window.height - 100, away_y))
                self.movement.teleport_to(away_x, away_y)
                time.sleep(0.1)

    def _explore_for_enemies(self) -> None:
        """Teleport around to find more enemies."""
        window = self.input.window_manager.get_window()
        if not window:
            return

        # Use spiral/expanding search pattern
        self.search_direction += 0.6  # ~34 degrees per teleport
        distance = 180

        dx = int(math.cos(self.search_direction) * distance)
        dy = int(math.sin(self.search_direction) * distance)

        target_x = window.width // 2 + dx
        target_y = window.height // 2 + dy

        # Clamp to screen
        target_x = max(100, min(window.width - 100, target_x))
        target_y = max(100, min(window.height - 100, target_y))

        self.movement.teleport_to(target_x, target_y)
        self.clear_stats.teleports += 1

        time.sleep(0.15)

    def _teleport_away(self) -> None:
        """Teleport away from current position (skip pack)."""
        window = self.input.window_manager.get_window()
        if not window:
            return

        # Random direction away
        angle = random.uniform(0, 2 * math.pi)
        distance = 250

        target_x = window.width // 2 + int(math.cos(angle) * distance)
        target_y = window.height // 2 + int(math.sin(angle) * distance)

        target_x = max(100, min(window.width - 100, target_x))
        target_y = max(100, min(window.height - 100, target_y))

        self.movement.teleport_to(target_x, target_y)
        self.clear_stats.teleports += 1

    def _log_session_stats(self) -> None:
        """Log session statistics."""
        self.logger.info(
            f"Freeform clear complete: "
            f"{self.clear_stats.monsters_killed} kills, "
            f"{self.clear_stats.packs_cleared} packs, "
            f"{self.clear_stats.teleports} teleports, "
            f"{self.clear_stats.items_picked} items"
        )
