"""
Secret Cow Level farming run - Act 1 Hell.

Strategy:
1. Open cow portal (Wirt's Leg + Tome of Town Portal in Cube)
   OR enter existing portal in Rogue Encampment
2. Enter the red portal to Cow Level
3. Clear Cow Level (very high monster density)
4. Optional: Find and kill the Cow King
5. Loot drops (high density = lots of drops)
6. Town portal back

Note: Cow Level is excellent for:
- High density farming (many cows per screen)
- No cold immunes (good for Blizz Sorc)
- No fire immunes (good for Fire Druid)
- Cow King can drop unique items
"""

import time
import math
from typing import Optional, Tuple, List

from .base_run import BaseRun, RunConfig, RunStatus
from src.vision.entities import DetectedEnemy


class CowLevelRun(BaseRun):
    """Secret Cow Level farming run implementation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config = RunConfig(
            name="cow_level",
            waypoint_act=1,
            waypoint_name="Rogue Encampment",
            max_deaths=3,
            timeout_seconds=300,  # 5 minute timeout
            pickup_items=True,
            use_static_field=False,  # Regular monsters
        )

        # Cow Level-specific state
        self.portal_entered = False
        self.monsters_killed = 0
        self.cow_king_killed = False

    def execute(self) -> RunStatus:
        """Execute the full Cow Level run."""
        self.start()

        try:
            # Step 1: Find and enter the Cow Level portal
            # Assume portal is already open in town (common setup)
            if not self._enter_cow_portal():
                self.abort("Failed to enter Cow Level portal")
                return self.status

            # Step 2: Clear Cow Level
            cleared = self._clear_cows()

            # Step 3: Optional - Find Cow King
            # self._find_cow_king()  # Uncomment if you want to hunt the King

            # Step 4: Return to town
            if not self.return_to_town():
                self.abort("Failed to return to town")
                return self.status

            self.complete(success=True)

        except Exception as e:
            self.abort(f"Exception: {str(e)}")

        return self.status

    def _enter_cow_portal(self) -> bool:
        """Find and enter the Cow Level portal in town.

        The portal should already be open (created via Cube recipe beforehand).
        Looks for the distinctive red portal near the Rogue Encampment waypoint.
        """

        def detect_red_portal(image) -> Optional[Tuple[int, int]]:
            """Detect the red Cow Level portal."""
            import cv2
            import numpy as np

            # Cow portal is a distinctive red color
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Red color in HSV (red wraps around 0/180)
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 100, 100])
            upper_red2 = np.array([180, 255, 255])

            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)

            # Find contours
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                area = cv2.contourArea(contour)
                if 1000 < area < 15000:  # Portal-sized area
                    x, y, w, h = cv2.boundingRect(contour)
                    # Portal is taller than wide
                    aspect_ratio = h / max(w, 1)
                    if aspect_ratio > 1.2:
                        if y > 100 and y < image.shape[0] - 100:
                            return (x + w // 2, y + h // 2)

            return None

        # Search town for the cow portal
        for search_count in range(20):
            screenshot = self.get_screenshot()
            if screenshot:
                portal_pos = detect_red_portal(screenshot.image)
                if portal_pos:
                    # Found portal! Click to enter
                    self.input.click(portal_pos[0], portal_pos[1], relative=True)
                    time.sleep(2)  # Loading screen
                    self.portal_entered = True
                    return True

            # Walk around town to find the portal
            window = self.input.window_manager.get_window()
            if window:
                # Search pattern in town
                angle = search_count * 0.8
                distance = 100
                dx = int(math.cos(angle) * distance)
                dy = int(math.sin(angle) * distance)

                target_x = window.width // 2 + dx
                target_y = window.height // 2 + dy

                # Use teleport if available, otherwise click to walk
                self.movement.teleport_to(target_x, target_y)

            time.sleep(0.3)

        return False

    def _clear_cows(self) -> int:
        """Clear Cow Level of monsters.

        Cow Level has very high density - cows come in large packs.
        """
        monsters_killed = 0
        clear_teleports = 0
        max_clear_teleports = 100  # More teleports for high density area

        while clear_teleports < max_clear_teleports:
            # Check vitals
            _, should_continue = self.check_vitals()
            if not should_continue:
                break

            if self.is_timed_out():
                break

            screenshot = self.get_screenshot()
            if screenshot is None:
                continue

            # Look for monsters (cows have health bars)
            enemies = self.enemy_detector.detect_by_health_bars(screenshot.image)

            if enemies:
                # Kill visible pack
                killed = self._kill_cow_pack(enemies, screenshot.image)
                monsters_killed += killed

                # Loot after killing
                if killed > 0:
                    self.pick_up_items(max_items=5)
            else:
                # No monsters visible, teleport to next area
                window = self.input.window_manager.get_window()
                if window:
                    # Use expanding search pattern
                    angle = clear_teleports * 0.35
                    distance = 130 + (clear_teleports // 8) * 25
                    dx = int(math.cos(angle) * distance)
                    dy = int(math.sin(angle) * distance)

                    target_x = window.width // 2 + dx
                    target_y = window.height // 2 + dy
                    self.movement.teleport_to(target_x, target_y)

                clear_teleports += 1
                time.sleep(0.15)

        self.monsters_killed = monsters_killed
        return monsters_killed

    def _kill_cow_pack(self, enemies: List[DetectedEnemy], image) -> int:
        """Kill a pack of cows.

        Cows come in large groups - use AoE effectively.
        """
        killed = 0
        pack_timeout = time.time() + 12  # 12 seconds per pack max (larger packs)

        while enemies and time.time() < pack_timeout:
            # Check vitals
            vitals, should_continue = self.check_vitals()
            if not should_continue:
                break

            # Find the center of the pack for AoE
            if len(enemies) > 1:
                avg_x = sum(e.center[0] for e in enemies) // len(enemies)
                avg_y = sum(e.center[1] for e in enemies) // len(enemies)
                target_x, target_y = avg_x, avg_y
            else:
                target_x, target_y = enemies[0].center

            window = self.input.window_manager.get_window()
            if window:
                center_x = window.width // 2
                center_y = window.height // 2

                # Check distance to pack center
                dist = math.sqrt(
                    (target_x - center_x) ** 2 +
                    (target_y - center_y) ** 2
                )

                # Position management depends on build
                build_type = self.brain.config.get("character.build", "blizzard")

                if build_type == "elemental_druid":
                    # Druid wants to be CLOSER for Armageddon
                    if dist > 250:
                        # Teleport closer
                        approach_x = center_x + (target_x - center_x) // 2
                        approach_y = center_y + (target_y - center_y) // 2
                        self.movement.teleport_to(approach_x, approach_y)
                        time.sleep(0.1)
                        continue
                else:
                    # Sorc wants to keep distance
                    if dist < 150:
                        away_x = center_x - (target_x - center_x)
                        away_y = center_y - (target_y - center_y)
                        away_x = max(100, min(window.width - 100, away_x))
                        away_y = max(100, min(window.height - 100, away_y))
                        self.movement.teleport_to(away_x, away_y)
                        time.sleep(0.1)
                        continue

            # Cast AoE on pack center (build.cast_blizzard works for both builds)
            self.combat.build.cast_blizzard(target_x, target_y)

            # Follow up
            self.combat.build.cast_glacial_spike(target_x, target_y)

            time.sleep(0.3)

            # Re-detect enemies
            screenshot = self.get_screenshot()
            if screenshot:
                new_enemies = self.enemy_detector.detect_by_health_bars(screenshot.image)
                if len(new_enemies) < len(enemies):
                    killed += len(enemies) - len(new_enemies)
                enemies = new_enemies

        return killed

    def _find_cow_king(self) -> bool:
        """Find and kill the Cow King.

        The Cow King is a unique monster (super unique).
        He has a distinctive larger size and unique name plate.
        """

        def detect_cow_king(image) -> Optional[Tuple[int, int]]:
            """Detect the Cow King by unique name plate or larger health bar."""
            # Cow King has a golden/unique name plate
            # This is a simplified detection - in practice you'd use
            # template matching or Gemini vision
            import cv2
            import numpy as np

            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Look for gold name plate (unique monsters)
            lower_gold = np.array([15, 100, 150])
            upper_gold = np.array([35, 255, 255])
            mask = cv2.inRange(hsv, lower_gold, upper_gold)

            # Find contours
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                area = cv2.contourArea(contour)
                if 500 < area < 5000:
                    x, y, w, h = cv2.boundingRect(contour)
                    # Name plates are wide and short
                    aspect_ratio = w / max(h, 1)
                    if aspect_ratio > 3:
                        # This looks like a unique name plate
                        return (x + w // 2, y + 50)  # Aim below name

            return None

        # Search for Cow King
        for search_count in range(30):
            _, should_continue = self.check_vitals()
            if not should_continue:
                return False

            if self.is_timed_out():
                return False

            screenshot = self.get_screenshot()
            if screenshot:
                king_pos = detect_cow_king(screenshot.image)
                if king_pos:
                    # Found the King! Kill him
                    self.combat.build.attack_target(
                        king_pos[0], king_pos[1],
                        use_static=False,
                        is_boss=True
                    )
                    time.sleep(2)

                    # Loot his drops
                    self.pick_up_items(max_items=10)
                    self.cow_king_killed = True
                    return True

                # Handle regular cows we encounter
                enemies = self.enemy_detector.detect_by_health_bars(screenshot.image)
                if enemies:
                    self._kill_cow_pack(enemies, screenshot.image)

            # Teleport to continue searching
            window = self.input.window_manager.get_window()
            if window:
                angle = search_count * 0.5
                distance = 150
                dx = int(math.cos(angle) * distance)
                dy = int(math.sin(angle) * distance)

                target_x = window.width // 2 + dx
                target_y = window.height // 2 + dy
                self.movement.teleport_to(target_x, target_y)

            time.sleep(0.2)

        return False
