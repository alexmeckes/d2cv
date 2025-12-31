"""
Ancient Tunnels farming run - Act 2 Hell.

Strategy:
1. Take waypoint to Lost City
2. Find entrance to Ancient Tunnels
3. Clear Ancient Tunnels (area has no cold immunes - perfect for Blizz Sorc)
4. Find and open Super Chest if present
5. Loot valuable drops
6. Town portal back

Note: Ancient Tunnels is a level 85 area with no cold immune monsters,
making it ideal for Blizzard Sorceress magic finding.
"""

import time
import math
from typing import Optional, Tuple, List

from .base_run import BaseRun, RunConfig, RunStatus
from src.vision.entities import DetectedEnemy


class AncientTunnelsRun(BaseRun):
    """Ancient Tunnels farming run implementation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config = RunConfig(
            name="ancient_tunnels",
            waypoint_act=2,
            waypoint_name="Lost City",
            max_deaths=3,
            timeout_seconds=300,  # 5 minute timeout (area clear takes longer)
            pickup_items=True,
            use_static_field=False,  # Regular monsters, no need for static
        )

        # Ancient Tunnels-specific state
        self.tunnels_entered = False
        self.monsters_killed = 0
        self.super_chest_found = False

    def execute(self) -> RunStatus:
        """Execute the full Ancient Tunnels run."""
        self.start()

        try:
            # Step 1: Take waypoint to Lost City
            if not self._go_to_lost_city():
                self.abort("Failed to reach Lost City")
                return self.status

            # Step 2: Find entrance to Ancient Tunnels
            if not self._find_tunnels_entrance():
                self.abort("Failed to find Ancient Tunnels entrance")
                return self.status

            # Step 3: Clear Ancient Tunnels
            cleared = self._clear_tunnels()

            # Step 4: Look for Super Chest
            self._find_super_chest()

            # Step 5: Return to town
            if not self.return_to_town():
                self.abort("Failed to return to town")
                return self.status

            self.complete(success=True)

        except Exception as e:
            self.abort(f"Exception: {str(e)}")

        return self.status

    def _go_to_lost_city(self) -> bool:
        """Take waypoint to Lost City."""
        # Open waypoint menu
        if not self.town.use_waypoint():
            return False

        time.sleep(0.5)

        # Navigate waypoint UI to select Lost City
        window = self.input.window_manager.get_window()
        if window:
            # Act 2 tab
            act2_tab_x = window.width // 2 - 150
            act2_tab_y = 150
            self.input.click(act2_tab_x, act2_tab_y, relative=True)
            time.sleep(0.3)

            # Lost City entry
            lost_city_x = window.width // 2
            lost_city_y = 320  # Mid-section of Act 2 waypoint list
            self.input.click(lost_city_x, lost_city_y, relative=True)
            time.sleep(2)  # Loading screen

        return True

    def _find_tunnels_entrance(self) -> bool:
        """Find the trapdoor entrance to Ancient Tunnels in Lost City."""

        def detect_trapdoor(image) -> Optional[Tuple[int, int]]:
            """Detect the trapdoor entrance."""
            import cv2
            import numpy as np

            # Ancient Tunnels entrance is a trapdoor in the ground
            # It has a distinctive dark/brown coloring
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Look for brownish trapdoor
            lower_brown = np.array([10, 50, 30])
            upper_brown = np.array([25, 200, 150])
            mask = cv2.inRange(hsv, lower_brown, upper_brown)

            # Find contours
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                area = cv2.contourArea(contour)
                if 800 < area < 8000:  # Trapdoor-sized area
                    x, y, w, h = cv2.boundingRect(contour)
                    # Trapdoor is roughly square
                    aspect_ratio = w / max(h, 1)
                    if 0.6 < aspect_ratio < 1.6:
                        if y > 150 and y < image.shape[0] - 100:
                            return (x + w // 2, y + h // 2)

            return None

        # Search Lost City for the trapdoor
        for teleport_count in range(40):
            # Check vitals
            _, should_continue = self.check_vitals()
            if not should_continue:
                return False

            if self.is_timed_out():
                return False

            # Check for monsters and handle them
            screenshot = self.get_screenshot()
            if screenshot:
                # First, check for trapdoor
                trapdoor_pos = detect_trapdoor(screenshot.image)
                if trapdoor_pos:
                    # Found entrance! Click to enter
                    self.input.click(trapdoor_pos[0], trapdoor_pos[1], relative=True)
                    time.sleep(2)  # Loading
                    self.tunnels_entered = True
                    return True

                # Handle any monsters we encounter
                self._handle_monsters(screenshot.image)

            # Teleport in search pattern
            # Lost City is outdoor - use wider search pattern
            window = self.input.window_manager.get_window()
            if window:
                angle = teleport_count * 0.5
                distance = 150 + (teleport_count // 6) * 50
                dx = int(math.cos(angle) * distance)
                dy = int(math.sin(angle) * distance)

                target_x = window.width // 2 + dx
                target_y = window.height // 2 + dy
                self.movement.teleport_to(target_x, target_y)

            time.sleep(0.2)

        return False

    def _clear_tunnels(self) -> int:
        """Clear Ancient Tunnels of monsters."""
        monsters_killed = 0
        clear_teleports = 0
        max_clear_teleports = 80  # More teleports for area clear

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

            # Look for monsters
            enemies = self.enemy_detector.detect_by_health_bars(screenshot.image)

            if enemies:
                # Kill visible monsters
                killed = self._kill_pack(enemies, screenshot.image)
                monsters_killed += killed

                # Loot after killing
                if killed > 0:
                    self.pick_up_items(max_items=5)
            else:
                # No monsters visible, teleport to next area
                window = self.input.window_manager.get_window()
                if window:
                    # Use expanding spiral pattern
                    angle = clear_teleports * 0.4
                    distance = 120 + (clear_teleports // 10) * 20
                    dx = int(math.cos(angle) * distance)
                    dy = int(math.sin(angle) * distance)

                    target_x = window.width // 2 + dx
                    target_y = window.height // 2 + dy
                    self.movement.teleport_to(target_x, target_y)

                clear_teleports += 1
                time.sleep(0.15)

        self.monsters_killed = monsters_killed
        return monsters_killed

    def _kill_pack(self, enemies: List[DetectedEnemy], image) -> int:
        """Kill a pack of monsters."""
        killed = 0
        pack_timeout = time.time() + 10  # 10 seconds per pack max

        while enemies and time.time() < pack_timeout:
            # Check vitals
            vitals, should_continue = self.check_vitals()
            if not should_continue:
                break

            # Target the closest/strongest enemy
            target = enemies[0]

            window = self.input.window_manager.get_window()
            if window:
                center_x = window.width // 2
                center_y = window.height // 2

                # Check distance
                dist = math.sqrt(
                    (target.center[0] - center_x) ** 2 +
                    (target.center[1] - center_y) ** 2
                )

                # If too close, back off
                if dist < 150:
                    away_x = center_x - (target.center[0] - center_x)
                    away_y = center_y - (target.center[1] - center_y)
                    away_x = max(100, min(window.width - 100, away_x))
                    away_y = max(100, min(window.height - 100, away_y))
                    self.movement.teleport_to(away_x, away_y)
                    time.sleep(0.1)
                    continue

            # Cast Blizzard on pack
            self.combat.build.cast_blizzard(target.center[0], target.center[1])

            # Use Glacial Spike for single targets
            self.combat.build.cast_glacial_spike(target.center[0], target.center[1])

            time.sleep(0.3)

            # Re-detect enemies
            screenshot = self.get_screenshot()
            if screenshot:
                new_enemies = self.enemy_detector.detect_by_health_bars(screenshot.image)
                if len(new_enemies) < len(enemies):
                    killed += len(enemies) - len(new_enemies)
                enemies = new_enemies

        return killed

    def _handle_monsters(self, image) -> None:
        """Handle any monsters encountered while searching."""
        enemies = self.enemy_detector.detect_by_health_bars(image)
        if enemies:
            self._kill_pack(enemies, image)

    def _find_super_chest(self) -> bool:
        """Look for and open the Super Chest if present."""

        def detect_chest(image) -> Optional[Tuple[int, int]]:
            """Detect a super chest."""
            import cv2
            import numpy as np

            # Super chests have a distinctive golden/brown color
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Golden chest color
            lower_gold = np.array([15, 100, 100])
            upper_gold = np.array([35, 255, 255])
            mask = cv2.inRange(hsv, lower_gold, upper_gold)

            # Find contours
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                area = cv2.contourArea(contour)
                if 200 < area < 3000:  # Chest-sized area
                    x, y, w, h = cv2.boundingRect(contour)
                    if y > 100 and y < image.shape[0] - 100:
                        return (x + w // 2, y + h // 2)

            return None

        # Search for super chest (brief search)
        for teleport_count in range(15):
            _, should_continue = self.check_vitals()
            if not should_continue:
                return False

            screenshot = self.get_screenshot()
            if screenshot:
                chest_pos = detect_chest(screenshot.image)
                if chest_pos:
                    # Found chest! Click to open
                    self.input.click(chest_pos[0], chest_pos[1], relative=True)
                    time.sleep(0.5)

                    # Pick up any drops
                    self.pick_up_items(max_items=10)
                    self.super_chest_found = True
                    return True

            # Quick teleport pattern
            window = self.input.window_manager.get_window()
            if window:
                angle = teleport_count * 0.8
                distance = 100
                dx = int(math.cos(angle) * distance)
                dy = int(math.sin(angle) * distance)

                target_x = window.width // 2 + dx
                target_y = window.height // 2 + dy
                self.movement.teleport_to(target_x, target_y)

            time.sleep(0.15)

        return False
