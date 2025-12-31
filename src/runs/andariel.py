"""
Andariel farming run - Act 1 Hell.

Strategy:
1. Take waypoint to Catacombs Level 2
2. Teleport to find stairs to Level 3
3. Teleport through Level 3 to find stairs to Level 4
4. Find Andariel in Level 4
5. Kill Andariel (Static Field + Blizzard)
6. Loot drops
7. Town portal back
"""

import time
import math
from typing import Optional, Tuple

from .base_run import BaseRun, RunConfig, RunStatus
from src.vision.entities import DetectedEnemy


class AndarielRun(BaseRun):
    """Andariel farming run implementation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config = RunConfig(
            name="andariel",
            waypoint_act=1,
            waypoint_name="Catacombs Level 2",
            max_deaths=3,
            timeout_seconds=180,  # 3 minute timeout
            pickup_items=True,
            use_static_field=True,
        )

        # Andariel-specific state
        self.andariel_found = False
        self.andariel_dead = False
        self.catacombs4_entered = False

    def execute(self) -> RunStatus:
        """Execute the full Andariel run."""
        self.start()

        try:
            # Step 1: Take waypoint to Catacombs Level 2
            if not self._go_to_catacombs2():
                self.abort("Failed to reach Catacombs Level 2")
                return self.status

            # Step 2: Find and enter Catacombs Level 3
            if not self._find_catacombs3():
                self.abort("Failed to find Catacombs Level 3")
                return self.status

            # Step 3: Find and enter Catacombs Level 4
            if not self._find_catacombs4():
                self.abort("Failed to find Catacombs Level 4")
                return self.status

            # Step 4: Find Andariel
            if not self._find_andariel():
                self.abort("Failed to find Andariel")
                return self.status

            # Step 5: Kill Andariel
            if not self._kill_andariel():
                self.abort("Failed to kill Andariel")
                return self.status

            # Step 6: Loot
            if self.config.pickup_items:
                self._loot_andariel()

            # Step 7: Return to town
            if not self.return_to_town():
                self.abort("Failed to return to town")
                return self.status

            self.complete(success=True)

        except Exception as e:
            self.abort(f"Exception: {str(e)}")

        return self.status

    def _go_to_catacombs2(self) -> bool:
        """Take waypoint to Catacombs Level 2."""
        # Open waypoint menu
        if not self.town.use_waypoint():
            return False

        time.sleep(0.5)

        # Navigate waypoint UI to select Catacombs Level 2
        window = self.input.window_manager.get_window()
        if window:
            # Act 1 tab (should be default or first tab)
            act1_tab_x = window.width // 2 - 200
            act1_tab_y = 150
            self.input.click(act1_tab_x, act1_tab_y, relative=True)
            time.sleep(0.3)

            # Catacombs Level 2 entry (lower in the list)
            catacombs2_x = window.width // 2
            catacombs2_y = 380  # Lower portion of Act 1 waypoint list
            self.input.click(catacombs2_x, catacombs2_y, relative=True)
            time.sleep(2)  # Loading screen

        return True

    def _detect_stairs(self, image) -> Optional[Tuple[int, int]]:
        """Detect stairs/level transition in Catacombs."""
        import cv2
        import numpy as np

        # Catacombs stairs have a distinctive dark entrance
        # Look for dark rectangular areas that could be stairs
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find very dark regions (stairs appear as dark holes)
        _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            area = cv2.contourArea(contour)
            if 500 < area < 10000:  # Stairs-sized dark area
                x, y, w, h = cv2.boundingRect(contour)
                # Stairs tend to be somewhat rectangular
                aspect_ratio = w / max(h, 1)
                if 0.5 < aspect_ratio < 2.5:
                    # Check if it's in a valid screen region (not UI)
                    if y > 100 and y < image.shape[0] - 100:
                        return (x + w // 2, y + h // 2)

        return None

    def _find_catacombs3(self) -> bool:
        """Teleport through Catacombs 2 to find stairs to Level 3."""
        for teleport_count in range(50):
            # Check vitals
            _, should_continue = self.check_vitals()
            if not should_continue:
                return False

            if self.is_timed_out():
                return False

            # Capture and look for stairs
            screenshot = self.get_screenshot()
            if screenshot:
                stairs_pos = self._detect_stairs(screenshot.image)
                if stairs_pos:
                    # Found stairs! Click to enter
                    self.input.click(stairs_pos[0], stairs_pos[1], relative=True)
                    time.sleep(2)  # Loading
                    return True

            # Teleport in search pattern
            window = self.input.window_manager.get_window()
            if window:
                # Catacombs layout varies - use spiral search
                angle = teleport_count * 0.6
                distance = 100 + (teleport_count // 4) * 30
                dx = int(math.cos(angle) * distance)
                dy = int(math.sin(angle) * distance)

                target_x = window.width // 2 + dx
                target_y = window.height // 2 + dy
                self.movement.teleport_to(target_x, target_y)

            time.sleep(0.15)

        return False

    def _find_catacombs4(self) -> bool:
        """Teleport through Catacombs 3 to find stairs to Level 4."""
        for teleport_count in range(50):
            # Check vitals
            _, should_continue = self.check_vitals()
            if not should_continue:
                return False

            if self.is_timed_out():
                return False

            # Capture and look for stairs
            screenshot = self.get_screenshot()
            if screenshot:
                stairs_pos = self._detect_stairs(screenshot.image)
                if stairs_pos:
                    # Found stairs! Click to enter
                    self.input.click(stairs_pos[0], stairs_pos[1], relative=True)
                    time.sleep(2)  # Loading
                    self.catacombs4_entered = True
                    return True

            # Teleport in search pattern
            window = self.input.window_manager.get_window()
            if window:
                # Continue spiral search
                angle = teleport_count * 0.6
                distance = 100 + (teleport_count // 4) * 30
                dx = int(math.cos(angle) * distance)
                dy = int(math.sin(angle) * distance)

                target_x = window.width // 2 + dx
                target_y = window.height // 2 + dy
                self.movement.teleport_to(target_x, target_y)

            time.sleep(0.15)

        return False

    def _find_andariel(self) -> bool:
        """Find Andariel in Catacombs Level 4."""

        def detect_andariel(image) -> Optional[DetectedEnemy]:
            """Detect Andariel on screen."""
            # Andariel has a large health bar and distinctive green poison aura
            enemies = self.enemy_detector.detect_by_health_bars(image, min_width=50)

            # Andariel has a boss-sized health bar
            for enemy in enemies:
                if enemy.width > 70:  # Boss-sized health bar
                    enemy.name = "Andariel"
                    enemy.is_boss = True
                    return enemy

            return None

        # Catacombs 4 is small - Andariel is in her chamber
        for teleport_count in range(25):
            _, should_continue = self.check_vitals()
            if not should_continue:
                return False

            if self.is_timed_out():
                return False

            screenshot = self.get_screenshot()
            if screenshot:
                andariel = detect_andariel(screenshot.image)
                if andariel:
                    self.andariel_found = True
                    return True

            # Spiral search pattern
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

    def _kill_andariel(self) -> bool:
        """Kill Andariel using Blizzard Sorceress rotation."""

        def get_andariel_position(image) -> Optional[Tuple[int, int]]:
            """Get Andariel's current position."""
            enemies = self.enemy_detector.detect_by_health_bars(image, min_width=50)
            for enemy in enemies:
                if enemy.width > 70:
                    return enemy.center
            return None

        # Combat loop
        static_casts = 0
        max_static_casts = 4  # Static field 4 times first
        kill_timeout = time.time() + 45  # 45 second kill timeout (Andy is easier)

        while time.time() < kill_timeout:
            # Check vitals - Andariel does poison damage
            vitals, should_continue = self.check_vitals()
            if not should_continue:
                return False

            # Get screenshot and find Andariel
            screenshot = self.get_screenshot()
            if screenshot is None:
                continue

            andy_pos = get_andariel_position(screenshot.image)

            # Check if Andariel is dead
            if andy_pos is None:
                time.sleep(0.5)
                screenshot = self.get_screenshot()
                if screenshot:
                    andy_pos = get_andariel_position(screenshot.image)
                    if andy_pos is None:
                        self.andariel_dead = True
                        return True

            if andy_pos:
                # Position for attacking
                window = self.input.window_manager.get_window()
                if window:
                    center_x = window.width // 2
                    center_y = window.height // 2

                    # Calculate distance to Andariel
                    dist = math.sqrt(
                        (andy_pos[0] - center_x) ** 2 +
                        (andy_pos[1] - center_y) ** 2
                    )

                    # If too close, teleport back (Andariel has melee attacks)
                    if dist < 180:
                        away_x = center_x + (center_x - andy_pos[0])
                        away_y = center_y + (center_y - andy_pos[1])
                        away_x = max(100, min(window.width - 100, away_x))
                        away_y = max(100, min(window.height - 100, away_y))
                        self.movement.teleport_to(away_x, away_y)
                        continue

                # Cast Static Field first
                if self.config.use_static_field and static_casts < max_static_casts:
                    self.combat.build.cast_static_field()
                    static_casts += 1
                    continue

                # Cast Blizzard on Andariel
                self.combat.build.cast_blizzard(andy_pos[0], andy_pos[1])

                # Fill time with Glacial Spike
                self.combat.build.cast_glacial_spike(andy_pos[0], andy_pos[1])

            time.sleep(0.1)

        return False

    def _loot_andariel(self) -> int:
        """Loot items from Andariel kill."""
        # Wait for items to drop
        time.sleep(0.5)

        # Teleport to where Andariel died
        window = self.input.window_manager.get_window()
        if window:
            self.movement.teleport_to(window.width // 2, window.height // 2)

        time.sleep(0.3)

        # Pick up items
        return self.pick_up_items(max_items=15)
