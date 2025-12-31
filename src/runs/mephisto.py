"""
Mephisto farming run - Act 3 Hell.

Strategy:
1. Take waypoint to Durance of Hate Level 2
2. Teleport to find stairs to Level 3
3. Teleport around Level 3 to find Mephisto
4. Kill Mephisto (Static Field + Blizzard)
5. Loot drops
6. Town portal back
"""

import time
import math
from typing import Optional, Tuple

from .base_run import BaseRun, RunConfig, RunStatus
from src.vision.entities import DetectedEnemy


class MephistoRun(BaseRun):
    """Mephisto farming run implementation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config = RunConfig(
            name="mephisto",
            waypoint_act=3,
            waypoint_name="Durance of Hate Level 2",
            max_deaths=3,
            timeout_seconds=180,  # 3 minute timeout
            pickup_items=True,
            use_static_field=True,
        )

        # Mephisto-specific state
        self.mephisto_found = False
        self.mephisto_dead = False
        self.durance3_entered = False

    def execute(self) -> RunStatus:
        """Execute the full Mephisto run."""
        self.start()

        try:
            # Step 1: Take waypoint to Durance Level 2
            if not self._go_to_durance2():
                self.abort("Failed to reach Durance Level 2")
                return self.status

            # Step 2: Find and enter Durance Level 3
            if not self._find_durance3():
                self.abort("Failed to find Durance Level 3")
                return self.status

            # Step 3: Find Mephisto
            if not self._find_mephisto():
                self.abort("Failed to find Mephisto")
                return self.status

            # Step 4: Kill Mephisto
            if not self._kill_mephisto():
                self.abort("Failed to kill Mephisto")
                return self.status

            # Step 5: Loot
            if self.config.pickup_items:
                self._loot_mephisto()

            # Step 6: Return to town
            if not self.return_to_town():
                self.abort("Failed to return to town")
                return self.status

            self.complete(success=True)

        except Exception as e:
            self.abort(f"Exception: {str(e)}")

        return self.status

    def _go_to_durance2(self) -> bool:
        """Take waypoint to Durance of Hate Level 2."""
        # Open waypoint menu
        if not self.town.use_waypoint():
            return False

        time.sleep(0.5)

        # TODO: Navigate waypoint UI to select Durance Level 2
        # For now, assume waypoint UI is open and we need to click the right location
        # This requires template matching or fixed positions for waypoint entries

        # Approximate position for Act 3 waypoints in 1280x720
        # Durance of Hate Level 2 is typically in the bottom section
        window = self.input.window_manager.get_window()
        if window:
            # Act 3 tab (if not already selected)
            act3_tab_x = window.width // 2 - 100
            act3_tab_y = 150
            self.input.click(act3_tab_x, act3_tab_y, relative=True)
            time.sleep(0.3)

            # Durance Level 2 entry (approximate)
            durance2_x = window.width // 2
            durance2_y = 400  # Lower portion of waypoint list
            self.input.click(durance2_x, durance2_y, relative=True)
            time.sleep(2)  # Loading screen

        return True

    def _find_durance3(self) -> bool:
        """Teleport through Durance 2 to find stairs to Level 3."""

        def detect_stairs(image):
            """Detect stairs/level transition."""
            # Look for the distinctive red portal/stairs to level 3
            # This is a simplified detection - real implementation needs templates
            from src.vision.detector import ColorDetector

            # Durance 3 entrance has a reddish glow
            hsv_img = __import__('cv2').cvtColor(image, __import__('cv2').COLOR_BGR2HSV)
            mask = __import__('cv2').inRange(
                hsv_img,
                __import__('numpy').array([0, 100, 100]),
                __import__('numpy').array([15, 255, 255])
            )

            # Find contours
            contours, _ = __import__('cv2').findContours(
                mask, __import__('cv2').RETR_EXTERNAL, __import__('cv2').CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                area = __import__('cv2').contourArea(contour)
                if area > 1000:  # Significant red area
                    x, y, w, h = __import__('cv2').boundingRect(contour)
                    return (x + w // 2, y + h // 2)

            return None

        # Teleport pattern for Durance 2
        # The level has a specific layout - generally teleport in one direction
        # following the "blood river" or walls

        for teleport_count in range(60):  # Max teleports
            # Check vitals
            _, should_continue = self.check_vitals()
            if not should_continue:
                return False

            if self.is_timed_out():
                return False

            # Capture and look for stairs
            screenshot = self.get_screenshot()
            if screenshot:
                stairs_pos = detect_stairs(screenshot.image)
                if stairs_pos:
                    # Found stairs! Click to enter
                    self.input.click(stairs_pos[0], stairs_pos[1], relative=True)
                    time.sleep(2)  # Loading
                    self.durance3_entered = True
                    return True

            # Teleport in search pattern
            # Durance 2 layout: generally want to go in a specific direction
            # This is a simplified approach - real bot would use minimap
            window = self.input.window_manager.get_window()
            if window:
                # Teleport towards bottom-right generally (common layout)
                # Alternate between directions to explore
                directions = [
                    (150, 100),   # Right-down
                    (100, 150),   # Down-right
                    (150, 50),    # Right
                    (50, 150),    # Down
                ]
                dx, dy = directions[teleport_count % len(directions)]

                target_x = window.width // 2 + dx
                target_y = window.height // 2 + dy
                self.movement.teleport_to(target_x, target_y)

            time.sleep(0.15)

        return False

    def _find_mephisto(self) -> bool:
        """Find Mephisto in Durance Level 3."""

        def detect_mephisto(image) -> Optional[DetectedEnemy]:
            """Detect Mephisto on screen."""
            # Look for Mephisto's large health bar or distinctive appearance
            enemies = self.enemy_detector.detect_by_health_bars(image, min_width=60)

            # Mephisto has a very large health bar
            for enemy in enemies:
                if enemy.width > 80:  # Boss-sized health bar
                    enemy.name = "Mephisto"
                    enemy.is_boss = True
                    return enemy

            return None

        # Durance 3 is small - Mephisto is in the center on his platform
        # Teleport around to find him

        for teleport_count in range(30):
            _, should_continue = self.check_vitals()
            if not should_continue:
                return False

            if self.is_timed_out():
                return False

            screenshot = self.get_screenshot()
            if screenshot:
                mephisto = detect_mephisto(screenshot.image)
                if mephisto:
                    self.mephisto_found = True
                    return True

            # Spiral search pattern
            window = self.input.window_manager.get_window()
            if window:
                angle = teleport_count * 0.8
                distance = 120
                dx = int(math.cos(angle) * distance)
                dy = int(math.sin(angle) * distance)

                target_x = window.width // 2 + dx
                target_y = window.height // 2 + dy
                self.movement.teleport_to(target_x, target_y)

            time.sleep(0.15)

        return False

    def _kill_mephisto(self) -> bool:
        """Kill Mephisto using Blizzard Sorceress rotation."""

        def get_mephisto_position(image) -> Optional[Tuple[int, int]]:
            """Get Mephisto's current position."""
            enemies = self.enemy_detector.detect_by_health_bars(image, min_width=60)
            for enemy in enemies:
                if enemy.width > 80:
                    return enemy.center
            return None

        # Combat loop
        static_casts = 0
        max_static_casts = 5  # Static field 5 times first
        kill_timeout = time.time() + 60  # 60 second kill timeout

        while time.time() < kill_timeout:
            # Check vitals
            vitals, should_continue = self.check_vitals()
            if not should_continue:
                return False

            # Get screenshot and find Mephisto
            screenshot = self.get_screenshot()
            if screenshot is None:
                continue

            meph_pos = get_mephisto_position(screenshot.image)

            # Check if Mephisto is dead (no health bar found)
            if meph_pos is None:
                # Double-check by waiting a moment
                time.sleep(0.5)
                screenshot = self.get_screenshot()
                if screenshot:
                    meph_pos = get_mephisto_position(screenshot.image)
                    if meph_pos is None:
                        self.mephisto_dead = True
                        return True

            if meph_pos:
                # Position for attacking (stay at safe distance)
                window = self.input.window_manager.get_window()
                if window:
                    center_x = window.width // 2
                    center_y = window.height // 2

                    # Calculate distance to Mephisto
                    dist = math.sqrt(
                        (meph_pos[0] - center_x) ** 2 +
                        (meph_pos[1] - center_y) ** 2
                    )

                    # If too close, teleport back
                    if dist < 150:
                        # Teleport away from Mephisto
                        away_x = center_x + (center_x - meph_pos[0])
                        away_y = center_y + (center_y - meph_pos[1])
                        # Clamp to screen
                        away_x = max(100, min(window.width - 100, away_x))
                        away_y = max(100, min(window.height - 100, away_y))
                        self.movement.teleport_to(away_x, away_y)
                        continue

                # Cast Static Field first (chunks boss HP)
                if self.config.use_static_field and static_casts < max_static_casts:
                    self.combat.build.cast_static_field()
                    static_casts += 1
                    continue

                # Cast Blizzard on Mephisto
                self.combat.build.cast_blizzard(meph_pos[0], meph_pos[1])

                # Fill time with Glacial Spike
                self.combat.build.cast_glacial_spike(meph_pos[0], meph_pos[1])

            time.sleep(0.1)

        return False

    def _loot_mephisto(self) -> int:
        """Loot items from Mephisto kill."""
        # Wait for items to drop
        time.sleep(0.5)

        # Teleport to where Mephisto died (center of platform)
        window = self.input.window_manager.get_window()
        if window:
            # Mephisto dies on his platform - teleport there
            self.movement.teleport_to(window.width // 2, window.height // 2 - 50)

        time.sleep(0.3)

        # Pick up items
        return self.pick_up_items(max_items=15)
