"""
Mephisto farming run - Gemini-powered version.

Uses Gemini Flash for:
- Navigation (finding stairs, Mephisto)
- Combat decisions (threat assessment, positioning)
- Item evaluation (what to pick up)

Latency handling:
- Async calls where possible
- Caching recent responses
- Fallback to last known state if slow
- Reactive brain handles instant decisions
"""

import time
import math
import asyncio
import threading
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, Future

from .base_run import BaseRun, RunConfig, RunStatus
from src.brain.navigation_advisor import (
    NavigationAdvisor, NavigationTarget, Direction,
    get_navigation_advisor
)
from src.brain.combat_advisor import (
    CombatAdvisor, CombatAction, ThreatLevel,
    get_combat_advisor
)
from src.brain.item_evaluator import ItemEvaluator
from src.state.session_logger import get_logger


@dataclass
class GeminiState:
    """Cached state from Gemini calls."""
    last_nav_advice: Optional[Dict] = None
    last_nav_time: float = 0
    last_combat_advice: Optional[Dict] = None
    last_combat_time: float = 0
    pending_nav_future: Optional[Future] = None
    pending_combat_future: Optional[Future] = None


class MephistoGeminiRun(BaseRun):
    """Mephisto run using Gemini for all decisions."""

    # Cache durations
    NAV_CACHE_DURATION = 2.0      # Reuse nav advice for 2 seconds
    COMBAT_CACHE_DURATION = 1.0  # Reuse combat advice for 1 second

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config = RunConfig(
            name="mephisto_gemini",
            waypoint_act=3,
            waypoint_name="Durance of Hate Level 2",
            max_deaths=3,
            timeout_seconds=180,
            pickup_items=True,
            use_static_field=True,
        )

        # Gemini advisors
        self.nav = get_navigation_advisor()
        self.combat = get_combat_advisor()
        self.logger = get_logger("mephisto_gemini")

        # State
        self.gemini_state = GeminiState()
        self.mephisto_dead = False

        # Thread pool for async Gemini calls
        self.executor = ThreadPoolExecutor(max_workers=2)

        # Stats
        self.gemini_calls = 0
        self.cache_hits = 0
        self.total_latency = 0

    def execute(self) -> RunStatus:
        """Execute the full Mephisto run."""
        self.start()

        try:
            # Step 1: Go to Durance Level 2 via waypoint
            if not self._go_to_durance2():
                self.abort("Failed to reach Durance Level 2")
                return self.status

            # Step 2: Navigate to Durance Level 3 (Gemini-powered)
            if not self._navigate_to_durance3():
                self.abort("Failed to find Durance Level 3")
                return self.status

            # Step 3: Find and kill Mephisto (Gemini-powered)
            if not self._find_and_kill_mephisto():
                self.abort("Failed to kill Mephisto")
                return self.status

            # Step 4: Loot (Gemini-powered item evaluation)
            if self.config.pickup_items:
                self._loot_with_gemini()

            # Step 5: Return to town
            if not self.return_to_town():
                self.abort("Failed to return to town")
                return self.status

            self.complete(success=True)
            self._log_stats()

        except Exception as e:
            self.logger.error(f"Run exception: {e}")
            self.abort(f"Exception: {str(e)}")

        finally:
            self.executor.shutdown(wait=False)

        return self.status

    def _go_to_durance2(self) -> bool:
        """Take waypoint to Durance of Hate Level 2."""
        if not self.town.use_waypoint():
            return False

        time.sleep(0.5)

        window = self.input.window_manager.get_window()
        if window:
            # Act 3 tab
            self.input.click(window.width // 2 - 100, 150, relative=True)
            time.sleep(0.3)

            # Durance Level 2
            self.input.click(window.width // 2, 400, relative=True)
            time.sleep(2)

        return True

    def _navigate_to_durance3(self) -> bool:
        """Navigate to Durance Level 3 using Gemini."""
        self.logger.info("Navigating to Durance Level 3...")

        for teleport_count in range(80):  # Max teleports
            # Check vitals
            _, should_continue = self.check_vitals()
            if not should_continue:
                return False

            if self.is_timed_out():
                return False

            # Capture screenshot
            screenshot = self.get_screenshot()
            if screenshot is None:
                continue

            # Get navigation advice (with caching and async)
            nav_advice = self._get_nav_advice_smart(
                screenshot.image,
                "stairs to Durance Level 3"
            )

            # If stairs found, click them
            if nav_advice and nav_advice.get("target_found"):
                target_loc = nav_advice.get("target_location")
                if target_loc:
                    self.logger.info(f"Found stairs at {target_loc}")
                    self.input.click(target_loc[0], target_loc[1], relative=True)
                    time.sleep(2)  # Loading
                    return True

            # Teleport in suggested direction
            direction = nav_advice.get("direction", Direction.EAST) if nav_advice else Direction.EAST
            self._teleport_direction(direction)

            time.sleep(0.15)

        return False

    def _find_and_kill_mephisto(self) -> bool:
        """Find and kill Mephisto using Gemini."""
        self.logger.info("Searching for Mephisto...")

        # Phase 1: Find Mephisto
        mephisto_location = None
        for search_count in range(40):
            _, should_continue = self.check_vitals()
            if not should_continue:
                return False

            screenshot = self.get_screenshot()
            if screenshot is None:
                continue

            # Look for Mephisto
            nav_advice = self._get_nav_advice_smart(
                screenshot.image,
                "Mephisto boss"
            )

            if nav_advice and nav_advice.get("target_found"):
                mephisto_location = nav_advice.get("target_location")
                self.logger.info(f"Found Mephisto at {mephisto_location}")
                break

            # Teleport to find him
            direction = nav_advice.get("direction", Direction.SOUTH) if nav_advice else Direction.SOUTH
            self._teleport_direction(direction)
            time.sleep(0.15)

        if not mephisto_location:
            return False

        # Phase 2: Kill Mephisto
        self.logger.info("Engaging Mephisto...")
        static_casts = 0
        kill_timeout = time.time() + 60

        while time.time() < kill_timeout:
            # Check vitals
            vitals, should_continue = self.check_vitals()
            if not should_continue:
                return False

            screenshot = self.get_screenshot()
            if screenshot is None:
                continue

            # Get combat advice
            combat_advice = self._get_combat_advice_smart(
                screenshot.image,
                vitals.health_percent if hasattr(vitals, 'health_percent') else 1.0
            )

            # Handle combat advice
            if combat_advice:
                action = combat_advice.get("action", CombatAction.ENGAGE)

                if action == CombatAction.CHICKEN:
                    self._emergency_exit()
                    return False
                elif action == CombatAction.RETREAT:
                    self._teleport_away_from(mephisto_location)
                    continue

                # Check if Mephisto is dead
                if combat_advice.get("threat_level") == ThreatLevel.SAFE:
                    if not combat_advice.get("boss_present", True):
                        self.logger.info("Mephisto defeated!")
                        self.mephisto_dead = True
                        return True

            # Find current Mephisto position
            nav_advice = self._get_nav_advice_smart(
                screenshot.image,
                "Mephisto boss",
                force=False  # Use cache if recent
            )

            if nav_advice and nav_advice.get("target_found"):
                mephisto_location = nav_advice.get("target_location")

            if mephisto_location:
                # Maintain distance
                window = self.input.window_manager.get_window()
                if window:
                    center = (window.width // 2, window.height // 2)
                    dist = math.sqrt(
                        (mephisto_location[0] - center[0]) ** 2 +
                        (mephisto_location[1] - center[1]) ** 2
                    )

                    if dist < 150:
                        self._teleport_away_from(mephisto_location)
                        continue

                # Cast Static Field first
                if self.config.use_static_field and static_casts < 5:
                    self.combat.build.cast_static_field()
                    static_casts += 1
                    continue

                # Cast Blizzard
                self.combat.build.cast_blizzard(mephisto_location[0], mephisto_location[1])
                self.combat.build.cast_glacial_spike(mephisto_location[0], mephisto_location[1])

            time.sleep(0.1)

        return False

    def _loot_with_gemini(self) -> int:
        """Loot items using Gemini batch evaluation."""
        self.logger.info("Looting...")
        time.sleep(0.5)

        # Move to loot area
        window = self.input.window_manager.get_window()
        if window:
            self.movement.teleport_to(window.width // 2, window.height // 2 - 50)

        time.sleep(0.3)

        # Show items
        self.input.show_items(hold=True)
        time.sleep(0.2)

        # Capture and evaluate with Gemini
        screenshot = self.get_screenshot()
        if screenshot is None:
            self.input.show_items(hold=False)
            return 0

        # Use batch evaluation (single Gemini call for all items)
        pickup_list = self.item_evaluator.get_pickup_list_batch(
            screenshot.image,
            max_items=10
        )

        picked = 0
        for item in pickup_list:
            if item.should_pickup:
                self.logger.info(f"Picking up: {item.name} ({item.evaluation.reason})")

                # Need to find item position on screen
                # For batch mode, we need to re-detect or use Gemini to locate
                nav_advice = self._get_nav_advice_smart(
                    screenshot.image,
                    f"item {item.name}",
                    force=True
                )

                if nav_advice and nav_advice.get("target_found"):
                    loc = nav_advice.get("target_location")
                    if loc:
                        self.input.click(loc[0], loc[1], relative=True)
                        time.sleep(0.2)
                        picked += 1

                        self.stats.record_item(
                            name=item.name,
                            rarity=item.detected.rarity.value,
                            area=self.config.name,
                            llm_evaluation=item.evaluation.reason,
                        )

        self.input.show_items(hold=False)
        return picked

    def _get_nav_advice_smart(
        self,
        image,
        target: str,
        force: bool = False
    ) -> Optional[Dict]:
        """Get navigation advice with caching and async."""
        now = time.time()

        # Check cache
        if not force and self.gemini_state.last_nav_advice:
            if now - self.gemini_state.last_nav_time < self.NAV_CACHE_DURATION:
                self.cache_hits += 1
                return self.gemini_state.last_nav_advice

        # Check if async call is pending
        if self.gemini_state.pending_nav_future:
            if self.gemini_state.pending_nav_future.done():
                try:
                    result = self.gemini_state.pending_nav_future.result()
                    self.gemini_state.last_nav_advice = result
                    self.gemini_state.last_nav_time = now
                    self.gemini_state.pending_nav_future = None
                    return result
                except:
                    self.gemini_state.pending_nav_future = None
            else:
                # Still pending, use cached
                return self.gemini_state.last_nav_advice

        # Make sync call (for now - could make async)
        self.gemini_calls += 1
        start = time.time()

        try:
            if "stairs" in target.lower():
                advice = self.nav.find_stairs(image, target)
            elif "boss" in target.lower() or "mephisto" in target.lower():
                advice = self.nav.find_boss(image, "Mephisto")
            else:
                advice = self.nav.find_target(image, NavigationTarget.ITEM, target)

            latency = (time.time() - start) * 1000
            self.total_latency += latency

            result = {
                "target_found": advice.target_found,
                "target_location": advice.target_location,
                "direction": advice.suggested_direction,
                "reasoning": advice.reasoning,
                "confidence": advice.confidence,
            }

            self.gemini_state.last_nav_advice = result
            self.gemini_state.last_nav_time = now

            return result

        except Exception as e:
            self.logger.error(f"Nav advice failed: {e}")
            return self.gemini_state.last_nav_advice

    def _get_combat_advice_smart(
        self,
        image,
        health_percent: float
    ) -> Optional[Dict]:
        """Get combat advice with caching."""
        now = time.time()

        # Check cache
        if self.gemini_state.last_combat_advice:
            if now - self.gemini_state.last_combat_time < self.COMBAT_CACHE_DURATION:
                self.cache_hits += 1
                return self.gemini_state.last_combat_advice

        self.gemini_calls += 1
        start = time.time()

        try:
            advice = self.combat.get_combat_advice(
                image,
                health_percent=health_percent,
                current_action="fighting_mephisto",
                force=True
            )

            latency = (time.time() - start) * 1000
            self.total_latency += latency

            result = {
                "action": advice.recommended_action,
                "threat_level": advice.threat_level,
                "boss_present": advice.boss_present,
                "reasoning": advice.reasoning,
                "immunities": advice.immunities_detected,
            }

            self.gemini_state.last_combat_advice = result
            self.gemini_state.last_combat_time = now

            return result

        except Exception as e:
            self.logger.error(f"Combat advice failed: {e}")
            return self.gemini_state.last_combat_advice

    def _teleport_direction(self, direction: Direction) -> None:
        """Teleport in a direction."""
        window = self.input.window_manager.get_window()
        if not window:
            return

        center_x = window.width // 2
        center_y = window.height // 2
        offset = 200

        offsets = {
            Direction.NORTH: (0, -offset),
            Direction.SOUTH: (0, offset),
            Direction.EAST: (offset, 0),
            Direction.WEST: (-offset, 0),
            Direction.NORTHEAST: (offset, -offset),
            Direction.NORTHWEST: (-offset, -offset),
            Direction.SOUTHEAST: (offset, offset),
            Direction.SOUTHWEST: (-offset, offset),
        }

        dx, dy = offsets.get(direction, (offset, 0))
        self.movement.teleport_to(center_x + dx, center_y + dy)

    def _teleport_away_from(self, location: Tuple[int, int]) -> None:
        """Teleport away from a location."""
        window = self.input.window_manager.get_window()
        if not window:
            return

        center_x = window.width // 2
        center_y = window.height // 2

        # Teleport opposite direction
        dx = center_x - location[0]
        dy = center_y - location[1]

        # Normalize and scale
        dist = max(1, math.sqrt(dx*dx + dy*dy))
        dx = int(dx / dist * 200)
        dy = int(dy / dist * 200)

        target_x = max(100, min(window.width - 100, center_x + dx))
        target_y = max(100, min(window.height - 100, center_y + dy))

        self.movement.teleport_to(target_x, target_y)

    def _log_stats(self) -> None:
        """Log Gemini usage stats."""
        avg_latency = self.total_latency / max(1, self.gemini_calls)
        total_calls = self.gemini_calls + self.cache_hits
        cache_rate = self.cache_hits / max(1, total_calls)

        self.logger.info(
            f"Gemini stats: {self.gemini_calls} calls, "
            f"{self.cache_hits} cache hits ({cache_rate:.0%}), "
            f"avg latency {avg_latency:.0f}ms"
        )
