"""
Base class for farming runs.
"""

import time
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum, auto

from src.capture import ScreenCapture
from src.vision import GameStateDetector, ItemDetector, EnemyDetector, VitalsState
from src.actions import InputController, CombatController, MovementController, TownController
from src.brain import ReactiveBrain, Action
from src.brain.item_evaluator import ItemEvaluator
from src.state import SessionStats


class RunStatus(Enum):
    """Status of a run."""
    NOT_STARTED = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    ABORTED = auto()


@dataclass
class RunConfig:
    """Configuration for a run."""
    name: str
    waypoint_act: int
    waypoint_name: str
    max_deaths: int = 3
    timeout_seconds: int = 300  # 5 minute timeout
    pickup_items: bool = True
    use_static_field: bool = True  # For bosses


class BaseRun(ABC):
    """Abstract base class for all farming runs."""

    def __init__(
        self,
        screen: ScreenCapture,
        input_ctrl: InputController,
        combat: CombatController,
        movement: MovementController,
        town: TownController,
        game_state: GameStateDetector,
        item_detector: ItemDetector,
        enemy_detector: EnemyDetector,
        brain: ReactiveBrain,
        stats: SessionStats,
        item_evaluator: Optional[ItemEvaluator] = None,
    ):
        self.screen = screen
        self.input = input_ctrl
        self.combat = combat
        self.movement = movement
        self.town = town
        self.game_state = game_state
        self.item_detector = item_detector
        self.enemy_detector = enemy_detector
        self.brain = brain
        self.stats = stats
        self.item_evaluator = item_evaluator

        # Run state
        self.status = RunStatus.NOT_STARTED
        self.start_time = 0
        self.deaths = 0
        self.items_found: List[str] = []

        # Config (override in subclass)
        self.config = RunConfig(
            name="base",
            waypoint_act=1,
            waypoint_name="",
        )

    @abstractmethod
    def execute(self) -> RunStatus:
        """Execute the full run. Override in subclasses."""
        pass

    def start(self) -> None:
        """Start the run."""
        self.status = RunStatus.IN_PROGRESS
        self.start_time = time.time()
        self.deaths = 0
        self.items_found = []
        self.stats.start_run(self.config.name)

    def complete(self, success: bool = True) -> None:
        """Mark run as complete."""
        self.status = RunStatus.COMPLETED if success else RunStatus.FAILED
        self.stats.end_run(success, self.deaths)

    def abort(self, reason: str = "") -> None:
        """Abort the run."""
        self.status = RunStatus.ABORTED
        self.stats.end_run(False, self.deaths, error=reason)

    def is_timed_out(self) -> bool:
        """Check if run has exceeded timeout."""
        return time.time() - self.start_time > self.config.timeout_seconds

    def get_screenshot(self):
        """Capture and return current screenshot."""
        return self.screen.capture()

    def check_vitals(self) -> Tuple[VitalsState, bool]:
        """Check health/mana and handle emergencies.

        Returns:
            (vitals, should_continue) - False if we need to chicken
        """
        screenshot = self.get_screenshot()
        if screenshot is None:
            return VitalsState(0, 0, False, False), False

        vitals = self.game_state.get_vitals(screenshot.image)
        belt = self.game_state.get_belt_state(screenshot.image)

        # Get reactive brain decisions
        decisions = self.brain.evaluate(
            vitals=vitals,
            belt=belt,
            enemies=[],
            items=[],
        )

        # Execute survival actions
        for decision in decisions:
            if decision.action == Action.CHICKEN:
                self._emergency_exit()
                return vitals, False

            if decision.action == Action.USE_HEALTH_POTION:
                slot = decision.details.get("slot", 1)
                self.input.use_potion(slot)
                self.brain.record_action(Action.USE_HEALTH_POTION)

            if decision.action == Action.USE_MANA_POTION:
                slot = decision.details.get("slot", 3)
                self.input.use_potion(slot)
                self.brain.record_action(Action.USE_MANA_POTION)

        return vitals, True

    def _emergency_exit(self) -> None:
        """Emergency exit - town portal or save & quit."""
        self.town.cast_town_portal()
        time.sleep(2)  # Wait for portal

        # Try to find and enter portal
        screenshot = self.get_screenshot()
        if screenshot:
            from src.vision.entities import PortalDetector
            portal_detector = PortalDetector()
            portal = portal_detector.detect_town_portal(screenshot.image)
            if portal:
                self.input.click(portal.center[0], portal.center[1], relative=True)
                time.sleep(2)

    def teleport_until_found(
        self,
        detect_func,
        max_teleports: int = 50,
        search_pattern: str = "spiral"
    ) -> Optional[any]:
        """Teleport around until something is detected.

        Args:
            detect_func: Function that takes screenshot and returns detected object or None
            max_teleports: Maximum teleports before giving up
            search_pattern: "spiral", "random", or "linear"

        Returns:
            Detected object or None
        """
        for i in range(max_teleports):
            # Check vitals
            _, should_continue = self.check_vitals()
            if not should_continue:
                return None

            # Check for timeout
            if self.is_timed_out():
                self.abort("Timeout during search")
                return None

            # Capture and detect
            screenshot = self.get_screenshot()
            if screenshot:
                result = detect_func(screenshot.image)
                if result:
                    return result

            # Teleport to next position
            if search_pattern == "spiral":
                # Spiral outward
                angle = i * 0.5
                distance = 100 + i * 20
                import math
                dx = int(math.cos(angle) * distance)
                dy = int(math.sin(angle) * distance)

                window = self.input.window_manager.get_window()
                if window:
                    target_x = window.width // 2 + dx
                    target_y = window.height // 2 + dy
                    self.movement.teleport_to(target_x, target_y)
            else:
                # Random direction
                import random
                directions = ["up", "down", "left", "right", "up-left", "up-right"]
                self.movement.teleport_direction(random.choice(directions))

            time.sleep(0.1)

        return None

    def pick_up_items(self, max_items: int = 20) -> int:
        """Pick up items on ground.

        Uses item_evaluator if available for smarter pickup decisions.

        Returns:
            Number of items picked up
        """
        picked = 0

        for _ in range(max_items):
            # Check vitals between pickups
            _, should_continue = self.check_vitals()
            if not should_continue:
                break

            # Show items (hold alt)
            self.input.show_items(hold=True)
            time.sleep(0.1)

            # Capture and detect items
            screenshot = self.get_screenshot()
            if screenshot is None:
                break

            items = self.item_detector.detect_items(screenshot.image)

            if not items:
                # Check for gold
                gold = self.item_detector.detect_gold(screenshot.image)
                if gold:
                    items = gold[:3]

            if not items:
                break

            # Use item evaluator if available
            if self.item_evaluator:
                pickup_list = self.item_evaluator.get_pickup_list(
                    items,
                    screenshot.image,
                    max_items=5
                )
                if not pickup_list:
                    break

                # Pick up the highest priority item
                eval_item = pickup_list[0]
                item = eval_item.detected
                item_name = eval_item.name or item.rarity.value

                self.input.click(item.center[0], item.center[1], relative=True)
                time.sleep(0.2)

                self.items_found.append(item.rarity.value)
                self.stats.record_item(
                    name=item_name,
                    rarity=item.rarity.value,
                    area=self.config.name,
                    llm_evaluation=eval_item.evaluation.reason if eval_item.evaluation else None,
                )
                picked += 1
            else:
                # Fallback: simple rarity filter
                valuable_items = [
                    item for item in items
                    if item.rarity.value in ("unique", "set", "rare", "rune")
                ]

                if not valuable_items:
                    break

                item = valuable_items[0]
                self.input.click(item.center[0], item.center[1], relative=True)
                time.sleep(0.2)

                self.items_found.append(item.rarity.value)
                self.stats.record_item(
                    name=item.name or "Unknown",
                    rarity=item.rarity.value,
                    area=self.config.name,
                )
                picked += 1

        # Release alt
        self.input.show_items(hold=False)

        return picked

    def return_to_town(self) -> bool:
        """Cast TP and return to town.

        Returns:
            True if successfully returned to town
        """
        self.town.cast_town_portal()
        time.sleep(1.5)

        # Find and enter portal
        for _ in range(5):
            screenshot = self.get_screenshot()
            if screenshot is None:
                continue

            from src.vision.entities import PortalDetector
            portal_detector = PortalDetector()
            portal = portal_detector.detect_town_portal(screenshot.image)

            if portal:
                self.input.click(portal.center[0], portal.center[1], relative=True)
                time.sleep(2)  # Loading
                return True

            time.sleep(0.3)

        return False
