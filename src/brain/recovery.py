"""
Error recovery system - handles stuck detection and intelligent recovery.
"""

import time
from typing import Optional, List, Callable, Dict, Any
from dataclasses import dataclass
from enum import Enum, auto

from .deliberative import DeliberativeBrain, RecoveryPlan, get_deliberative_brain
from src.state.session_logger import get_logger


class RecoveryAction(Enum):
    """Possible recovery actions."""
    WAIT = auto()
    TELEPORT_RANDOM = auto()
    RETURN_TO_TOWN = auto()
    USE_WAYPOINT = auto()
    DRINK_POTION = auto()
    SAVE_AND_EXIT = auto()
    RESTART_RUN = auto()
    PAUSE_BOT = auto()


@dataclass
class StuckDetection:
    """Stuck detection state."""
    last_position: Optional[tuple] = None
    position_unchanged_count: int = 0
    last_action_time: float = 0
    consecutive_failures: int = 0
    last_screen_hash: Optional[int] = None
    screen_unchanged_count: int = 0


@dataclass
class RecoveryContext:
    """Context for recovery decisions."""
    health_percent: float
    mana_percent: float
    location: str
    last_actions: List[str]
    current_state: str
    error_description: str
    screen_description: str
    stuck_duration: float
    consecutive_failures: int


class ErrorRecoverySystem:
    """Handles error detection and recovery."""

    def __init__(
        self,
        brain: Optional[DeliberativeBrain] = None,
        use_llm: bool = True,
    ):
        """Initialize the recovery system.

        Args:
            brain: Deliberative brain for LLM-based recovery
            use_llm: Whether to use LLM for complex recovery decisions
        """
        self._brain = brain
        self.use_llm = use_llm
        self.logger = get_logger("recovery")

        # Stuck detection
        self.stuck = StuckDetection()

        # Thresholds
        self.stuck_position_threshold = 5  # Same position N times = stuck
        self.stuck_screen_threshold = 10   # Same screen N times = stuck
        self.action_timeout = 30.0         # No action for N seconds = stuck

        # Action history
        self.action_history: List[str] = []
        self.recovery_attempts: List[Dict[str, Any]] = []

        # Callbacks
        self.on_recovery_needed: Optional[Callable[[RecoveryContext], None]] = None

    @property
    def brain(self) -> DeliberativeBrain:
        if self._brain is None:
            self._brain = get_deliberative_brain()
        return self._brain

    def update_position(self, x: int, y: int) -> bool:
        """Update current position and check for stuck.

        Args:
            x: Current X position (screen center or minimap)
            y: Current Y position

        Returns:
            True if stuck detected
        """
        current_pos = (x, y)

        if self.stuck.last_position == current_pos:
            self.stuck.position_unchanged_count += 1
        else:
            self.stuck.position_unchanged_count = 0
            self.stuck.last_position = current_pos

        return self.stuck.position_unchanged_count >= self.stuck_position_threshold

    def update_screen(self, screen_hash: int) -> bool:
        """Update screen hash and check for stuck.

        Args:
            screen_hash: Hash of current screen (for comparison)

        Returns:
            True if stuck detected
        """
        if self.stuck.last_screen_hash == screen_hash:
            self.stuck.screen_unchanged_count += 1
        else:
            self.stuck.screen_unchanged_count = 0
            self.stuck.last_screen_hash = screen_hash

        return self.stuck.screen_unchanged_count >= self.stuck_screen_threshold

    def record_action(self, action: str) -> None:
        """Record an action was taken."""
        self.stuck.last_action_time = time.time()
        self.action_history.append(action)

        # Keep history limited
        if len(self.action_history) > 50:
            self.action_history.pop(0)

    def record_failure(self) -> None:
        """Record a failure occurred."""
        self.stuck.consecutive_failures += 1
        self.logger.warning(
            f"Failure recorded, consecutive: {self.stuck.consecutive_failures}"
        )

    def record_success(self) -> None:
        """Record a success (reset failure counter)."""
        self.stuck.consecutive_failures = 0

    def check_action_timeout(self) -> bool:
        """Check if no action has been taken for too long.

        Returns:
            True if timed out
        """
        if self.stuck.last_action_time == 0:
            return False

        elapsed = time.time() - self.stuck.last_action_time
        return elapsed > self.action_timeout

    def is_stuck(self) -> bool:
        """Check if bot appears to be stuck."""
        return (
            self.stuck.position_unchanged_count >= self.stuck_position_threshold or
            self.stuck.screen_unchanged_count >= self.stuck_screen_threshold or
            self.check_action_timeout()
        )

    def get_recovery_action(self, context: RecoveryContext) -> RecoveryAction:
        """Get the appropriate recovery action.

        Uses heuristics first, then LLM for complex situations.

        Args:
            context: Current context for recovery decision

        Returns:
            RecoveryAction to take
        """
        # Emergency health check
        if context.health_percent < 0.2:
            self.logger.info("Low health - returning to town")
            return RecoveryAction.RETURN_TO_TOWN

        # Too many failures - pause
        if context.consecutive_failures >= 5:
            self.logger.warning("Too many failures - pausing bot")
            return RecoveryAction.PAUSE_BOT

        # Simple stuck - try teleporting
        if self.stuck.position_unchanged_count < 10:
            self.logger.info("Mildly stuck - teleporting random")
            return RecoveryAction.TELEPORT_RANDOM

        # Moderately stuck - return to town
        if self.stuck.position_unchanged_count < 20:
            self.logger.info("Moderately stuck - returning to town")
            return RecoveryAction.RETURN_TO_TOWN

        # Use LLM for complex recovery
        if self.use_llm:
            return self._get_llm_recovery(context)

        # Fallback: save and exit
        self.logger.warning("Severely stuck - save and exit")
        return RecoveryAction.SAVE_AND_EXIT

    def _get_llm_recovery(self, context: RecoveryContext) -> RecoveryAction:
        """Get recovery action from LLM.

        Args:
            context: Recovery context

        Returns:
            RecoveryAction based on LLM advice
        """
        try:
            plan = self.brain.get_recovery_plan(
                health_percent=context.health_percent,
                mana_percent=context.mana_percent,
                location=context.location,
                last_actions=context.last_actions[-5:],
                current_state=context.current_state,
                error_description=context.error_description,
                screen_description=context.screen_description,
            )

            self.logger.info(f"LLM diagnosis: {plan.diagnosis}")
            self.logger.info(f"Severity: {plan.severity}")

            # Record recovery attempt
            self.recovery_attempts.append({
                "time": time.time(),
                "context": context,
                "plan": plan,
            })

            if plan.should_abort:
                return RecoveryAction.SAVE_AND_EXIT

            # Map LLM actions to RecoveryAction
            if plan.actions:
                first_action = plan.actions[0]
                action_type = first_action.get("action", "").lower()

                action_mapping = {
                    "teleport": RecoveryAction.TELEPORT_RANDOM,
                    "town": RecoveryAction.RETURN_TO_TOWN,
                    "waypoint": RecoveryAction.USE_WAYPOINT,
                    "potion": RecoveryAction.DRINK_POTION,
                    "wait": RecoveryAction.WAIT,
                    "exit": RecoveryAction.SAVE_AND_EXIT,
                    "restart": RecoveryAction.RESTART_RUN,
                }

                for keyword, action in action_mapping.items():
                    if keyword in action_type:
                        return action

            # Default based on severity
            severity_actions = {
                "low": RecoveryAction.TELEPORT_RANDOM,
                "medium": RecoveryAction.RETURN_TO_TOWN,
                "high": RecoveryAction.SAVE_AND_EXIT,
                "critical": RecoveryAction.PAUSE_BOT,
            }
            return severity_actions.get(plan.severity.lower(), RecoveryAction.RETURN_TO_TOWN)

        except Exception as e:
            self.logger.error(f"LLM recovery failed: {e}")
            return RecoveryAction.RETURN_TO_TOWN

    def execute_recovery(
        self,
        action: RecoveryAction,
        teleport_func: Optional[Callable] = None,
        town_func: Optional[Callable] = None,
        potion_func: Optional[Callable] = None,
        save_exit_func: Optional[Callable] = None,
    ) -> bool:
        """Execute a recovery action.

        Args:
            action: Recovery action to execute
            teleport_func: Function to teleport randomly
            town_func: Function to return to town
            potion_func: Function to use potion
            save_exit_func: Function to save and exit game

        Returns:
            True if recovery executed successfully
        """
        self.logger.info(f"Executing recovery: {action.name}")

        try:
            if action == RecoveryAction.WAIT:
                time.sleep(2)
                return True

            elif action == RecoveryAction.TELEPORT_RANDOM:
                if teleport_func:
                    teleport_func()
                    time.sleep(0.2)
                return True

            elif action == RecoveryAction.RETURN_TO_TOWN:
                if town_func:
                    return town_func()
                return False

            elif action == RecoveryAction.DRINK_POTION:
                if potion_func:
                    potion_func()
                return True

            elif action == RecoveryAction.SAVE_AND_EXIT:
                if save_exit_func:
                    save_exit_func()
                return True

            elif action == RecoveryAction.RESTART_RUN:
                # This should be handled by the run manager
                return True

            elif action == RecoveryAction.PAUSE_BOT:
                # This should be handled by the bot controller
                return False

        except Exception as e:
            self.logger.error(f"Recovery execution failed: {e}")
            return False

        return True

    def reset(self) -> None:
        """Reset stuck detection state."""
        self.stuck = StuckDetection()
        self.logger.debug("Stuck detection reset")

    def get_stats(self) -> Dict[str, Any]:
        """Get recovery system statistics."""
        return {
            "position_unchanged_count": self.stuck.position_unchanged_count,
            "screen_unchanged_count": self.stuck.screen_unchanged_count,
            "consecutive_failures": self.stuck.consecutive_failures,
            "recovery_attempts": len(self.recovery_attempts),
            "action_history_length": len(self.action_history),
        }


# Global instance
_recovery_system: Optional[ErrorRecoverySystem] = None


def get_recovery_system() -> ErrorRecoverySystem:
    """Get the global recovery system instance."""
    global _recovery_system
    if _recovery_system is None:
        _recovery_system = ErrorRecoverySystem()
    return _recovery_system
