"""
Bot state machine - manages game states and transitions.
"""

import time
from typing import Optional, Callable, Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod


class BotState(Enum):
    """High-level bot states."""
    IDLE = auto()
    INITIALIZING = auto()
    IN_TOWN = auto()
    TRAVELING = auto()  # Using waypoint, teleporting to destination
    FARMING = auto()  # In a run, looking for enemies/boss
    COMBAT = auto()  # Fighting enemies
    LOOTING = auto()  # Picking up items
    RETURNING = auto()  # Going back to town
    DEAD = auto()
    ERROR = auto()
    PAUSED = auto()


class RunPhase(Enum):
    """Phases within a farming run."""
    NOT_STARTED = auto()
    LEAVING_TOWN = auto()
    NAVIGATING = auto()  # Teleporting to target area
    SEARCHING = auto()  # Looking for boss/enemies
    KILLING = auto()  # Combat phase
    LOOTING = auto()  # Picking up drops
    EXITING = auto()  # Leaving area
    COMPLETED = auto()


@dataclass
class StateContext:
    """Context data passed between states."""
    # Current run info
    current_run: Optional[str] = None
    run_phase: RunPhase = RunPhase.NOT_STARTED

    # Vitals
    health_percent: float = 1.0
    mana_percent: float = 1.0

    # Combat
    enemies_on_screen: int = 0
    boss_found: bool = False
    boss_dead: bool = False

    # Loot
    items_on_ground: int = 0
    items_to_pickup: List[Any] = field(default_factory=list)

    # Navigation
    current_area: Optional[str] = None
    target_area: Optional[str] = None

    # Counters
    deaths_this_run: int = 0
    total_deaths: int = 0
    runs_completed: int = 0

    # Timestamps
    run_start_time: float = 0
    state_enter_time: float = 0

    # Error handling
    last_error: Optional[str] = None
    consecutive_errors: int = 0


class StateHandler(ABC):
    """Abstract base class for state handlers."""

    @abstractmethod
    def enter(self, context: StateContext) -> None:
        """Called when entering this state."""
        pass

    @abstractmethod
    def update(self, context: StateContext) -> Optional[BotState]:
        """Called each tick. Returns new state or None to stay."""
        pass

    @abstractmethod
    def exit(self, context: StateContext) -> None:
        """Called when leaving this state."""
        pass


class IdleHandler(StateHandler):
    """Handler for IDLE state - waiting to start."""

    def enter(self, context: StateContext) -> None:
        pass

    def update(self, context: StateContext) -> Optional[BotState]:
        # Stay idle until externally triggered
        return None

    def exit(self, context: StateContext) -> None:
        pass


class InTownHandler(StateHandler):
    """Handler for IN_TOWN state."""

    def __init__(self, town_callback: Optional[Callable] = None):
        self.town_callback = town_callback
        self.town_tasks_complete = False

    def enter(self, context: StateContext) -> None:
        self.town_tasks_complete = False
        context.run_phase = RunPhase.NOT_STARTED

    def update(self, context: StateContext) -> Optional[BotState]:
        # Execute town routine if callback provided
        if not self.town_tasks_complete and self.town_callback:
            self.town_callback(context)
            self.town_tasks_complete = True

        # If we have a run to do, start traveling
        if context.current_run:
            return BotState.TRAVELING

        return None

    def exit(self, context: StateContext) -> None:
        context.run_start_time = time.time()


class TravelingHandler(StateHandler):
    """Handler for TRAVELING state - using waypoints."""

    def __init__(self, travel_callback: Optional[Callable] = None):
        self.travel_callback = travel_callback

    def enter(self, context: StateContext) -> None:
        context.run_phase = RunPhase.LEAVING_TOWN

    def update(self, context: StateContext) -> Optional[BotState]:
        # Check for danger during travel
        if context.health_percent < 0.3:
            return BotState.RETURNING

        if self.travel_callback:
            arrived = self.travel_callback(context)
            if arrived:
                return BotState.FARMING

        # Timeout check
        if time.time() - context.state_enter_time > 60:
            context.last_error = "Travel timeout"
            return BotState.ERROR

        return None

    def exit(self, context: StateContext) -> None:
        context.run_phase = RunPhase.NAVIGATING


class FarmingHandler(StateHandler):
    """Handler for FARMING state - searching for enemies."""

    def __init__(self, search_callback: Optional[Callable] = None):
        self.search_callback = search_callback

    def enter(self, context: StateContext) -> None:
        context.run_phase = RunPhase.SEARCHING
        context.boss_found = False
        context.boss_dead = False

    def update(self, context: StateContext) -> Optional[BotState]:
        # Check for danger
        if context.health_percent < 0.2:
            return BotState.RETURNING

        # Check for enemies
        if context.enemies_on_screen > 0 or context.boss_found:
            return BotState.COMBAT

        # Check for items (area cleared)
        if context.items_on_ground > 0 and context.boss_dead:
            return BotState.LOOTING

        # Search for boss/enemies
        if self.search_callback:
            self.search_callback(context)

        # Run complete check
        if context.boss_dead and context.items_on_ground == 0:
            return BotState.RETURNING

        return None

    def exit(self, context: StateContext) -> None:
        pass


class CombatHandler(StateHandler):
    """Handler for COMBAT state."""

    def __init__(self, combat_callback: Optional[Callable] = None):
        self.combat_callback = combat_callback

    def enter(self, context: StateContext) -> None:
        context.run_phase = RunPhase.KILLING

    def update(self, context: StateContext) -> Optional[BotState]:
        # Emergency exit
        if context.health_percent < 0.15:
            return BotState.RETURNING

        # Execute combat
        if self.combat_callback:
            self.combat_callback(context)

        # Check if combat is over
        if context.enemies_on_screen == 0:
            if context.boss_found:
                context.boss_dead = True
            return BotState.FARMING  # Go back to searching/looting

        return None

    def exit(self, context: StateContext) -> None:
        pass


class LootingHandler(StateHandler):
    """Handler for LOOTING state."""

    def __init__(self, loot_callback: Optional[Callable] = None):
        self.loot_callback = loot_callback

    def enter(self, context: StateContext) -> None:
        context.run_phase = RunPhase.LOOTING

    def update(self, context: StateContext) -> Optional[BotState]:
        # Check for danger while looting
        if context.enemies_on_screen > 0:
            return BotState.COMBAT

        if context.health_percent < 0.3:
            return BotState.RETURNING

        # Pick up items
        if self.loot_callback and context.items_to_pickup:
            self.loot_callback(context)

        # Done looting
        if not context.items_to_pickup and context.items_on_ground == 0:
            return BotState.FARMING

        return None

    def exit(self, context: StateContext) -> None:
        pass


class ReturningHandler(StateHandler):
    """Handler for RETURNING state - going back to town."""

    def __init__(self, return_callback: Optional[Callable] = None):
        self.return_callback = return_callback

    def enter(self, context: StateContext) -> None:
        context.run_phase = RunPhase.EXITING

    def update(self, context: StateContext) -> Optional[BotState]:
        if self.return_callback:
            in_town = self.return_callback(context)
            if in_town:
                context.runs_completed += 1
                context.run_phase = RunPhase.COMPLETED
                return BotState.IN_TOWN

        return None

    def exit(self, context: StateContext) -> None:
        context.current_run = None


class DeadHandler(StateHandler):
    """Handler for DEAD state."""

    def __init__(self, respawn_callback: Optional[Callable] = None):
        self.respawn_callback = respawn_callback

    def enter(self, context: StateContext) -> None:
        context.deaths_this_run += 1
        context.total_deaths += 1

    def update(self, context: StateContext) -> Optional[BotState]:
        # Too many deaths - abort run
        if context.deaths_this_run >= 3:
            context.last_error = "Too many deaths"
            return BotState.ERROR

        if self.respawn_callback:
            respawned = self.respawn_callback(context)
            if respawned:
                return BotState.IN_TOWN

        return None

    def exit(self, context: StateContext) -> None:
        pass


class BotStateMachine:
    """Main state machine for the bot."""

    def __init__(self):
        self.state = BotState.IDLE
        self.context = StateContext()

        # State handlers
        self.handlers: Dict[BotState, StateHandler] = {
            BotState.IDLE: IdleHandler(),
            BotState.IN_TOWN: InTownHandler(),
            BotState.TRAVELING: TravelingHandler(),
            BotState.FARMING: FarmingHandler(),
            BotState.COMBAT: CombatHandler(),
            BotState.LOOTING: LootingHandler(),
            BotState.RETURNING: ReturningHandler(),
            BotState.DEAD: DeadHandler(),
        }

        # Callbacks for state transitions
        self.on_state_change: Optional[Callable[[BotState, BotState], None]] = None

    def set_handler(self, state: BotState, handler: StateHandler) -> None:
        """Set a custom handler for a state."""
        self.handlers[state] = handler

    def transition_to(self, new_state: BotState) -> None:
        """Transition to a new state."""
        if new_state == self.state:
            return

        old_state = self.state

        # Exit old state
        if old_state in self.handlers:
            self.handlers[old_state].exit(self.context)

        # Update state
        self.state = new_state
        self.context.state_enter_time = time.time()

        # Enter new state
        if new_state in self.handlers:
            self.handlers[new_state].enter(self.context)

        # Notify callback
        if self.on_state_change:
            self.on_state_change(old_state, new_state)

    def update(self) -> None:
        """Execute one tick of the state machine."""
        if self.state == BotState.PAUSED:
            return

        handler = self.handlers.get(self.state)
        if handler:
            new_state = handler.update(self.context)
            if new_state:
                self.transition_to(new_state)

    def start(self) -> None:
        """Start the bot from idle."""
        if self.state == BotState.IDLE:
            self.transition_to(BotState.INITIALIZING)

    def pause(self) -> None:
        """Pause the bot."""
        self.transition_to(BotState.PAUSED)

    def resume(self) -> None:
        """Resume from paused state."""
        if self.state == BotState.PAUSED:
            self.transition_to(BotState.IN_TOWN)

    def stop(self) -> None:
        """Stop the bot."""
        self.transition_to(BotState.IDLE)
        self.context = StateContext()  # Reset context

    def is_running(self) -> bool:
        """Check if bot is actively running."""
        return self.state not in (BotState.IDLE, BotState.PAUSED, BotState.ERROR)
