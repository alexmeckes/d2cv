"""
Main bot controller - ties all components together.
"""

import time
import threading
from typing import Optional, Callable
from dataclasses import dataclass

from src.config import get_config, Config
from src.capture import ScreenCapture, WindowManager
from src.vision import (
    GameStateDetector, ItemDetector, EnemyDetector, PortalDetector,
    TemplateMatcher, GameOCR
)
from src.actions import InputController, CombatController, MovementController, TownController
from src.brain import ReactiveBrain, DeliberativeBrain, get_deliberative_brain
from src.brain.item_evaluator import ItemEvaluator
from src.state import SessionStats, BotStateMachine, BotState
from src.runs.run_manager import RunManager, RunResult


@dataclass
class BotStatus:
    """Current bot status for GUI/logging."""
    state: str
    current_run: Optional[str]
    health_percent: float
    mana_percent: float
    runs_completed: int
    items_found: int
    runtime_seconds: float


class D2Bot:
    """Main Diablo 2 bot controller."""

    def __init__(self, config: Optional[Config] = None):
        """Initialize the bot with all components."""
        self.config = config or get_config()

        # Core components
        self.window_manager = WindowManager()
        self.screen = ScreenCapture(self.window_manager)
        self.input = InputController(self.window_manager)

        # Vision components
        self.template_matcher = TemplateMatcher()
        self.game_state_detector = GameStateDetector()
        self.item_detector = ItemDetector()
        self.enemy_detector = EnemyDetector(self.template_matcher)
        self.portal_detector = PortalDetector(self.template_matcher)
        self.ocr = GameOCR()

        # Action controllers
        self.combat = CombatController(self.input)
        self.movement = MovementController(self.input)
        self.town = TownController(self.input, self.movement)

        # Brain
        self.brain = ReactiveBrain()

        # LLM-powered brain (lazy loaded)
        self._deliberative_brain: Optional[DeliberativeBrain] = None
        self.item_evaluator = ItemEvaluator(
            ocr=self.ocr,
            use_llm=self.config.llm.enabled,
        )

        # State
        self.stats = SessionStats()
        self.state_machine = BotStateMachine()

        # Run manager
        self.run_manager = RunManager(
            screen=self.screen,
            input_ctrl=self.input,
            combat=self.combat,
            movement=self.movement,
            town=self.town,
            game_state=self.game_state_detector,
            item_detector=self.item_detector,
            enemy_detector=self.enemy_detector,
            brain=self.brain,
            stats=self.stats,
            item_evaluator=self.item_evaluator,
        )

        # Threading
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Callbacks
        self.on_status_update: Optional[Callable[[BotStatus], None]] = None
        self.on_run_complete: Optional[Callable[[RunResult], None]] = None
        self.on_screenshot: Optional[Callable, None] = None

        # Start time
        self._start_time = 0

    @property
    def deliberative_brain(self) -> DeliberativeBrain:
        """Lazy-load deliberative brain."""
        if self._deliberative_brain is None:
            self._deliberative_brain = get_deliberative_brain()
        return self._deliberative_brain

    def initialize(self) -> bool:
        """Initialize bot and verify game is running.

        Returns:
            True if initialization successful
        """
        # Find game window
        window = self.window_manager.find_window()
        if window is None:
            print("ERROR: Could not find Diablo 2 window")
            print("Make sure the game is running")
            return False

        print(f"Found game window: {window.title}")
        print(f"  Size: {window.width}x{window.height}")

        # Verify resolution matches config
        expected_w = self.config.window_width
        expected_h = self.config.window_height
        if window.width != expected_w or window.height != expected_h:
            print(f"WARNING: Window size {window.width}x{window.height} "
                  f"doesn't match expected {expected_w}x{expected_h}")
            print("Vision detection may not work correctly")

        # Load templates (if any exist)
        templates_loaded = self.template_matcher.load_templates_from_dir("ui")
        templates_loaded += self.template_matcher.load_templates_from_dir("bosses")
        print(f"Loaded {templates_loaded} templates")

        # Test screen capture
        screenshot = self.screen.capture()
        if screenshot is None:
            print("ERROR: Could not capture screen")
            return False

        print("Screen capture working")

        # Benchmark capture speed
        fps = self.screen.get_fps_benchmark(30)
        print(f"Capture speed: {fps:.1f} FPS")

        return True

    def start(self) -> bool:
        """Start the bot in a background thread.

        Returns:
            True if started successfully
        """
        if self._running:
            print("Bot is already running")
            return False

        if not self.initialize():
            return False

        self._running = True
        self._stop_event.clear()
        self._start_time = time.time()

        self._thread = threading.Thread(target=self._main_loop, daemon=True)
        self._thread.start()

        print("Bot started")
        return True

    def stop(self) -> None:
        """Stop the bot."""
        if not self._running:
            return

        print("Stopping bot...")
        self._stop_event.set()
        self.run_manager.stop()

        if self._thread:
            self._thread.join(timeout=5)

        self._running = False
        print("Bot stopped")

    def is_running(self) -> bool:
        """Check if bot is running."""
        return self._running

    def get_status(self) -> BotStatus:
        """Get current bot status."""
        # Get latest vitals if possible
        health = 1.0
        mana = 1.0

        screenshot = self.screen.capture(refresh_window=False)
        if screenshot:
            vitals = self.game_state_detector.get_vitals(screenshot.image)
            health = vitals.health_percent
            mana = vitals.mana_percent

        return BotStatus(
            state=self.state_machine.state.name,
            current_run=self.run_manager.current_run.config.name if self.run_manager.current_run else None,
            health_percent=health,
            mana_percent=mana,
            runs_completed=self.stats.runs_completed,
            items_found=sum(self.stats.items_by_rarity.values()),
            runtime_seconds=time.time() - self._start_time if self._start_time else 0,
        )

    def _main_loop(self) -> None:
        """Main bot loop - runs in background thread."""
        print("Entering main loop")

        # Focus game window
        self.window_manager.focus_window()
        time.sleep(0.5)

        while not self._stop_event.is_set():
            try:
                # Execute runs
                result = self.run_manager.execute_single_run()

                if result:
                    print(f"Run complete: {result.run_name} - {result.status.name}")
                    print(f"  Duration: {result.duration:.1f}s, Items: {result.items_found}")

                    if self.on_run_complete:
                        self.on_run_complete(result)

                # Update status callback
                if self.on_status_update:
                    self.on_status_update(self.get_status())

                # Brief pause between runs
                if not self._stop_event.is_set():
                    # Town routine between runs
                    self._do_town_routine()
                    time.sleep(1)

            except Exception as e:
                print(f"Error in main loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(5)  # Pause before retrying

        print("Exited main loop")

    def _do_town_routine(self) -> None:
        """Execute town routine between runs."""
        # Basic town routine
        # TODO: Implement full routine with repair, stash, etc.

        # For now, just make sure we're in town and healthy
        screenshot = self.screen.capture()
        if screenshot:
            vitals = self.game_state_detector.get_vitals(screenshot.image)

            # Use potions if needed
            if vitals.health_percent < 0.8:
                self.input.use_potion(1)
            if vitals.mana_percent < 0.5:
                self.input.use_potion(3)

    def capture_debug_screenshot(self, filename: str = "debug.png") -> bool:
        """Capture and save a debug screenshot.

        Returns:
            True if successful
        """
        import cv2

        screenshot = self.screen.capture()
        if screenshot:
            cv2.imwrite(filename, screenshot.image)
            print(f"Saved debug screenshot to {filename}")
            return True
        return False


# Convenience function for quick testing
def create_bot() -> D2Bot:
    """Create a bot instance with default config."""
    return D2Bot()
