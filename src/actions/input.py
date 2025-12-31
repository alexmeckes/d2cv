import time
import random
from typing import Tuple, Optional
import pydirectinput

from src.capture.window import WindowManager


class InputController:
    """Handles mouse and keyboard input for the game."""

    def __init__(self, window_manager: Optional[WindowManager] = None):
        self.window_manager = window_manager or WindowManager()

        # Disable pydirectinput's pause between actions for speed
        pydirectinput.PAUSE = 0.02

        # Human-like delay ranges (in seconds)
        self.click_delay = (0.02, 0.05)
        self.key_delay = (0.03, 0.08)
        self.movement_delay = (0.01, 0.03)

    def _humanize_delay(self, delay_range: Tuple[float, float]) -> None:
        """Add a small random delay to seem more human-like."""
        time.sleep(random.uniform(*delay_range))

    def _to_screen_coords(self, x: int, y: int) -> Tuple[int, int]:
        """Convert window-relative coords to screen coords."""
        window = self.window_manager.get_window()
        if window is None:
            return (x, y)
        return (window.x + x, window.y + y)

    def click(
        self,
        x: int,
        y: int,
        button: str = "left",
        relative: bool = True,
        clicks: int = 1
    ) -> None:
        """Click at the specified position.

        Args:
            x, y: Coordinates to click
            button: "left" or "right"
            relative: If True, coords are relative to game window
            clicks: Number of clicks
        """
        if relative:
            x, y = self._to_screen_coords(x, y)

        pydirectinput.moveTo(x, y)
        self._humanize_delay(self.movement_delay)

        for _ in range(clicks):
            pydirectinput.click(x, y, button=button)
            if clicks > 1:
                self._humanize_delay(self.click_delay)

        self._humanize_delay(self.click_delay)

    def right_click(self, x: int, y: int, relative: bool = True) -> None:
        """Right-click at position."""
        self.click(x, y, button="right", relative=relative)

    def move_to(self, x: int, y: int, relative: bool = True) -> None:
        """Move mouse to position without clicking."""
        if relative:
            x, y = self._to_screen_coords(x, y)
        pydirectinput.moveTo(x, y)
        self._humanize_delay(self.movement_delay)

    def press_key(self, key: str, hold_time: float = 0.05) -> None:
        """Press and release a key.

        Args:
            key: Key to press (e.g., 'f1', 'a', 'space', 'shift')
            hold_time: How long to hold the key (seconds)
        """
        pydirectinput.keyDown(key)
        time.sleep(hold_time)
        pydirectinput.keyUp(key)
        self._humanize_delay(self.key_delay)

    def hold_key(self, key: str) -> None:
        """Hold a key down (call release_key to let go)."""
        pydirectinput.keyDown(key)

    def release_key(self, key: str) -> None:
        """Release a held key."""
        pydirectinput.keyUp(key)
        self._humanize_delay(self.key_delay)

    def type_text(self, text: str, interval: float = 0.05) -> None:
        """Type a string of text."""
        for char in text:
            pydirectinput.press(char)
            time.sleep(interval)

    # D2-specific helpers

    def cast_skill(self, x: int, y: int, skill_key: Optional[str] = None) -> None:
        """Cast a skill at target location.

        Args:
            x, y: Target coordinates (relative to window)
            skill_key: Hotkey to press before casting (e.g., 'f1')
        """
        if skill_key:
            self.press_key(skill_key)

        self.right_click(x, y, relative=True)

    def teleport(self, x: int, y: int) -> None:
        """Teleport to target location (assumes teleport is on right-click)."""
        self.right_click(x, y, relative=True)

    def use_potion(self, slot: int) -> None:
        """Use a potion from belt.

        Args:
            slot: Belt slot (1-4)
        """
        if slot < 1 or slot > 4:
            return
        self.press_key(str(slot))

    def open_inventory(self) -> None:
        """Open/close inventory."""
        self.press_key("i")

    def town_portal(self) -> None:
        """Cast town portal (assumes default hotkey)."""
        self.press_key("t")

    def pick_up_item(self, x: int, y: int) -> None:
        """Click to pick up an item."""
        self.click(x, y, relative=True)

    def show_items(self, hold: bool = True) -> None:
        """Show/hide item labels on ground."""
        if hold:
            self.hold_key("alt")
        else:
            self.release_key("alt")
