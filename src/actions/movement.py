"""
Movement actions - teleporting, walking, pathfinding.
"""

import time
import math
from typing import Optional, Tuple, List
from dataclasses import dataclass

from .input import InputController
from src.config import get_config


@dataclass
class Position:
    """A position on screen or in game world."""
    x: int
    y: int

    def distance_to(self, other: 'Position') -> float:
        """Calculate distance to another position."""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def direction_to(self, other: 'Position') -> Tuple[float, float]:
        """Get normalized direction vector to another position."""
        dx = other.x - self.x
        dy = other.y - self.y
        dist = self.distance_to(other)
        if dist == 0:
            return (0, 0)
        return (dx / dist, dy / dist)


class MovementController:
    """Handles character movement via teleport or walking."""

    def __init__(self, input_ctrl: Optional[InputController] = None):
        self.input = input_ctrl or InputController()
        self.config = get_config()

        # Teleport settings
        self.teleport_key = "f1"  # Will be overridden by config
        self.teleport_delay = 0.12  # Time between teleports
        self.max_teleport_distance = 300  # Max screen pixels per teleport

        # Movement state
        self.last_teleport_time = 0
        self.last_position: Optional[Position] = None
        self.stuck_counter = 0
        self.stuck_threshold = 5  # Teleports without movement = stuck

    def set_teleport_key(self, key: str) -> None:
        """Set the teleport skill hotkey."""
        self.teleport_key = key

    def teleport_to(self, x: int, y: int) -> bool:
        """Teleport to a specific screen position.

        Args:
            x, y: Target screen coordinates (relative to window)

        Returns:
            True if teleport was executed
        """
        current_time = time.time()

        # Respect teleport delay
        if current_time - self.last_teleport_time < self.teleport_delay:
            time.sleep(self.teleport_delay - (current_time - self.last_teleport_time))

        # Select teleport skill and cast
        self.input.press_key(self.teleport_key)
        time.sleep(0.02)
        self.input.right_click(x, y, relative=True)

        self.last_teleport_time = time.time()
        time.sleep(self.config.timing.after_teleport)

        return True

    def teleport_towards(
        self,
        target_x: int,
        target_y: int,
        from_x: Optional[int] = None,
        from_y: Optional[int] = None
    ) -> Tuple[int, int]:
        """Teleport towards a target, respecting max teleport distance.

        Args:
            target_x, target_y: Ultimate target position
            from_x, from_y: Current position (defaults to screen center)

        Returns:
            (x, y) position we teleported to
        """
        window = self.input.window_manager.get_window()
        if not window:
            return (target_x, target_y)

        # Default from center of screen (character position)
        if from_x is None:
            from_x = window.width // 2
        if from_y is None:
            from_y = window.height // 2

        # Calculate distance
        dx = target_x - from_x
        dy = target_y - from_y
        distance = math.sqrt(dx * dx + dy * dy)

        # If close enough, teleport directly
        if distance <= self.max_teleport_distance:
            self.teleport_to(target_x, target_y)
            return (target_x, target_y)

        # Otherwise, teleport max distance towards target
        ratio = self.max_teleport_distance / distance
        tele_x = int(from_x + dx * ratio)
        tele_y = int(from_y + dy * ratio)

        self.teleport_to(tele_x, tele_y)
        return (tele_x, tele_y)

    def teleport_path(
        self,
        waypoints: List[Tuple[int, int]],
        tolerance: int = 50
    ) -> bool:
        """Teleport along a path of waypoints.

        Args:
            waypoints: List of (x, y) positions to teleport through
            tolerance: How close to get to each waypoint

        Returns:
            True if path was completed
        """
        window = self.input.window_manager.get_window()
        if not window:
            return False

        for wx, wy in waypoints:
            # Teleport until we reach this waypoint
            max_attempts = 10
            for _ in range(max_attempts):
                center_x = window.width // 2
                center_y = window.height // 2

                distance = math.sqrt((wx - center_x) ** 2 + (wy - center_y) ** 2)
                if distance <= tolerance:
                    break

                self.teleport_towards(wx, wy, center_x, center_y)

        return True

    def walk_to(self, x: int, y: int, hold_time: float = 0.5) -> None:
        """Walk to a position by holding left-click.

        Args:
            x, y: Target screen position
            hold_time: How long to hold movement
        """
        self.input.click(x, y, button="left", relative=True)
        time.sleep(hold_time)

    def teleport_direction(
        self,
        direction: str,
        distance: int = 200
    ) -> bool:
        """Teleport in a cardinal direction.

        Args:
            direction: "up", "down", "left", "right", or diagonal combinations
            distance: How far to teleport

        Returns:
            True if teleport executed
        """
        window = self.input.window_manager.get_window()
        if not window:
            return False

        center_x = window.width // 2
        center_y = window.height // 2

        # Calculate target based on direction
        dx, dy = 0, 0

        if "up" in direction:
            dy = -distance
        if "down" in direction:
            dy = distance
        if "left" in direction:
            dx = -distance
        if "right" in direction:
            dx = distance

        # Normalize diagonal movement
        if dx != 0 and dy != 0:
            factor = distance / math.sqrt(dx * dx + dy * dy)
            dx = int(dx * factor)
            dy = int(dy * factor)

        target_x = center_x + dx
        target_y = center_y + dy

        return self.teleport_to(target_x, target_y)

    def spiral_search(
        self,
        start_radius: int = 100,
        max_radius: int = 500,
        step: int = 100
    ) -> None:
        """Spiral outward from current position (for searching).

        Useful for finding bosses, waypoints, or exits.
        """
        window = self.input.window_manager.get_window()
        if not window:
            return

        center_x = window.width // 2
        center_y = window.height // 2

        # Spiral pattern
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # right, down, left, up

        radius = start_radius
        while radius <= max_radius:
            for dx, dy in directions:
                target_x = center_x + int(dx * radius)
                target_y = center_y + int(dy * radius)

                # Clamp to screen
                target_x = max(50, min(window.width - 50, target_x))
                target_y = max(50, min(window.height - 50, target_y))

                self.teleport_to(target_x, target_y)
                time.sleep(0.2)  # Pause to observe

            radius += step

    def is_stuck(self, current_position: Position) -> bool:
        """Check if character is stuck (not moving despite teleporting).

        Call this after teleporting with the detected character position.
        """
        if self.last_position is None:
            self.last_position = current_position
            return False

        # Check if position changed significantly
        if current_position.distance_to(self.last_position) < 20:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        self.last_position = current_position
        return self.stuck_counter >= self.stuck_threshold

    def unstuck(self) -> None:
        """Try to get unstuck by teleporting in random directions."""
        import random

        directions = ["up", "down", "left", "right", "up-left", "up-right", "down-left", "down-right"]

        for _ in range(3):
            direction = random.choice(directions)
            self.teleport_direction(direction, distance=150)
            time.sleep(0.2)

        self.stuck_counter = 0
