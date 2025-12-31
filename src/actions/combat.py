"""
Combat actions - skill casting, targeting, attack patterns.
"""

import time
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

from .input import InputController
from src.config import get_config


class SkillSlot(Enum):
    """Skill hotkey slots."""
    F1 = "f1"
    F2 = "f2"
    F3 = "f3"
    F4 = "f4"
    F5 = "f5"
    F6 = "f6"
    F7 = "f7"
    F8 = "f8"


@dataclass
class SkillConfig:
    """Configuration for a skill."""
    name: str
    hotkey: SkillSlot
    is_right_click: bool = True  # Most attack skills use right-click
    cast_delay: float = 0.3  # Time to wait after casting
    is_aoe: bool = False  # Area of effect skill
    is_buff: bool = False  # Self-buff skill
    mana_cost: int = 0  # Approximate mana cost


class BlizzardSorc:
    """Combat controller for Blizzard Sorceress build."""

    def __init__(self, input_ctrl: Optional[InputController] = None):
        self.input = input_ctrl or InputController()
        self.config = get_config()

        # Skill configurations
        self.skills = {
            "blizzard": SkillConfig(
                name="Blizzard",
                hotkey=SkillSlot.F1,
                is_right_click=True,
                cast_delay=0.4,
                is_aoe=True,
                mana_cost=35
            ),
            "glacial_spike": SkillConfig(
                name="Glacial Spike",
                hotkey=SkillSlot.F2,
                is_right_click=True,
                cast_delay=0.2,
                is_aoe=False,
                mana_cost=10
            ),
            "ice_blast": SkillConfig(
                name="Ice Blast",
                hotkey=SkillSlot.F3,
                is_right_click=False,  # Left-click skill
                cast_delay=0.15,
                mana_cost=8
            ),
            "teleport": SkillConfig(
                name="Teleport",
                hotkey=SkillSlot.F4,
                is_right_click=True,
                cast_delay=0.1,
                mana_cost=20
            ),
            "frozen_armor": SkillConfig(
                name="Frozen Armor",
                hotkey=SkillSlot.F5,
                is_right_click=True,
                cast_delay=0.3,
                is_buff=True,
                mana_cost=5
            ),
            "static_field": SkillConfig(
                name="Static Field",
                hotkey=SkillSlot.F6,
                is_right_click=True,
                cast_delay=0.2,
                is_aoe=True,
                mana_cost=12
            ),
        }

        # Combat state
        self.current_right_skill: Optional[str] = None
        self.last_blizzard_time = 0
        self.blizzard_cooldown = 1.8  # Blizzard has internal cooldown

    def select_skill(self, skill_name: str) -> bool:
        """Select a skill for right-click use.

        Args:
            skill_name: Name of skill to select

        Returns:
            True if skill was selected
        """
        if skill_name not in self.skills:
            return False

        skill = self.skills[skill_name]
        self.input.press_key(skill.hotkey.value)
        self.current_right_skill = skill_name
        time.sleep(0.05)
        return True

    def cast_blizzard(self, x: int, y: int) -> bool:
        """Cast Blizzard at target location.

        Args:
            x, y: Target coordinates (relative to window)

        Returns:
            True if cast was attempted
        """
        current_time = time.time()

        # Check cooldown
        if current_time - self.last_blizzard_time < self.blizzard_cooldown:
            return False

        # Select and cast
        self.select_skill("blizzard")
        self.input.right_click(x, y, relative=True)

        self.last_blizzard_time = current_time
        time.sleep(self.skills["blizzard"].cast_delay)
        return True

    def cast_glacial_spike(self, x: int, y: int) -> bool:
        """Cast Glacial Spike at target (for freezing)."""
        self.select_skill("glacial_spike")
        self.input.right_click(x, y, relative=True)
        time.sleep(self.skills["glacial_spike"].cast_delay)
        return True

    def cast_static_field(self) -> bool:
        """Cast Static Field (for boss killing)."""
        self.select_skill("static_field")
        # Static field is centered on character, just right-click anywhere
        window = self.input.window_manager.get_window()
        if window:
            center_x = window.width // 2
            center_y = window.height // 2
            self.input.right_click(center_x, center_y, relative=True)
            time.sleep(self.skills["static_field"].cast_delay)
            return True
        return False

    def teleport(self, x: int, y: int) -> bool:
        """Teleport to target location.

        Args:
            x, y: Target coordinates (relative to window)

        Returns:
            True if teleport was attempted
        """
        self.select_skill("teleport")
        self.input.right_click(x, y, relative=True)
        time.sleep(self.config.timing.after_teleport)
        return True

    def cast_frozen_armor(self) -> bool:
        """Cast Frozen Armor buff on self."""
        self.select_skill("frozen_armor")
        # Self-buff, click on character
        window = self.input.window_manager.get_window()
        if window:
            center_x = window.width // 2
            center_y = window.height // 2
            self.input.right_click(center_x, center_y, relative=True)
            time.sleep(self.skills["frozen_armor"].cast_delay)
            return True
        return False

    def attack_target(
        self,
        x: int,
        y: int,
        use_static: bool = False,
        is_boss: bool = False
    ) -> None:
        """Execute attack rotation on a target.

        Args:
            x, y: Target coordinates
            use_static: Use static field first (for bosses)
            is_boss: Target is a boss (use boss rotation)
        """
        if is_boss and use_static:
            # Static field first to chunk boss HP
            for _ in range(3):
                self.cast_static_field()

        # Main damage - Blizzard
        if self.cast_blizzard(x, y):
            # Fill cooldown with glacial spike
            self.cast_glacial_spike(x, y)
            time.sleep(0.5)
            self.cast_glacial_spike(x, y)

    def kite_and_attack(
        self,
        enemy_x: int,
        enemy_y: int,
        kite_direction: str = "back"
    ) -> None:
        """Cast Blizzard then teleport away (kiting pattern).

        Args:
            enemy_x, enemy_y: Enemy position
            kite_direction: Direction to kite ("back", "left", "right")
        """
        window = self.input.window_manager.get_window()
        if not window:
            return

        # Cast Blizzard on enemy
        self.cast_blizzard(enemy_x, enemy_y)

        # Calculate kite position
        center_x = window.width // 2
        center_y = window.height // 2

        if kite_direction == "back":
            # Teleport away from enemy (opposite direction)
            kite_x = center_x + (center_x - enemy_x)
            kite_y = center_y + (center_y - enemy_y)
        elif kite_direction == "left":
            kite_x = center_x - 200
            kite_y = center_y
        else:  # right
            kite_x = center_x + 200
            kite_y = center_y

        # Clamp to screen bounds
        kite_x = max(50, min(window.width - 50, kite_x))
        kite_y = max(50, min(window.height - 50, kite_y))

        # Teleport to kite position
        self.teleport(kite_x, kite_y)


class CombatController:
    """Generic combat controller that delegates to build-specific controllers."""

    def __init__(self, input_ctrl: Optional[InputController] = None):
        self.input = input_ctrl or InputController()
        self.config = get_config()

        # Initialize build-specific controller based on config
        build_type = self.config.get("character.build", "blizzard")
        self.build = self._create_build(build_type)

    def _create_build(self, build_type: str):
        """Create the appropriate build controller.

        Args:
            build_type: Build name from config

        Returns:
            Build-specific combat controller
        """
        if build_type == "elemental_druid":
            from .builds import ElementalDruid
            return ElementalDruid(self.input)
        else:
            # Default to Blizzard Sorc
            return BlizzardSorc(self.input)

    def attack(self, x: int, y: int, is_boss: bool = False) -> None:
        """Attack a target."""
        self.build.attack_target(x, y, use_static=is_boss, is_boss=is_boss)

    def teleport(self, x: int, y: int) -> bool:
        """Teleport to location."""
        return self.build.teleport(x, y)

    def cast_buff(self) -> None:
        """Cast defensive buff."""
        self.build.cast_frozen_armor()

    def kite(self, enemy_x: int, enemy_y: int) -> None:
        """Kite away from enemy."""
        self.build.kite_and_attack(enemy_x, enemy_y)
