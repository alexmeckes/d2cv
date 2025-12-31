"""
Elemental Druid build - Volcano + Armageddon fire build.

Playstyle:
- Keep Armageddon active (auto-meteors around character)
- Summon Oak Sage for HP buffer
- Cast Volcano on enemy packs (main damage)
- Fill with Fissure for additional ground damage
- Position IN CENTER of packs (Armageddon benefits from proximity)

Key differences from Blizzard Sorc:
- Druid wants to be CLOSER to enemies
- No kiting needed - Oak Sage provides survivability
- Fire build = skip fire immunes
"""

import time
from typing import Optional
from dataclasses import dataclass

from ..input import InputController
from src.config import get_config


class SkillSlot:
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
    hotkey: str
    is_right_click: bool = True
    cast_delay: float = 0.3
    is_aoe: bool = False
    is_buff: bool = False
    mana_cost: int = 0
    duration: float = 0  # For buffs with duration


class ElementalDruid:
    """Combat controller for Volcano + Armageddon Druid build."""

    def __init__(self, input_ctrl: Optional[InputController] = None):
        self.input = input_ctrl or InputController()
        self.config = get_config()

        # Skill configurations
        self.skills = {
            "volcano": SkillConfig(
                name="Volcano",
                hotkey=SkillSlot.F1,
                is_right_click=True,
                cast_delay=0.5,
                is_aoe=True,
                mana_cost=30
            ),
            "armageddon": SkillConfig(
                name="Armageddon",
                hotkey=SkillSlot.F2,
                is_right_click=True,
                cast_delay=0.3,
                is_buff=True,
                mana_cost=35,
                duration=12.0  # ~12 seconds duration
            ),
            "fissure": SkillConfig(
                name="Fissure",
                hotkey=SkillSlot.F3,
                is_right_click=True,
                cast_delay=0.4,
                is_aoe=True,
                mana_cost=20
            ),
            "molten_boulder": SkillConfig(
                name="Molten Boulder",
                hotkey=SkillSlot.F4,
                is_right_click=True,
                cast_delay=0.3,
                is_aoe=True,
                mana_cost=15
            ),
            "oak_sage": SkillConfig(
                name="Oak Sage",
                hotkey=SkillSlot.F5,
                is_right_click=True,
                cast_delay=0.3,
                is_buff=True,
                mana_cost=15,
                duration=0  # Permanent until killed
            ),
            "teleport": SkillConfig(
                name="Teleport",
                hotkey=SkillSlot.F6,
                is_right_click=True,
                cast_delay=0.1,
                mana_cost=20
            ),
            "cyclone_armor": SkillConfig(
                name="Cyclone Armor",
                hotkey=SkillSlot.F7,
                is_right_click=True,
                cast_delay=0.2,
                is_buff=True,
                mana_cost=10
            ),
        }

        # Combat state
        self.current_right_skill: Optional[str] = None
        self.last_armageddon_time = 0
        self.armageddon_duration = 12.0
        self.last_volcano_time = 0
        self.volcano_cooldown = 1.0  # Volcano has short cooldown
        self.oak_sage_active = False

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
        self.input.press_key(skill.hotkey)
        self.current_right_skill = skill_name
        time.sleep(0.05)
        return True

    def cast_volcano(self, x: int, y: int) -> bool:
        """Cast Volcano at target location.

        Args:
            x, y: Target coordinates (relative to window)

        Returns:
            True if cast was attempted
        """
        current_time = time.time()

        # Check cooldown
        if current_time - self.last_volcano_time < self.volcano_cooldown:
            return False

        # Select and cast
        self.select_skill("volcano")
        self.input.right_click(x, y, relative=True)

        self.last_volcano_time = current_time
        time.sleep(self.skills["volcano"].cast_delay)
        return True

    def cast_fissure(self, x: int, y: int) -> bool:
        """Cast Fissure at target location."""
        self.select_skill("fissure")
        self.input.right_click(x, y, relative=True)
        time.sleep(self.skills["fissure"].cast_delay)
        return True

    def cast_molten_boulder(self, x: int, y: int) -> bool:
        """Cast Molten Boulder towards target."""
        self.select_skill("molten_boulder")
        self.input.right_click(x, y, relative=True)
        time.sleep(self.skills["molten_boulder"].cast_delay)
        return True

    def cast_armageddon(self) -> bool:
        """Cast Armageddon buff on self.

        Returns:
            True if cast (or already active)
        """
        current_time = time.time()

        # Check if still active
        if current_time - self.last_armageddon_time < self.armageddon_duration:
            return True  # Still active, no need to recast

        self.select_skill("armageddon")
        window = self.input.window_manager.get_window()
        if window:
            center_x = window.width // 2
            center_y = window.height // 2
            self.input.right_click(center_x, center_y, relative=True)
            self.last_armageddon_time = current_time
            time.sleep(self.skills["armageddon"].cast_delay)
            return True
        return False

    def cast_oak_sage(self) -> bool:
        """Summon Oak Sage spirit.

        Only casts if not currently active.
        """
        if self.oak_sage_active:
            return True

        self.select_skill("oak_sage")
        window = self.input.window_manager.get_window()
        if window:
            center_x = window.width // 2
            center_y = window.height // 2
            self.input.right_click(center_x, center_y, relative=True)
            self.oak_sage_active = True
            time.sleep(self.skills["oak_sage"].cast_delay)
            return True
        return False

    def cast_cyclone_armor(self) -> bool:
        """Cast Cyclone Armor defensive buff."""
        self.select_skill("cyclone_armor")
        window = self.input.window_manager.get_window()
        if window:
            center_x = window.width // 2
            center_y = window.height // 2
            self.input.right_click(center_x, center_y, relative=True)
            time.sleep(self.skills["cyclone_armor"].cast_delay)
            return True
        return False

    def teleport(self, x: int, y: int) -> bool:
        """Teleport to target location (requires Enigma).

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
        """Cast defensive buff - for Druid this is Cyclone Armor."""
        return self.cast_cyclone_armor()

    def ensure_buffs(self) -> None:
        """Ensure all buffs are active."""
        self.cast_armageddon()
        self.cast_oak_sage()
        self.cast_cyclone_armor()

    def attack_target(
        self,
        x: int,
        y: int,
        use_static: bool = False,
        is_boss: bool = False
    ) -> None:
        """Execute attack rotation on a target.

        For Elemental Druid:
        1. Ensure Armageddon is active
        2. Cast Volcano on target
        3. Fill with Fissure

        Args:
            x, y: Target coordinates
            use_static: Ignored for Druid (no static field)
            is_boss: Target is a boss (longer rotation)
        """
        # Ensure Armageddon is running
        self.cast_armageddon()

        # Main damage - Volcano
        if self.cast_volcano(x, y):
            # Fill with Fissure
            self.cast_fissure(x, y)
            time.sleep(0.3)
            self.cast_fissure(x, y)

        if is_boss:
            # Extended rotation for bosses
            self.cast_molten_boulder(x, y)
            time.sleep(0.3)
            self.cast_volcano(x, y)

    def aggressive_attack(self, x: int, y: int) -> None:
        """Aggressive attack pattern - teleport into pack and nuke.

        Elemental Druid benefits from being close due to Armageddon.
        """
        window = self.input.window_manager.get_window()
        if not window:
            return

        # Ensure buffs
        self.cast_armageddon()

        # Teleport closer to enemies (Armageddon range)
        center_x = window.width // 2
        center_y = window.height // 2

        # Move towards enemy but not on top
        approach_x = center_x + (x - center_x) // 2
        approach_y = center_y + (y - center_y) // 2

        self.teleport(approach_x, approach_y)

        # Now cast ground spells
        self.cast_volcano(x, y)
        self.cast_fissure(x, y)

    def kite_and_attack(
        self,
        enemy_x: int,
        enemy_y: int,
        kite_direction: str = "back"
    ) -> None:
        """Cast spells then reposition.

        Note: Druid doesn't need to kite as much due to Oak Sage HP boost,
        but this is available for dangerous situations.
        """
        window = self.input.window_manager.get_window()
        if not window:
            return

        # Cast Volcano on enemy first
        self.cast_volcano(enemy_x, enemy_y)

        # Only kite if low HP or dangerous situation
        # For now, just cast more spells
        self.cast_fissure(enemy_x, enemy_y)

    def cast_static_field(self) -> bool:
        """Druid doesn't have Static Field - cast Volcano instead."""
        window = self.input.window_manager.get_window()
        if window:
            center_x = window.width // 2
            center_y = window.height // 2
            return self.cast_volcano(center_x, center_y - 50)
        return False

    def cast_blizzard(self, x: int, y: int) -> bool:
        """Compatibility method - Druid uses Volcano instead."""
        return self.cast_volcano(x, y)

    def cast_glacial_spike(self, x: int, y: int) -> bool:
        """Compatibility method - Druid uses Fissure instead."""
        return self.cast_fissure(x, y)

    def reset_oak_sage(self) -> None:
        """Reset Oak Sage status (call if sage died)."""
        self.oak_sage_active = False
