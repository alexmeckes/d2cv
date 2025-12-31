"""
Town actions - portal, repair, heal, stash, vendors.
"""

import time
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .input import InputController
from .movement import MovementController
from src.config import get_config


class TownAct(Enum):
    """Town acts with their waypoint/NPC layouts."""
    ACT1 = 1  # Rogue Encampment
    ACT2 = 2  # Lut Gholein
    ACT3 = 3  # Kurast Docks
    ACT4 = 4  # Pandemonium Fortress
    ACT5 = 5  # Harrogath


@dataclass
class NPCLocation:
    """Screen position of an NPC in town."""
    name: str
    x: int
    y: int
    act: TownAct


# Approximate NPC positions for 1280x720 resolution
# These need calibration for actual gameplay
NPC_POSITIONS = {
    TownAct.ACT3: {
        "ormus": NPCLocation("Ormus", 640, 300, TownAct.ACT3),  # Healer/magic vendor
        "hratli": NPCLocation("Hratli", 400, 350, TownAct.ACT3),  # Repair
        "alkor": NPCLocation("Alkor", 800, 400, TownAct.ACT3),  # Potions
        "stash": NPCLocation("Stash", 500, 250, TownAct.ACT3),
        "waypoint": NPCLocation("Waypoint", 640, 500, TownAct.ACT3),
    },
    TownAct.ACT1: {
        "akara": NPCLocation("Akara", 300, 300, TownAct.ACT1),  # Healer/magic
        "charsi": NPCLocation("Charsi", 700, 350, TownAct.ACT1),  # Repair
        "stash": NPCLocation("Stash", 500, 400, TownAct.ACT1),
        "waypoint": NPCLocation("Waypoint", 640, 300, TownAct.ACT1),
    },
    TownAct.ACT2: {
        "fara": NPCLocation("Fara", 350, 350, TownAct.ACT2),  # Repair
        "lysander": NPCLocation("Lysander", 600, 300, TownAct.ACT2),  # Potions
        "stash": NPCLocation("Stash", 450, 400, TownAct.ACT2),
        "waypoint": NPCLocation("Waypoint", 640, 350, TownAct.ACT2),
    },
}


class TownController:
    """Handles town-related actions."""

    def __init__(
        self,
        input_ctrl: Optional[InputController] = None,
        movement_ctrl: Optional[MovementController] = None
    ):
        self.input = input_ctrl or InputController()
        self.movement = movement_ctrl or MovementController(self.input)
        self.config = get_config()

        # Town portal hotkey (default)
        self.tp_key = "t"

        # Current act (detected or set)
        self.current_act: Optional[TownAct] = None

    def cast_town_portal(self) -> bool:
        """Cast a town portal.

        Returns:
            True if portal was cast
        """
        self.input.press_key(self.tp_key)
        time.sleep(0.5)  # Portal cast animation
        return True

    def use_town_portal(self, portal_x: int, portal_y: int) -> bool:
        """Click on an existing town portal to enter it.

        Args:
            portal_x, portal_y: Screen position of portal

        Returns:
            True if clicked
        """
        self.input.click(portal_x, portal_y, relative=True)
        time.sleep(1.5)  # Loading screen
        return True

    def chicken(self) -> bool:
        """Emergency exit - save and quit.

        This is faster than town portal when in danger.
        """
        # ESC to open menu, then Save & Exit
        self.input.press_key("escape")
        time.sleep(0.3)

        # Click Save & Exit button (approximate position)
        # This needs calibration
        window = self.input.window_manager.get_window()
        if window:
            # Save & Exit is typically in the menu
            self.input.click(window.width // 2, window.height // 2 + 50, relative=True)
            time.sleep(0.5)
            return True

        return False

    def interact_with_npc(self, npc_name: str) -> bool:
        """Click on an NPC to interact.

        Args:
            npc_name: Name of NPC to interact with

        Returns:
            True if interaction attempted
        """
        if self.current_act is None:
            return False

        positions = NPC_POSITIONS.get(self.current_act, {})
        npc = positions.get(npc_name.lower())

        if npc is None:
            return False

        # Move near NPC first
        self.movement.teleport_towards(npc.x, npc.y)
        time.sleep(0.2)

        # Click to interact
        self.input.click(npc.x, npc.y, relative=True)
        time.sleep(0.5)  # Wait for dialog

        return True

    def open_stash(self) -> bool:
        """Open the stash."""
        return self.interact_with_npc("stash")

    def close_ui(self) -> None:
        """Close any open UI (inventory, stash, vendor)."""
        self.input.press_key("escape")
        time.sleep(0.2)

    def use_waypoint(self, destination: Optional[str] = None) -> bool:
        """Interact with waypoint.

        Args:
            destination: Waypoint destination to select (None = just open)

        Returns:
            True if waypoint interaction started
        """
        if not self.interact_with_npc("waypoint"):
            return False

        time.sleep(0.5)  # Wait for waypoint UI

        # TODO: Select specific destination from waypoint menu
        # This requires template matching or fixed positions

        return True

    def repair_items(self) -> bool:
        """Go to repair NPC and repair all items."""
        if self.current_act is None:
            return False

        # Get repair NPC for current act
        repair_npcs = {
            TownAct.ACT1: "charsi",
            TownAct.ACT2: "fara",
            TownAct.ACT3: "hratli",
            TownAct.ACT4: "halbu",  # Not in NPC_POSITIONS yet
            TownAct.ACT5: "larzuk",
        }

        npc_name = repair_npcs.get(self.current_act)
        if not npc_name:
            return False

        if not self.interact_with_npc(npc_name):
            return False

        # Click "Repair" button in trade window
        # Position varies by NPC, approximate for now
        window = self.input.window_manager.get_window()
        if window:
            # Repair button is typically bottom-left of trade window
            repair_x = 200
            repair_y = window.height - 150
            self.input.click(repair_x, repair_y, relative=True)
            time.sleep(0.3)

        self.close_ui()
        return True

    def buy_potions(self, health: int = 0, mana: int = 0) -> bool:
        """Buy potions from vendor.

        Args:
            health: Number of health potions to buy
            mana: Number of mana potions to buy

        Returns:
            True if purchase attempted
        """
        if self.current_act is None:
            return False

        # Get potion vendor for current act
        potion_npcs = {
            TownAct.ACT1: "akara",
            TownAct.ACT2: "lysander",
            TownAct.ACT3: "alkor",
        }

        npc_name = potion_npcs.get(self.current_act)
        if not npc_name:
            return False

        if not self.interact_with_npc(npc_name):
            return False

        # TODO: Navigate vendor UI to buy specific potions
        # This requires detecting potion positions in shop

        self.close_ui()
        return True

    def heal_at_healer(self) -> bool:
        """Visit healer to restore HP/mana (free).

        Note: In D2, talking to certain NPCs heals you.
        """
        healer_npcs = {
            TownAct.ACT1: "akara",
            TownAct.ACT2: "fara",
            TownAct.ACT3: "ormus",
        }

        if self.current_act is None:
            return False

        npc_name = healer_npcs.get(self.current_act)
        if not npc_name:
            return False

        self.interact_with_npc(npc_name)
        self.close_ui()
        return True

    def identify_items(self) -> bool:
        """Identify all items with Cain.

        Requires Cain to be rescued.
        """
        if not self.interact_with_npc("cain"):
            return False

        # Click "Identify Items" option
        # Position varies, approximate
        window = self.input.window_manager.get_window()
        if window:
            self.input.click(window.width // 2, window.height // 2, relative=True)
            time.sleep(0.5)

        self.close_ui()
        return True

    def full_town_routine(self) -> None:
        """Execute full town routine: heal, repair, stash, identify."""
        # Heal first (free)
        self.heal_at_healer()

        # Repair items
        self.repair_items()

        # Open stash and deposit items
        # TODO: Implement item stashing logic
        # self.open_stash()
        # self.close_ui()

        # Identify items with Cain
        # self.identify_items()
