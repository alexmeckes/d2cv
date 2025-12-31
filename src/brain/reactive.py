"""
Reactive brain - fast heuristic-based decisions for real-time gameplay.

This module handles time-critical decisions that don't need LLM reasoning:
- Potion usage
- Combat targeting
- Emergency escape
- Basic movement
"""

import time
from typing import Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

from src.config import get_config
from src.vision.game_state import VitalsState, BeltState
from src.vision.entities import DetectedItem, DetectedEnemy, ItemRarity


class Action(Enum):
    """Actions the reactive brain can decide to take."""
    NONE = "none"
    USE_HEALTH_POTION = "use_health_potion"
    USE_MANA_POTION = "use_mana_potion"
    USE_REJUV_POTION = "use_rejuv_potion"
    CHICKEN = "chicken"  # Emergency town portal
    HEAL_MERC = "heal_merc"
    ATTACK = "attack"
    TELEPORT = "teleport"
    PICK_UP_ITEM = "pick_up_item"
    CAST_BUFF = "cast_buff"


@dataclass
class ActionDecision:
    """A decision made by the reactive brain."""
    action: Action
    priority: int  # Higher = more urgent
    target: Optional[Tuple[int, int]] = None  # (x, y) for targeted actions
    details: dict = field(default_factory=dict)

    def __lt__(self, other):
        return self.priority > other.priority  # Higher priority first


class ReactiveBrain:
    """Makes fast, heuristic-based decisions for gameplay."""

    # Priority levels
    PRIORITY_EMERGENCY = 100
    PRIORITY_SURVIVAL = 80
    PRIORITY_COMBAT = 50
    PRIORITY_LOOT = 30
    PRIORITY_BUFF = 10

    def __init__(self):
        self.config = get_config()

        # Cooldowns to prevent action spam
        self.last_health_potion = 0
        self.last_mana_potion = 0
        self.last_merc_heal = 0
        self.last_attack = 0

        # Cooldown durations (seconds)
        self.potion_cooldown = 1.0
        self.merc_heal_cooldown = 2.0
        self.attack_cooldown = 0.1

        # State tracking
        self.consecutive_low_health = 0
        self.last_vitals: Optional[VitalsState] = None

    def evaluate(
        self,
        vitals: VitalsState,
        belt: BeltState,
        enemies: List[DetectedEnemy],
        items: List[DetectedItem],
        current_time: Optional[float] = None
    ) -> List[ActionDecision]:
        """Evaluate current game state and return prioritized actions.

        Args:
            vitals: Current health/mana state
            belt: Belt potion state
            enemies: Detected enemies on screen
            items: Detected items on ground
            current_time: Current timestamp (for cooldowns)

        Returns:
            List of actions sorted by priority (highest first)
        """
        if current_time is None:
            current_time = time.time()

        decisions = []

        # Check survival first
        survival_decisions = self._evaluate_survival(vitals, belt, current_time)
        decisions.extend(survival_decisions)

        # Check merc health
        merc_decision = self._evaluate_merc(vitals, belt, current_time)
        if merc_decision:
            decisions.append(merc_decision)

        # Combat decisions (only if not in emergency)
        if not any(d.action == Action.CHICKEN for d in decisions):
            combat_decisions = self._evaluate_combat(enemies, vitals)
            decisions.extend(combat_decisions)

        # Loot decisions (only if safe)
        if vitals.health_percent > 0.5 and not enemies:
            loot_decisions = self._evaluate_loot(items)
            decisions.extend(loot_decisions)

        # Sort by priority
        decisions.sort()

        self.last_vitals = vitals
        return decisions

    def _evaluate_survival(
        self,
        vitals: VitalsState,
        belt: BeltState,
        current_time: float
    ) -> List[ActionDecision]:
        """Evaluate survival-related decisions."""
        decisions = []

        # Emergency chicken (very low health)
        if vitals.health_percent <= self.config.thresholds.chicken:
            self.consecutive_low_health += 1
            if self.consecutive_low_health >= 2:  # Confirm low health
                decisions.append(ActionDecision(
                    action=Action.CHICKEN,
                    priority=self.PRIORITY_EMERGENCY,
                    details={"reason": f"Health critical: {vitals.health_percent:.0%}"}
                ))
                return decisions  # Don't add other actions if chickening
        else:
            self.consecutive_low_health = 0

        # Health potion
        if vitals.health_percent <= self.config.thresholds.health_potion:
            if current_time - self.last_health_potion >= self.potion_cooldown:
                if belt.total_health_potions > 0:
                    # Determine which slot to use
                    slot = 1 if belt.slot_1_count > 0 else 2
                    decisions.append(ActionDecision(
                        action=Action.USE_HEALTH_POTION,
                        priority=self.PRIORITY_SURVIVAL,
                        details={"slot": slot, "health": vitals.health_percent}
                    ))

        # Mana potion
        if vitals.mana_percent <= self.config.thresholds.mana_potion:
            if current_time - self.last_mana_potion >= self.potion_cooldown:
                if belt.total_mana_potions > 0:
                    slot = 3 if belt.slot_3_count > 0 else 4
                    decisions.append(ActionDecision(
                        action=Action.USE_MANA_POTION,
                        priority=self.PRIORITY_SURVIVAL - 10,  # Slightly lower than health
                        details={"slot": slot, "mana": vitals.mana_percent}
                    ))

        return decisions

    def _evaluate_merc(
        self,
        vitals: VitalsState,
        belt: BeltState,
        current_time: float
    ) -> Optional[ActionDecision]:
        """Evaluate mercenary healing."""
        if vitals.merc_health_percent is None:
            return None

        if vitals.merc_health_percent <= self.config.thresholds.merc_health_potion:
            if current_time - self.last_merc_heal >= self.merc_heal_cooldown:
                if belt.total_health_potions > 0:
                    return ActionDecision(
                        action=Action.HEAL_MERC,
                        priority=self.PRIORITY_COMBAT - 5,
                        details={"merc_health": vitals.merc_health_percent}
                    )

        return None

    def _evaluate_combat(
        self,
        enemies: List[DetectedEnemy],
        vitals: VitalsState
    ) -> List[ActionDecision]:
        """Evaluate combat decisions."""
        decisions = []

        if not enemies:
            return decisions

        # Find best target (prioritize bosses, then closest)
        best_target = None
        best_priority = 0

        for enemy in enemies:
            priority = self.PRIORITY_COMBAT
            if enemy.is_boss:
                priority += 20
            if enemy.is_unique:
                priority += 10

            if priority > best_priority:
                best_priority = priority
                best_target = enemy

        if best_target:
            # Only attack if we have mana
            if vitals.mana_percent > 0.1:
                decisions.append(ActionDecision(
                    action=Action.ATTACK,
                    priority=best_priority,
                    target=best_target.center,
                    details={
                        "enemy": best_target.name or "Enemy",
                        "is_boss": best_target.is_boss
                    }
                ))

        return decisions

    def _evaluate_loot(self, items: List[DetectedItem]) -> List[ActionDecision]:
        """Evaluate item pickup decisions."""
        decisions = []

        # Prioritize by rarity
        rarity_priority = {
            ItemRarity.UNIQUE: 10,
            ItemRarity.SET: 9,
            ItemRarity.RARE: 7,
            ItemRarity.RUNE: 10,  # Runes are high priority
            ItemRarity.MAGIC: 3,
            ItemRarity.GOLD: 1,
            ItemRarity.NORMAL: 0,
        }

        for item in items:
            priority = self.PRIORITY_LOOT + rarity_priority.get(item.rarity, 0)

            # Skip low-value items
            if item.rarity in (ItemRarity.NORMAL, ItemRarity.MAGIC):
                continue

            decisions.append(ActionDecision(
                action=Action.PICK_UP_ITEM,
                priority=priority,
                target=item.click_point,
                details={"rarity": item.rarity.value, "name": item.name}
            ))

        return decisions

    def record_action(self, action: Action, current_time: Optional[float] = None):
        """Record that an action was taken (for cooldown tracking)."""
        if current_time is None:
            current_time = time.time()

        if action == Action.USE_HEALTH_POTION:
            self.last_health_potion = current_time
        elif action == Action.USE_MANA_POTION:
            self.last_mana_potion = current_time
        elif action == Action.HEAL_MERC:
            self.last_merc_heal = current_time
        elif action == Action.ATTACK:
            self.last_attack = current_time

    def should_chicken(self, vitals: VitalsState) -> bool:
        """Quick check if we should immediately chicken."""
        return vitals.health_percent <= self.config.thresholds.chicken

    def get_potion_slot(self, potion_type: str, belt: BeltState) -> Optional[int]:
        """Get the belt slot to use for a potion type."""
        if potion_type == "health":
            if belt.slot_1_count > 0:
                return 1
            if belt.slot_2_count > 0:
                return 2
        elif potion_type == "mana":
            if belt.slot_3_count > 0:
                return 3
            if belt.slot_4_count > 0:
                return 4
        return None
