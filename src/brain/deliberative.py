"""
Deliberative brain - LLM-powered decisions for complex situations.

Handles:
- Item evaluation
- Error recovery
- Strategy optimization
- Inventory management
"""

import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

from .llm_client import LLMClient, get_llm_client, LLMResponse
from .prompts import (
    ITEM_EVALUATOR_SYSTEM, ITEM_EVALUATION_PROMPT,
    ERROR_RECOVERY_SYSTEM, ERROR_RECOVERY_PROMPT,
    STRATEGY_ADVISOR_SYSTEM, STRATEGY_PROMPT,
    INVENTORY_SYSTEM, INVENTORY_PROMPT,
    should_quick_pickup, should_quick_skip,
)
from src.config import get_config


class ItemValue(Enum):
    """Item value classification."""
    TRASH = "trash"
    CHARSI = "charsi"  # Vendor food
    SELF_USE = "self-use"
    TRADE_LOW = "trade-low"
    TRADE_MID = "trade-mid"
    TRADE_HIGH = "trade-high"
    GG = "gg"  # God-tier


@dataclass
class ItemEvaluation:
    """Result of LLM item evaluation."""
    item_name: str
    keep: bool
    reason: str
    value: ItemValue
    priority: int  # 1-10
    cached: bool = False
    latency_ms: float = 0


@dataclass
class RecoveryPlan:
    """Error recovery plan from LLM."""
    diagnosis: str
    severity: str
    actions: List[Dict[str, Any]]
    should_abort: bool


@dataclass
class StrategyAdvice:
    """Strategy recommendations from LLM."""
    assessment: str
    recommendations: List[Dict[str, Any]]
    suggested_run_order: List[str]
    config_changes: Dict[str, Any]


class DeliberativeBrain:
    """LLM-powered decision making for complex situations."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        """Initialize with LLM client."""
        self.config = get_config()
        self._client: Optional[LLMClient] = llm_client

        # Evaluation cache (in addition to LLM client cache)
        self._eval_cache: Dict[str, ItemEvaluation] = {}

        # Stats
        self.evaluations_made = 0
        self.quick_decisions = 0
        self.llm_calls = 0

    @property
    def client(self) -> LLMClient:
        """Lazy-load LLM client."""
        if self._client is None:
            self._client = get_llm_client()
        return self._client

    def evaluate_item(
        self,
        item_name: str,
        item_type: str = "unknown",
        rarity: str = "unknown",
        stats: str = "",
        use_cache: bool = True,
    ) -> ItemEvaluation:
        """Evaluate an item using the LLM.

        Args:
            item_name: Name of the item
            item_type: Type (armor, weapon, etc.)
            rarity: Rarity (unique, set, rare, magic)
            stats: Item stats text from OCR
            use_cache: Whether to use cached evaluations

        Returns:
            ItemEvaluation with keep/drop decision
        """
        self.evaluations_made += 1

        # Check quick rules first
        full_text = f"{item_name} {stats}".lower()

        if should_quick_pickup(full_text):
            self.quick_decisions += 1
            return ItemEvaluation(
                item_name=item_name,
                keep=True,
                reason="High-value item (quick rule)",
                value=ItemValue.TRADE_HIGH,
                priority=9,
            )

        if should_quick_skip(full_text):
            self.quick_decisions += 1
            return ItemEvaluation(
                item_name=item_name,
                keep=False,
                reason="Low-value item (quick rule)",
                value=ItemValue.TRASH,
                priority=1,
            )

        # Check evaluation cache
        cache_key = f"{item_name}:{rarity}:{stats[:100]}"
        if use_cache and cache_key in self._eval_cache:
            cached = self._eval_cache[cache_key]
            cached.cached = True
            return cached

        # Use LLM for evaluation
        self.llm_calls += 1

        prompt = ITEM_EVALUATION_PROMPT.format(
            item_name=item_name,
            item_type=item_type,
            rarity=rarity,
            stats=stats or "No stats visible",
        )

        start_time = time.time()
        response = self.client.complete_json(
            prompt=prompt,
            system_prompt=ITEM_EVALUATOR_SYSTEM,
            max_tokens=300,
            use_cache=use_cache,
        )
        latency = (time.time() - start_time) * 1000

        # Parse response
        keep = response.get("keep", False)
        reason = response.get("reason", "No reason provided")
        value_str = response.get("value", "trash")
        priority = response.get("priority", 5)

        # Convert value string to enum
        try:
            value = ItemValue(value_str)
        except ValueError:
            value = ItemValue.TRASH

        evaluation = ItemEvaluation(
            item_name=item_name,
            keep=keep,
            reason=reason,
            value=value,
            priority=priority,
            latency_ms=latency,
        )

        # Cache evaluation
        if use_cache:
            self._eval_cache[cache_key] = evaluation

        return evaluation

    def get_recovery_plan(
        self,
        health_percent: float,
        mana_percent: float,
        location: str,
        last_actions: List[str],
        current_state: str,
        error_description: str,
        screen_description: str = "",
    ) -> RecoveryPlan:
        """Get an error recovery plan from the LLM.

        Args:
            health_percent: Current health (0-100)
            mana_percent: Current mana (0-100)
            location: Current location description
            last_actions: List of recent actions
            current_state: Current bot state
            error_description: What went wrong
            screen_description: Description of what's on screen

        Returns:
            RecoveryPlan with actions to take
        """
        prompt = ERROR_RECOVERY_PROMPT.format(
            health_percent=int(health_percent * 100),
            mana_percent=int(mana_percent * 100),
            location=location,
            last_actions=", ".join(last_actions[-5:]),
            current_state=current_state,
            error_description=error_description,
            screen_description=screen_description or "No description available",
        )

        response = self.client.complete_json(
            prompt=prompt,
            system_prompt=ERROR_RECOVERY_SYSTEM,
            max_tokens=500,
            use_cache=False,  # Don't cache error recovery
        )

        return RecoveryPlan(
            diagnosis=response.get("diagnosis", "Unknown issue"),
            severity=response.get("severity", "medium"),
            actions=response.get("actions", []),
            should_abort=response.get("should_abort_run", False),
        )

    def get_strategy_advice(
        self,
        total_runs: int,
        success_rate: float,
        avg_run_time: float,
        total_deaths: int,
        items_summary: str,
        enabled_runs: List[str],
        health_threshold: float,
        mana_threshold: float,
        recent_issues: str = "",
    ) -> StrategyAdvice:
        """Get strategy optimization advice.

        Args:
            total_runs: Total runs completed
            success_rate: Success rate (0-1)
            avg_run_time: Average run time in seconds
            total_deaths: Total deaths
            items_summary: Summary of items found
            enabled_runs: Currently enabled runs
            health_threshold: Health potion threshold
            mana_threshold: Mana potion threshold
            recent_issues: Description of recent problems

        Returns:
            StrategyAdvice with recommendations
        """
        prompt = STRATEGY_PROMPT.format(
            total_runs=total_runs,
            success_rate=f"{success_rate:.1%}",
            avg_run_time=f"{avg_run_time:.1f}s",
            total_deaths=total_deaths,
            items_summary=items_summary,
            enabled_runs=", ".join(enabled_runs),
            health_threshold=int(health_threshold * 100),
            mana_threshold=int(mana_threshold * 100),
            recent_issues=recent_issues or "None reported",
        )

        response = self.client.complete_json(
            prompt=prompt,
            system_prompt=STRATEGY_ADVISOR_SYSTEM,
            max_tokens=600,
            use_cache=False,
        )

        return StrategyAdvice(
            assessment=response.get("assessment", "No assessment"),
            recommendations=response.get("recommendations", []),
            suggested_run_order=response.get("suggested_run_order", enabled_runs),
            config_changes=response.get("config_changes", {}),
        )

    def manage_inventory(
        self,
        inventory_items: List[Dict[str, str]],
        ground_items: List[Dict[str, str]],
        stash_slots_free: int,
        shared_stash_free: int,
    ) -> Dict[str, Any]:
        """Get inventory management decisions.

        Args:
            inventory_items: List of items in inventory
            ground_items: List of items on ground
            stash_slots_free: Free slots in personal stash
            shared_stash_free: Free slots in shared stash

        Returns:
            Dict with inventory_actions and ground_pickups
        """
        # Format inventory items
        inv_str = "\n".join(
            f"- {item.get('name', 'Unknown')}: {item.get('rarity', '?')} "
            f"({item.get('stats', 'no stats')})"
            for item in inventory_items
        )

        # Format ground items
        ground_str = "\n".join(
            f"- {item.get('name', 'Unknown')}: {item.get('rarity', '?')}"
            for item in ground_items
        ) or "None"

        prompt = INVENTORY_PROMPT.format(
            inventory_items=inv_str or "Empty",
            stash_slots_free=stash_slots_free,
            shared_stash_free=shared_stash_free,
            ground_items=ground_str,
        )

        return self.client.complete_json(
            prompt=prompt,
            system_prompt=INVENTORY_SYSTEM,
            max_tokens=800,
            use_cache=False,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        llm_stats = self.client.get_stats() if self._client else {}

        return {
            "evaluations_made": self.evaluations_made,
            "quick_decisions": self.quick_decisions,
            "llm_calls": self.llm_calls,
            "quick_decision_rate": f"{self.quick_decisions / max(1, self.evaluations_made):.1%}",
            "eval_cache_size": len(self._eval_cache),
            **llm_stats,
        }


# Singleton instance
_brain: Optional[DeliberativeBrain] = None


def get_deliberative_brain() -> DeliberativeBrain:
    """Get the global deliberative brain instance."""
    global _brain
    if _brain is None:
        _brain = DeliberativeBrain()
    return _brain
