"""
Item evaluator - uses Gemini Vision for intelligent item decisions.

Evaluation tiers:
1. Quick rules (instant, free) - rarity-based, known items
2. Gemini Vision (fast, cheap) - screenshot analysis
3. Text LLM fallback - if Gemini unavailable
"""

import time
from typing import Optional, List, Tuple
from dataclasses import dataclass
import numpy as np

from .deliberative import DeliberativeBrain, ItemEvaluation, ItemValue, get_deliberative_brain
from .prompts import should_quick_pickup, should_quick_skip
from src.vision.entities import DetectedItem, ItemRarity
from src.config import get_config
from src.state.session_logger import get_logger

# Lazy import to avoid circular deps
_gemini_client = None


def _get_gemini():
    """Lazy load Gemini client."""
    global _gemini_client
    if _gemini_client is None:
        try:
            from .gemini_vision import get_gemini_vision
            _gemini_client = get_gemini_vision()
        except Exception as e:
            get_logger("item_evaluator").warning(f"Gemini not available: {e}")
            _gemini_client = False  # Mark as unavailable
    return _gemini_client if _gemini_client else None


@dataclass
class EvaluatedItem:
    """An item that has been fully evaluated."""
    detected: DetectedItem
    name: Optional[str]
    stats: Optional[str]
    evaluation: ItemEvaluation
    should_pickup: bool
    evaluation_method: str = "unknown"  # "quick", "gemini", "llm"


class ItemEvaluator:
    """Evaluates items using Gemini Vision and quick rules."""

    def __init__(
        self,
        ocr=None,  # Kept for compatibility, but not used with Gemini
        brain: Optional[DeliberativeBrain] = None,
        use_llm: bool = True,
        use_gemini: bool = True,
        character_class: str = "sorceress",
        build: str = "blizzard",
    ):
        """Initialize the item evaluator.

        Args:
            ocr: OCR engine (deprecated, kept for compatibility)
            brain: Deliberative brain for text LLM fallback
            use_llm: Whether to use text LLM as fallback
            use_gemini: Whether to use Gemini Vision (recommended)
            character_class: Character class for value assessment
            build: Build type for value assessment
        """
        self._brain = brain
        self.use_llm = use_llm
        self.use_gemini = use_gemini
        self.character_class = character_class
        self.build = build
        self.config = get_config()
        self.logger = get_logger("item_evaluator")

        # Quick evaluation rules by rarity
        self.rarity_rules = {
            ItemRarity.UNIQUE: {"always_pickup": True, "min_priority": 8},
            ItemRarity.SET: {"always_pickup": True, "min_priority": 7},
            ItemRarity.RUNE: {"always_pickup": True, "min_priority": 9},
            ItemRarity.RARE: {"always_pickup": False, "min_priority": 5},
            ItemRarity.MAGIC: {"always_pickup": False, "min_priority": 3},
            ItemRarity.GOLD: {"always_pickup": True, "min_priority": 2},
            ItemRarity.NORMAL: {"always_pickup": False, "min_priority": 1},
        }

        # Stats
        self.items_evaluated = 0
        self.quick_evaluations = 0
        self.gemini_evaluations = 0
        self.llm_evaluations = 0

    @property
    def brain(self) -> DeliberativeBrain:
        if self._brain is None:
            self._brain = get_deliberative_brain()
        return self._brain

    def evaluate_item(
        self,
        item: DetectedItem,
        screenshot: Optional[np.ndarray] = None,
    ) -> EvaluatedItem:
        """Evaluate a single detected item.

        Args:
            item: Detected item from vision system
            screenshot: Full screenshot for Gemini analysis

        Returns:
            EvaluatedItem with full evaluation
        """
        self.items_evaluated += 1

        # Tier 1: Quick rules based on rarity
        rules = self.rarity_rules.get(
            item.rarity,
            {"always_pickup": False, "min_priority": 1}
        )

        # Always pickup high-value rarities
        if rules["always_pickup"] and item.rarity != ItemRarity.GOLD:
            self.quick_evaluations += 1
            evaluation = ItemEvaluation(
                item_name=item.name or item.rarity.value,
                keep=True,
                reason=f"Always keep {item.rarity.value} items",
                value=self._rarity_to_value(item.rarity),
                priority=rules["min_priority"],
            )
            return EvaluatedItem(
                detected=item,
                name=item.name,
                stats=None,
                evaluation=evaluation,
                should_pickup=True,
                evaluation_method="quick",
            )

        # Quick skip for normal/low magic
        if item.rarity in (ItemRarity.NORMAL, ItemRarity.MAGIC):
            self.quick_evaluations += 1
            evaluation = ItemEvaluation(
                item_name=item.name or item.rarity.value,
                keep=False,
                reason=f"Skip {item.rarity.value} items",
                value=ItemValue.TRASH,
                priority=1,
            )
            return EvaluatedItem(
                detected=item,
                name=item.name,
                stats=None,
                evaluation=evaluation,
                should_pickup=False,
                evaluation_method="quick",
            )

        # Tier 2: Gemini Vision for uncertain items
        if self.use_gemini and screenshot is not None:
            gemini = _get_gemini()
            if gemini:
                return self._evaluate_with_gemini(item, screenshot)

        # Tier 3: Text LLM fallback
        if self.use_llm and item.rarity == ItemRarity.RARE:
            return self._evaluate_with_llm(item)

        # Default: skip
        evaluation = ItemEvaluation(
            item_name=item.name or item.rarity.value,
            keep=False,
            reason="Default: skip uncertain item",
            value=ItemValue.TRASH,
            priority=1,
        )
        return EvaluatedItem(
            detected=item,
            name=item.name,
            stats=None,
            evaluation=evaluation,
            should_pickup=False,
            evaluation_method="default",
        )

    def _evaluate_with_gemini(
        self,
        item: DetectedItem,
        screenshot: np.ndarray,
    ) -> EvaluatedItem:
        """Evaluate item using Gemini Vision."""
        self.gemini_evaluations += 1
        gemini = _get_gemini()

        try:
            # Get item region for focused analysis
            item_region = (item.x, item.y, item.width, item.height)

            analysis = gemini.analyze_item(
                screenshot=screenshot,
                item_region=item_region,
                character_class=self.character_class,
                build=self.build,
            )

            # Convert Gemini analysis to ItemEvaluation
            value_mapping = {
                "trash": ItemValue.TRASH,
                "keep": ItemValue.SELF_USE,
                "valuable": ItemValue.TRADE_MID,
                "gg": ItemValue.GG,
            }

            evaluation = ItemEvaluation(
                item_name=analysis.name,
                keep=analysis.pickup,
                reason=analysis.reason,
                value=value_mapping.get(analysis.value_assessment, ItemValue.TRASH),
                priority=analysis.priority,
            )

            self.logger.info(
                f"Gemini: {analysis.name} -> {analysis.value_assessment} "
                f"({analysis.reason})"
            )

            return EvaluatedItem(
                detected=item,
                name=analysis.name,
                stats=", ".join(analysis.stats) if analysis.stats else None,
                evaluation=evaluation,
                should_pickup=analysis.pickup,
                evaluation_method="gemini",
            )

        except Exception as e:
            self.logger.error(f"Gemini evaluation failed: {e}")
            # Fall through to LLM or default
            if self.use_llm:
                return self._evaluate_with_llm(item)
            return self._default_evaluation(item)

    def _evaluate_with_llm(self, item: DetectedItem) -> EvaluatedItem:
        """Evaluate item using text LLM (fallback)."""
        self.llm_evaluations += 1

        try:
            evaluation = self.brain.evaluate_item(
                item_name=item.name or "Unknown Item",
                item_type="unknown",
                rarity=item.rarity.value,
                stats="",
            )

            return EvaluatedItem(
                detected=item,
                name=item.name,
                stats=None,
                evaluation=evaluation,
                should_pickup=evaluation.keep,
                evaluation_method="llm",
            )
        except Exception as e:
            self.logger.error(f"LLM evaluation failed: {e}")
            return self._default_evaluation(item)

    def _default_evaluation(self, item: DetectedItem) -> EvaluatedItem:
        """Default evaluation when all else fails."""
        evaluation = ItemEvaluation(
            item_name=item.name or item.rarity.value,
            keep=False,
            reason="Evaluation failed, skipping",
            value=ItemValue.TRASH,
            priority=1,
        )
        return EvaluatedItem(
            detected=item,
            name=item.name,
            stats=None,
            evaluation=evaluation,
            should_pickup=False,
            evaluation_method="default",
        )

    def evaluate_items(
        self,
        items: List[DetectedItem],
        screenshot: Optional[np.ndarray] = None,
        max_evaluate: int = 10,
    ) -> List[EvaluatedItem]:
        """Evaluate multiple items and return sorted by priority.

        Args:
            items: List of detected items
            screenshot: Full screenshot for Gemini analysis
            max_evaluate: Maximum items to fully evaluate

        Returns:
            List of evaluated items, sorted by priority (highest first)
        """
        evaluated = []

        for item in items[:max_evaluate]:
            eval_item = self.evaluate_item(item, screenshot)
            evaluated.append(eval_item)

        # Sort by priority (highest first)
        evaluated.sort(key=lambda x: x.evaluation.priority, reverse=True)

        return evaluated

    def evaluate_items_batch(
        self,
        screenshot: np.ndarray,
    ) -> List[EvaluatedItem]:
        """Evaluate all visible items in one Gemini call (most efficient).

        Args:
            screenshot: Full game screenshot with items visible

        Returns:
            List of evaluated items, sorted by priority
        """
        if not self.use_gemini:
            return []

        gemini = _get_gemini()
        if not gemini:
            return []

        self.gemini_evaluations += 1

        try:
            analyses = gemini.analyze_items_batch(
                screenshot=screenshot,
                character_class=self.character_class,
                build=self.build,
            )

            results = []
            for analysis in analyses:
                value_mapping = {
                    "trash": ItemValue.TRASH,
                    "keep": ItemValue.SELF_USE,
                    "valuable": ItemValue.TRADE_MID,
                    "gg": ItemValue.GG,
                }

                evaluation = ItemEvaluation(
                    item_name=analysis.name,
                    keep=analysis.pickup,
                    reason=analysis.reason,
                    value=value_mapping.get(analysis.value_assessment, ItemValue.TRASH),
                    priority=analysis.priority,
                )

                # Create a minimal DetectedItem (position unknown in batch mode)
                detected = DetectedItem(
                    x=0, y=0, width=0, height=0,
                    rarity=self._rarity_from_string(analysis.rarity),
                    name=analysis.name,
                )

                results.append(EvaluatedItem(
                    detected=detected,
                    name=analysis.name,
                    stats=", ".join(analysis.stats) if analysis.stats else None,
                    evaluation=evaluation,
                    should_pickup=analysis.pickup,
                    evaluation_method="gemini_batch",
                ))

            self.items_evaluated += len(results)
            return results

        except Exception as e:
            self.logger.error(f"Batch evaluation failed: {e}")
            return []

    def get_pickup_list(
        self,
        items: List[DetectedItem],
        screenshot: Optional[np.ndarray] = None,
        max_items: int = 5,
    ) -> List[EvaluatedItem]:
        """Get list of items that should be picked up, in priority order.

        Args:
            items: Detected items
            screenshot: Screenshot for Gemini analysis
            max_items: Maximum items to return

        Returns:
            Items to pick up, sorted by priority
        """
        evaluated = self.evaluate_items(items, screenshot)

        # Filter to items worth picking up
        pickup = [e for e in evaluated if e.should_pickup]

        return pickup[:max_items]

    def get_pickup_list_batch(
        self,
        screenshot: np.ndarray,
        max_items: int = 5,
    ) -> List[EvaluatedItem]:
        """Get pickup list using batch Gemini analysis (most efficient).

        Args:
            screenshot: Full game screenshot
            max_items: Maximum items to return

        Returns:
            Items to pick up, sorted by priority
        """
        evaluated = self.evaluate_items_batch(screenshot)
        pickup = [e for e in evaluated if e.should_pickup]
        return pickup[:max_items]

    def _rarity_to_value(self, rarity: ItemRarity) -> ItemValue:
        """Convert rarity to value estimate."""
        mapping = {
            ItemRarity.UNIQUE: ItemValue.TRADE_MID,
            ItemRarity.SET: ItemValue.TRADE_MID,
            ItemRarity.RUNE: ItemValue.TRADE_HIGH,
            ItemRarity.RARE: ItemValue.CHARSI,
            ItemRarity.MAGIC: ItemValue.TRASH,
            ItemRarity.GOLD: ItemValue.TRASH,
            ItemRarity.NORMAL: ItemValue.TRASH,
        }
        return mapping.get(rarity, ItemValue.TRASH)

    def _rarity_from_string(self, rarity_str: str) -> ItemRarity:
        """Convert string to ItemRarity."""
        mapping = {
            "unique": ItemRarity.UNIQUE,
            "set": ItemRarity.SET,
            "rare": ItemRarity.RARE,
            "magic": ItemRarity.MAGIC,
            "normal": ItemRarity.NORMAL,
            "rune": ItemRarity.RUNE,
            "gold": ItemRarity.GOLD,
        }
        return mapping.get(rarity_str.lower(), ItemRarity.NORMAL)

    def get_stats(self) -> dict:
        """Get evaluator statistics."""
        total = max(1, self.items_evaluated)
        gemini_stats = {}

        gemini = _get_gemini()
        if gemini:
            gemini_stats = gemini.get_stats()

        return {
            "items_evaluated": self.items_evaluated,
            "quick_evaluations": self.quick_evaluations,
            "gemini_evaluations": self.gemini_evaluations,
            "llm_evaluations": self.llm_evaluations,
            "quick_rate": f"{self.quick_evaluations / total:.1%}",
            "gemini_rate": f"{self.gemini_evaluations / total:.1%}",
            **gemini_stats,
        }
