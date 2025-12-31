"""
Item evaluator - combines OCR, quick rules, and LLM for item decisions.
"""

import time
from typing import Optional, List, Tuple
from dataclasses import dataclass
import numpy as np

from .deliberative import DeliberativeBrain, ItemEvaluation, ItemValue, get_deliberative_brain
from .prompts import should_quick_pickup, should_quick_skip
from src.vision.ocr import GameOCR
from src.vision.entities import DetectedItem, ItemRarity


@dataclass
class EvaluatedItem:
    """An item that has been fully evaluated."""
    detected: DetectedItem
    name: Optional[str]
    stats: Optional[str]
    evaluation: ItemEvaluation
    should_pickup: bool


class ItemEvaluator:
    """Evaluates items using OCR and LLM."""

    def __init__(
        self,
        ocr: Optional[GameOCR] = None,
        brain: Optional[DeliberativeBrain] = None,
        use_llm: bool = True,
    ):
        """Initialize the item evaluator.

        Args:
            ocr: OCR engine for reading item text
            brain: Deliberative brain for LLM evaluation
            use_llm: Whether to use LLM for complex evaluations
        """
        self._ocr = ocr
        self._brain = brain
        self.use_llm = use_llm

        # Quick evaluation rules by rarity
        self.rarity_rules = {
            ItemRarity.UNIQUE: {"always_keep": True, "min_priority": 8},
            ItemRarity.SET: {"always_keep": True, "min_priority": 7},
            ItemRarity.RUNE: {"always_keep": True, "min_priority": 9},
            ItemRarity.RARE: {"always_keep": False, "min_priority": 5},  # Evaluate
            ItemRarity.MAGIC: {"always_keep": False, "min_priority": 3},
            ItemRarity.GOLD: {"always_keep": True, "min_priority": 2},
            ItemRarity.NORMAL: {"always_keep": False, "min_priority": 1},
        }

        # Stats
        self.items_evaluated = 0
        self.llm_evaluations = 0

    @property
    def ocr(self) -> GameOCR:
        if self._ocr is None:
            self._ocr = GameOCR()
        return self._ocr

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
            screenshot: Full screenshot (for OCR if needed)

        Returns:
            EvaluatedItem with full evaluation
        """
        self.items_evaluated += 1

        # Get item name from OCR if possible
        name = item.name
        stats = None

        if screenshot is not None and name is None:
            # Extract item label region and OCR it
            x, y, w, h = item.x, item.y, item.width, item.height
            # Add some padding
            pad = 5
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(screenshot.shape[1], x + w + pad)
            y2 = min(screenshot.shape[0], y + h + pad)

            label_region = screenshot[y1:y2, x1:x2]
            name = self.ocr.read_item_label(label_region)

        # Quick rules based on rarity
        rules = self.rarity_rules.get(item.rarity, {"always_keep": False, "min_priority": 1})

        if rules["always_keep"]:
            # High-value rarities - always keep
            evaluation = ItemEvaluation(
                item_name=name or item.rarity.value,
                keep=True,
                reason=f"Always keep {item.rarity.value} items",
                value=self._rarity_to_value(item.rarity),
                priority=rules["min_priority"],
            )
            return EvaluatedItem(
                detected=item,
                name=name,
                stats=stats,
                evaluation=evaluation,
                should_pickup=True,
            )

        # Quick text-based rules
        if name:
            if should_quick_pickup(name):
                evaluation = ItemEvaluation(
                    item_name=name,
                    keep=True,
                    reason="High-value item detected",
                    value=ItemValue.TRADE_HIGH,
                    priority=9,
                )
                return EvaluatedItem(
                    detected=item,
                    name=name,
                    stats=stats,
                    evaluation=evaluation,
                    should_pickup=True,
                )

            if should_quick_skip(name):
                evaluation = ItemEvaluation(
                    item_name=name,
                    keep=False,
                    reason="Low-value item",
                    value=ItemValue.TRASH,
                    priority=1,
                )
                return EvaluatedItem(
                    detected=item,
                    name=name,
                    stats=stats,
                    evaluation=evaluation,
                    should_pickup=False,
                )

        # Use LLM for uncertain items (rares, some magics)
        if self.use_llm and item.rarity in (ItemRarity.RARE,):
            self.llm_evaluations += 1
            evaluation = self.brain.evaluate_item(
                item_name=name or "Unknown Item",
                item_type="unknown",
                rarity=item.rarity.value,
                stats=stats or "",
            )
            return EvaluatedItem(
                detected=item,
                name=name,
                stats=stats,
                evaluation=evaluation,
                should_pickup=evaluation.keep,
            )

        # Default: skip
        evaluation = ItemEvaluation(
            item_name=name or item.rarity.value,
            keep=False,
            reason="Default: skip uncertain item",
            value=ItemValue.TRASH,
            priority=1,
        )
        return EvaluatedItem(
            detected=item,
            name=name,
            stats=stats,
            evaluation=evaluation,
            should_pickup=False,
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
            screenshot: Full screenshot for OCR
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

    def get_pickup_list(
        self,
        items: List[DetectedItem],
        screenshot: Optional[np.ndarray] = None,
        max_items: int = 5,
    ) -> List[EvaluatedItem]:
        """Get list of items that should be picked up, in priority order.

        Args:
            items: Detected items
            screenshot: Screenshot for OCR
            max_items: Maximum items to return

        Returns:
            Items to pick up, sorted by priority
        """
        evaluated = self.evaluate_items(items, screenshot)

        # Filter to items worth picking up
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

    def get_stats(self) -> dict:
        """Get evaluator statistics."""
        return {
            "items_evaluated": self.items_evaluated,
            "llm_evaluations": self.llm_evaluations,
            "llm_rate": f"{self.llm_evaluations / max(1, self.items_evaluated):.1%}",
        }
