"""
Game data tracking - session statistics, inventory state, run history.
"""

import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


@dataclass
class ItemDrop:
    """Record of an item drop."""
    name: str
    rarity: str
    area: str
    timestamp: float
    picked_up: bool = True
    llm_evaluation: Optional[str] = None


@dataclass
class RunRecord:
    """Record of a completed run."""
    run_type: str  # "mephisto", "andariel", etc.
    start_time: float
    end_time: float
    success: bool
    deaths: int
    items_found: List[ItemDrop] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def duration(self) -> float:
        """Run duration in seconds."""
        return self.end_time - self.start_time

    @property
    def duration_str(self) -> str:
        """Human-readable duration."""
        d = int(self.duration)
        return f"{d // 60}m {d % 60}s"


class SessionStats:
    """Tracks statistics for the current bot session."""

    def __init__(self):
        self.session_start = time.time()
        self.runs: List[RunRecord] = []
        self.current_run: Optional[RunRecord] = None

        # Counters
        self.total_kills = 0
        self.total_gold = 0

        # Item tracking by rarity
        self.items_by_rarity: Dict[str, int] = {
            "unique": 0,
            "set": 0,
            "rare": 0,
            "rune": 0,
            "magic": 0,
        }

        # Notable drops (high value items)
        self.notable_drops: List[ItemDrop] = []

    def start_run(self, run_type: str) -> None:
        """Start tracking a new run."""
        self.current_run = RunRecord(
            run_type=run_type,
            start_time=time.time(),
            end_time=0,
            success=False,
            deaths=0,
        )

    def end_run(self, success: bool, deaths: int = 0, error: Optional[str] = None) -> None:
        """End the current run."""
        if self.current_run:
            self.current_run.end_time = time.time()
            self.current_run.success = success
            self.current_run.deaths = deaths
            self.current_run.error = error
            self.runs.append(self.current_run)
            self.current_run = None

    def record_item(
        self,
        name: str,
        rarity: str,
        area: str,
        picked_up: bool = True,
        llm_evaluation: Optional[str] = None
    ) -> None:
        """Record an item drop."""
        drop = ItemDrop(
            name=name,
            rarity=rarity,
            area=area,
            timestamp=time.time(),
            picked_up=picked_up,
            llm_evaluation=llm_evaluation,
        )

        # Add to current run if active
        if self.current_run:
            self.current_run.items_found.append(drop)

        # Update counters
        rarity_lower = rarity.lower()
        if rarity_lower in self.items_by_rarity:
            self.items_by_rarity[rarity_lower] += 1

        # Track notable drops
        if rarity_lower in ("unique", "set", "rune"):
            self.notable_drops.append(drop)

    def record_gold(self, amount: int) -> None:
        """Record gold pickup."""
        self.total_gold += amount

    @property
    def session_duration(self) -> float:
        """Session duration in seconds."""
        return time.time() - self.session_start

    @property
    def session_duration_str(self) -> str:
        """Human-readable session duration."""
        d = int(self.session_duration)
        hours = d // 3600
        minutes = (d % 3600) // 60
        seconds = d % 60
        return f"{hours}h {minutes}m {seconds}s"

    @property
    def runs_completed(self) -> int:
        return len(self.runs)

    @property
    def successful_runs(self) -> int:
        return sum(1 for r in self.runs if r.success)

    @property
    def total_deaths(self) -> int:
        return sum(r.deaths for r in self.runs)

    @property
    def average_run_time(self) -> float:
        """Average run time in seconds."""
        if not self.runs:
            return 0
        return sum(r.duration for r in self.runs) / len(self.runs)

    @property
    def runs_per_hour(self) -> float:
        """Runs per hour rate."""
        if self.session_duration < 60:
            return 0
        return self.runs_completed / (self.session_duration / 3600)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of session statistics."""
        return {
            "session_duration": self.session_duration_str,
            "runs_completed": self.runs_completed,
            "successful_runs": self.successful_runs,
            "success_rate": f"{self.successful_runs / max(1, self.runs_completed):.1%}",
            "total_deaths": self.total_deaths,
            "average_run_time": f"{self.average_run_time:.1f}s",
            "runs_per_hour": f"{self.runs_per_hour:.1f}",
            "items_found": self.items_by_rarity,
            "total_gold": self.total_gold,
            "notable_drops": len(self.notable_drops),
        }

    def get_recent_drops(self, count: int = 10) -> List[ItemDrop]:
        """Get most recent item drops."""
        all_drops = []
        for run in reversed(self.runs):
            all_drops.extend(reversed(run.items_found))
            if len(all_drops) >= count:
                break
        return all_drops[:count]


@dataclass
class InventorySlot:
    """Represents a slot in the inventory."""
    x: int  # Grid x position (0-9)
    y: int  # Grid y position (0-3)
    occupied: bool = False
    item_name: Optional[str] = None
    item_rarity: Optional[str] = None


class InventoryState:
    """Tracks inventory state."""

    # Standard inventory is 10x4 slots
    WIDTH = 10
    HEIGHT = 4

    def __init__(self):
        self.slots: List[List[InventorySlot]] = [
            [InventorySlot(x=x, y=y) for x in range(self.WIDTH)]
            for y in range(self.HEIGHT)
        ]
        self.last_update = 0

    def update_slot(
        self,
        x: int,
        y: int,
        occupied: bool,
        item_name: Optional[str] = None,
        item_rarity: Optional[str] = None
    ) -> None:
        """Update a specific inventory slot."""
        if 0 <= x < self.WIDTH and 0 <= y < self.HEIGHT:
            slot = self.slots[y][x]
            slot.occupied = occupied
            slot.item_name = item_name
            slot.item_rarity = item_rarity
        self.last_update = time.time()

    def count_free_slots(self) -> int:
        """Count number of free inventory slots."""
        return sum(
            1 for row in self.slots for slot in row if not slot.occupied
        )

    def is_full(self) -> bool:
        """Check if inventory is full."""
        return self.count_free_slots() == 0

    def find_free_slot(self) -> Optional[InventorySlot]:
        """Find a free slot."""
        for row in self.slots:
            for slot in row:
                if not slot.occupied:
                    return slot
        return None

    def clear(self) -> None:
        """Clear all slots (after stashing)."""
        for row in self.slots:
            for slot in row:
                slot.occupied = False
                slot.item_name = None
                slot.item_rarity = None
