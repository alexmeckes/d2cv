"""
Run manager - orchestrates farming runs.
"""

import time
import random
from typing import List, Optional, Dict, Type
from dataclasses import dataclass

from .base_run import BaseRun, RunStatus
from .mephisto import MephistoRun
from .mephisto_gemini import MephistoGeminiRun
from .andariel import AndarielRun
from .ancient_tunnels import AncientTunnelsRun
from .cow_level import CowLevelRun
from .freeform_clear import FreeformClearRun
from src.config import get_config


@dataclass
class RunResult:
    """Result of a completed run."""
    run_name: str
    status: RunStatus
    duration: float
    items_found: int
    deaths: int


class RunManager:
    """Manages the sequence of farming runs."""

    # Registry of available runs
    AVAILABLE_RUNS: Dict[str, Type[BaseRun]] = {
        "mephisto": MephistoRun,
        "mephisto_gemini": MephistoGeminiRun,  # Gemini-powered version
        "andariel": AndarielRun,
        "ancient_tunnels": AncientTunnelsRun,
        "cow_level": CowLevelRun,
        "freeform_clear": FreeformClearRun,  # Route-independent - kill anything visible
    }

    def __init__(self, **run_dependencies):
        """Initialize with dependencies needed for runs.

        Args:
            **run_dependencies: All dependencies needed by BaseRun
                (screen, input_ctrl, combat, movement, town, etc.)
        """
        self.config = get_config()
        self.run_dependencies = run_dependencies

        # Run queue
        self.enabled_runs: List[str] = self.config.runs_enabled
        self.current_run_index = 0
        self.randomize = self.config.get("runs.randomize", False)

        # Results tracking
        self.results: List[RunResult] = []
        self.consecutive_failures = 0
        self.max_consecutive_failures = 5

        # State
        self.is_running = False
        self.should_stop = False
        self.current_run: Optional[BaseRun] = None

    def get_next_run(self) -> Optional[BaseRun]:
        """Get the next run to execute.

        Returns:
            Instantiated run object or None if no runs available
        """
        if not self.enabled_runs:
            return None

        # Get next run name
        if self.randomize:
            run_name = random.choice(self.enabled_runs)
        else:
            run_name = self.enabled_runs[self.current_run_index]
            self.current_run_index = (self.current_run_index + 1) % len(self.enabled_runs)

        # Get run class
        run_class = self.AVAILABLE_RUNS.get(run_name)
        if run_class is None:
            print(f"Warning: Unknown run type '{run_name}'")
            return None

        # Instantiate run with dependencies
        return run_class(**self.run_dependencies)

    def execute_single_run(self) -> Optional[RunResult]:
        """Execute a single farming run.

        Returns:
            RunResult or None if no run available
        """
        run = self.get_next_run()
        if run is None:
            return None

        self.current_run = run
        start_time = time.time()

        try:
            status = run.execute()
        except Exception as e:
            print(f"Run exception: {e}")
            status = RunStatus.FAILED

        duration = time.time() - start_time

        result = RunResult(
            run_name=run.config.name,
            status=status,
            duration=duration,
            items_found=len(run.items_found),
            deaths=run.deaths,
        )

        self.results.append(result)
        self.current_run = None

        # Track failures
        if status in (RunStatus.FAILED, RunStatus.ABORTED):
            self.consecutive_failures += 1
        else:
            self.consecutive_failures = 0

        return result

    def run_loop(self, max_runs: int = 0, callback=None) -> None:
        """Execute runs in a loop.

        Args:
            max_runs: Maximum runs to execute (0 = infinite)
            callback: Optional callback(result) called after each run
        """
        self.is_running = True
        self.should_stop = False
        runs_completed = 0

        while not self.should_stop:
            # Check max runs
            if max_runs > 0 and runs_completed >= max_runs:
                break

            # Check consecutive failures
            if self.consecutive_failures >= self.max_consecutive_failures:
                print(f"Too many consecutive failures ({self.consecutive_failures}), stopping")
                break

            # Execute run
            result = self.execute_single_run()

            if result is None:
                print("No runs available")
                break

            runs_completed += 1

            # Callback
            if callback:
                callback(result)

            # Brief pause between runs
            if not self.should_stop:
                time.sleep(1)

        self.is_running = False

    def stop(self) -> None:
        """Signal the run loop to stop."""
        self.should_stop = True
        if self.current_run:
            self.current_run.abort("User stopped")

    def get_statistics(self) -> Dict:
        """Get statistics for all runs."""
        if not self.results:
            return {
                "total_runs": 0,
                "successful_runs": 0,
                "success_rate": "0%",
                "total_items": 0,
                "total_deaths": 0,
                "average_duration": "0s",
            }

        successful = sum(1 for r in self.results if r.status == RunStatus.COMPLETED)
        total_items = sum(r.items_found for r in self.results)
        total_deaths = sum(r.deaths for r in self.results)
        avg_duration = sum(r.duration for r in self.results) / len(self.results)

        return {
            "total_runs": len(self.results),
            "successful_runs": successful,
            "success_rate": f"{successful / len(self.results):.1%}",
            "total_items": total_items,
            "total_deaths": total_deaths,
            "average_duration": f"{avg_duration:.1f}s",
            "runs_by_type": self._runs_by_type(),
        }

    def _runs_by_type(self) -> Dict[str, int]:
        """Count runs by type."""
        counts = {}
        for result in self.results:
            counts[result.run_name] = counts.get(result.run_name, 0) + 1
        return counts
