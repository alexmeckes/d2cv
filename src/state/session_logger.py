"""
Session logger - persists session data to disk and provides structured logging.
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from pathlib import Path

from .game_data import SessionStats, RunRecord, ItemDrop


@dataclass
class SessionSummary:
    """Summary of a completed session."""
    session_id: str
    start_time: str
    end_time: str
    duration_seconds: float
    total_runs: int
    successful_runs: int
    total_deaths: int
    items_by_rarity: Dict[str, int]
    notable_drops: List[str]
    runs_per_hour: float
    average_run_time: float


class SessionLogger:
    """Logs session data to files and provides structured logging."""

    def __init__(self, log_dir: str = "./logs"):
        """Initialize the session logger.

        Args:
            log_dir: Directory to store log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.log_dir / "sessions").mkdir(exist_ok=True)
        (self.log_dir / "runs").mkdir(exist_ok=True)
        (self.log_dir / "items").mkdir(exist_ok=True)

        # Session ID (timestamp-based)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_start = datetime.now()

        # Set up logging
        self._setup_logging()

        # Item log file
        self.item_log_path = self.log_dir / "items" / f"items_{self.session_id}.json"
        self.items_logged: List[Dict] = []

    def _setup_logging(self):
        """Configure Python logging."""
        log_file = self.log_dir / f"d2cv_{self.session_id}.log"

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%H:%M:%S'
        )

        # File handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        # Configure root logger
        root_logger = logging.getLogger('d2cv')
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

        self.logger = root_logger

    def log_run_start(self, run_type: str) -> None:
        """Log the start of a run."""
        self.logger.info(f"Starting {run_type} run")

    def log_run_end(self, run: RunRecord) -> None:
        """Log the completion of a run."""
        status = "SUCCESS" if run.success else "FAILED"
        self.logger.info(
            f"Run complete: {run.run_type} - {status} "
            f"({run.duration_str}, {len(run.items_found)} items, {run.deaths} deaths)"
        )

        if run.error:
            self.logger.warning(f"Run error: {run.error}")

        # Save run data to file
        self._save_run(run)

    def log_item_found(self, item: ItemDrop) -> None:
        """Log an item drop."""
        self.logger.info(f"Item found: {item.name} ({item.rarity}) in {item.area}")

        # Add to items list
        item_dict = {
            "name": item.name,
            "rarity": item.rarity,
            "area": item.area,
            "timestamp": datetime.fromtimestamp(item.timestamp).isoformat(),
            "picked_up": item.picked_up,
            "llm_evaluation": item.llm_evaluation,
        }
        self.items_logged.append(item_dict)

        # Save items incrementally
        self._save_items()

    def log_death(self, location: str = "unknown") -> None:
        """Log a death."""
        self.logger.warning(f"Character died in {location}")

    def log_chicken(self, health_percent: float, location: str = "unknown") -> None:
        """Log an emergency exit (chicken)."""
        self.logger.warning(
            f"Emergency exit at {health_percent:.0%} health in {location}"
        )

    def log_error(self, error: str, exc_info: bool = False) -> None:
        """Log an error."""
        self.logger.error(error, exc_info=exc_info)

    def log_llm_call(self, prompt_type: str, tokens_used: int, cached: bool = False) -> None:
        """Log an LLM API call."""
        cache_str = " (cached)" if cached else ""
        self.logger.debug(f"LLM call: {prompt_type}, {tokens_used} tokens{cache_str}")

    def _save_run(self, run: RunRecord) -> None:
        """Save run data to JSON file."""
        run_data = {
            "run_type": run.run_type,
            "start_time": datetime.fromtimestamp(run.start_time).isoformat(),
            "end_time": datetime.fromtimestamp(run.end_time).isoformat(),
            "duration_seconds": run.duration,
            "success": run.success,
            "deaths": run.deaths,
            "items_found": [
                {
                    "name": item.name,
                    "rarity": item.rarity,
                    "area": item.area,
                }
                for item in run.items_found
            ],
            "error": run.error,
        }

        run_file = self.log_dir / "runs" / f"run_{self.session_id}_{int(run.start_time)}.json"
        with open(run_file, 'w', encoding='utf-8') as f:
            json.dump(run_data, f, indent=2)

    def _save_items(self) -> None:
        """Save items to JSON file."""
        with open(self.item_log_path, 'w', encoding='utf-8') as f:
            json.dump(self.items_logged, f, indent=2)

    def save_session_summary(self, stats: SessionStats) -> SessionSummary:
        """Save final session summary.

        Args:
            stats: Session statistics

        Returns:
            SessionSummary object
        """
        end_time = datetime.now()

        summary = SessionSummary(
            session_id=self.session_id,
            start_time=self.session_start.isoformat(),
            end_time=end_time.isoformat(),
            duration_seconds=stats.session_duration,
            total_runs=stats.runs_completed,
            successful_runs=stats.successful_runs,
            total_deaths=stats.total_deaths,
            items_by_rarity=stats.items_by_rarity.copy(),
            notable_drops=[d.name for d in stats.notable_drops],
            runs_per_hour=stats.runs_per_hour,
            average_run_time=stats.average_run_time,
        )

        # Save to file
        summary_file = self.log_dir / "sessions" / f"session_{self.session_id}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(summary), f, indent=2)

        self.logger.info(f"Session summary saved to {summary_file}")

        return summary

    @staticmethod
    def load_session_history(log_dir: str = "./logs", limit: int = 10) -> List[SessionSummary]:
        """Load recent session summaries.

        Args:
            log_dir: Directory containing log files
            limit: Maximum sessions to load

        Returns:
            List of SessionSummary objects (most recent first)
        """
        sessions_dir = Path(log_dir) / "sessions"
        if not sessions_dir.exists():
            return []

        summaries = []
        session_files = sorted(sessions_dir.glob("session_*.json"), reverse=True)

        for session_file in session_files[:limit]:
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    summaries.append(SessionSummary(**data))
            except (json.JSONDecodeError, TypeError):
                continue

        return summaries

    @staticmethod
    def get_aggregate_stats(log_dir: str = "./logs") -> Dict[str, Any]:
        """Get aggregate statistics across all sessions.

        Args:
            log_dir: Directory containing log files

        Returns:
            Dictionary with aggregate statistics
        """
        sessions = SessionLogger.load_session_history(log_dir, limit=100)

        if not sessions:
            return {
                "total_sessions": 0,
                "total_runs": 0,
                "total_items": {},
                "total_time_hours": 0,
            }

        total_runs = sum(s.total_runs for s in sessions)
        total_deaths = sum(s.total_deaths for s in sessions)
        total_time = sum(s.duration_seconds for s in sessions)

        # Aggregate items by rarity
        total_items: Dict[str, int] = {}
        for session in sessions:
            for rarity, count in session.items_by_rarity.items():
                total_items[rarity] = total_items.get(rarity, 0) + count

        return {
            "total_sessions": len(sessions),
            "total_runs": total_runs,
            "total_deaths": total_deaths,
            "total_items": total_items,
            "total_time_hours": total_time / 3600,
            "average_runs_per_session": total_runs / len(sessions) if sessions else 0,
            "all_notable_drops": [
                drop
                for s in sessions
                for drop in s.notable_drops
            ],
        }


# Global logger instance
_session_logger: Optional[SessionLogger] = None


def get_session_logger() -> SessionLogger:
    """Get or create the global session logger."""
    global _session_logger
    if _session_logger is None:
        _session_logger = SessionLogger()
    return _session_logger


def get_logger(name: str = "d2cv") -> logging.Logger:
    """Get a logger instance."""
    # Ensure session logger is initialized
    get_session_logger()
    return logging.getLogger(name)
