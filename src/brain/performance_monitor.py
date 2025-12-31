"""
Performance Monitor - tracks Gemini latency and provides real-time stats.

Logs:
- Per-call latency
- Rolling averages
- Cache hit rates
- Cost estimates
- Warnings when latency spikes
"""

import time
import statistics
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime

from src.state.session_logger import get_logger


@dataclass
class CallRecord:
    """Record of a single Gemini call."""
    timestamp: float
    call_type: str  # "navigation", "combat", "items", "screen"
    latency_ms: float
    cached: bool
    success: bool
    tokens_estimate: int = 0


@dataclass
class LatencyStats:
    """Latency statistics."""
    count: int
    min_ms: float
    max_ms: float
    avg_ms: float
    p50_ms: float  # Median
    p95_ms: float  # 95th percentile
    p99_ms: float  # 99th percentile
    std_dev: float


class PerformanceMonitor:
    """Monitors Gemini API performance in real-time."""

    # Latency thresholds (ms)
    LATENCY_WARN = 500      # Warn if > 500ms
    LATENCY_CRITICAL = 1000  # Critical if > 1s

    def __init__(
        self,
        window_size: int = 100,  # Rolling window for stats
        log_every_n: int = 10,   # Log summary every N calls
    ):
        """Initialize performance monitor.

        Args:
            window_size: Number of recent calls to track for rolling stats
            log_every_n: Log summary statistics every N calls
        """
        self.logger = get_logger("perf_monitor")
        self.window_size = window_size
        self.log_every_n = log_every_n

        # Call history (rolling window)
        self.calls: deque = deque(maxlen=window_size)

        # Lifetime stats
        self.total_calls = 0
        self.total_cached = 0
        self.total_latency_ms = 0
        self.total_tokens = 0
        self.start_time = time.time()

        # Per-type tracking
        self.calls_by_type: Dict[str, List[float]] = {
            "navigation": [],
            "combat": [],
            "items": [],
            "screen": [],
            "other": [],
        }

        # Latency alerts
        self.latency_warnings = 0
        self.latency_criticals = 0

        # Callbacks
        self.on_latency_warning: Optional[Callable[[float, str], None]] = None
        self.on_latency_critical: Optional[Callable[[float, str], None]] = None

    def record_call(
        self,
        call_type: str,
        latency_ms: float,
        cached: bool = False,
        success: bool = True,
        tokens: int = 0,
    ) -> None:
        """Record a Gemini API call.

        Args:
            call_type: Type of call (navigation, combat, items, screen)
            latency_ms: Call latency in milliseconds
            cached: Whether this was a cache hit
            success: Whether the call succeeded
            tokens: Estimated tokens used
        """
        record = CallRecord(
            timestamp=time.time(),
            call_type=call_type,
            latency_ms=latency_ms,
            cached=cached,
            success=success,
            tokens_estimate=tokens,
        )

        self.calls.append(record)
        self.total_calls += 1

        if cached:
            self.total_cached += 1
        else:
            self.total_latency_ms += latency_ms
            self.total_tokens += tokens

        # Track by type
        if call_type in self.calls_by_type:
            self.calls_by_type[call_type].append(latency_ms)
            # Keep per-type lists bounded
            if len(self.calls_by_type[call_type]) > self.window_size:
                self.calls_by_type[call_type] = self.calls_by_type[call_type][-self.window_size:]
        else:
            self.calls_by_type["other"].append(latency_ms)

        # Check latency thresholds
        if not cached:
            if latency_ms > self.LATENCY_CRITICAL:
                self.latency_criticals += 1
                self.logger.warning(
                    f"CRITICAL latency: {latency_ms:.0f}ms for {call_type}"
                )
                if self.on_latency_critical:
                    self.on_latency_critical(latency_ms, call_type)
            elif latency_ms > self.LATENCY_WARN:
                self.latency_warnings += 1
                self.logger.info(
                    f"High latency: {latency_ms:.0f}ms for {call_type}"
                )
                if self.on_latency_warning:
                    self.on_latency_warning(latency_ms, call_type)

        # Periodic logging
        if self.total_calls % self.log_every_n == 0:
            self._log_summary()

    def _log_summary(self) -> None:
        """Log a summary of recent performance."""
        stats = self.get_rolling_stats()
        cache_rate = self.total_cached / max(1, self.total_calls)

        self.logger.info(
            f"Gemini stats: {self.total_calls} calls, "
            f"{cache_rate:.0%} cached, "
            f"avg {stats.avg_ms:.0f}ms, "
            f"p95 {stats.p95_ms:.0f}ms"
        )

    def get_rolling_stats(self) -> LatencyStats:
        """Get statistics for recent non-cached calls."""
        latencies = [c.latency_ms for c in self.calls if not c.cached]

        if not latencies:
            return LatencyStats(
                count=0, min_ms=0, max_ms=0, avg_ms=0,
                p50_ms=0, p95_ms=0, p99_ms=0, std_dev=0
            )

        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)

        return LatencyStats(
            count=n,
            min_ms=min(latencies),
            max_ms=max(latencies),
            avg_ms=statistics.mean(latencies),
            p50_ms=sorted_latencies[n // 2],
            p95_ms=sorted_latencies[int(n * 0.95)] if n >= 20 else sorted_latencies[-1],
            p99_ms=sorted_latencies[int(n * 0.99)] if n >= 100 else sorted_latencies[-1],
            std_dev=statistics.stdev(latencies) if n > 1 else 0,
        )

    def get_stats_by_type(self, call_type: str) -> LatencyStats:
        """Get statistics for a specific call type."""
        latencies = self.calls_by_type.get(call_type, [])

        if not latencies:
            return LatencyStats(
                count=0, min_ms=0, max_ms=0, avg_ms=0,
                p50_ms=0, p95_ms=0, p99_ms=0, std_dev=0
            )

        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)

        return LatencyStats(
            count=n,
            min_ms=min(latencies),
            max_ms=max(latencies),
            avg_ms=statistics.mean(latencies),
            p50_ms=sorted_latencies[n // 2],
            p95_ms=sorted_latencies[int(n * 0.95)] if n >= 20 else sorted_latencies[-1],
            p99_ms=sorted_latencies[int(n * 0.99)] if n >= 100 else sorted_latencies[-1],
            std_dev=statistics.stdev(latencies) if n > 1 else 0,
        )

    def get_cost_estimate(self) -> Dict[str, float]:
        """Estimate API cost based on token usage."""
        # Gemini Flash pricing
        input_price = 0.075 / 1_000_000   # $0.075 per 1M tokens
        output_price = 0.30 / 1_000_000   # $0.30 per 1M tokens

        # Estimate: 60% input, 40% output
        input_tokens = self.total_tokens * 0.6
        output_tokens = self.total_tokens * 0.4

        input_cost = input_tokens * input_price
        output_cost = output_tokens * output_price
        total_cost = input_cost + output_cost

        runtime_hours = (time.time() - self.start_time) / 3600
        cost_per_hour = total_cost / max(0.01, runtime_hours)

        return {
            "total_tokens": self.total_tokens,
            "total_cost": total_cost,
            "cost_per_hour": cost_per_hour,
            "runtime_hours": runtime_hours,
        }

    def get_full_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        rolling = self.get_rolling_stats()
        cost = self.get_cost_estimate()
        runtime = time.time() - self.start_time

        actual_calls = self.total_calls - self.total_cached
        cache_rate = self.total_cached / max(1, self.total_calls)
        avg_latency = self.total_latency_ms / max(1, actual_calls)
        calls_per_minute = self.total_calls / max(1, runtime / 60)

        report = {
            "summary": {
                "total_calls": self.total_calls,
                "actual_api_calls": actual_calls,
                "cache_hits": self.total_cached,
                "cache_rate": f"{cache_rate:.1%}",
                "runtime_seconds": int(runtime),
                "calls_per_minute": f"{calls_per_minute:.1f}",
            },
            "latency": {
                "average_ms": f"{avg_latency:.0f}",
                "rolling_avg_ms": f"{rolling.avg_ms:.0f}",
                "p50_ms": f"{rolling.p50_ms:.0f}",
                "p95_ms": f"{rolling.p95_ms:.0f}",
                "p99_ms": f"{rolling.p99_ms:.0f}",
                "min_ms": f"{rolling.min_ms:.0f}",
                "max_ms": f"{rolling.max_ms:.0f}",
            },
            "alerts": {
                "latency_warnings": self.latency_warnings,
                "latency_criticals": self.latency_criticals,
            },
            "by_type": {},
            "cost": {
                "estimated_tokens": self.total_tokens,
                "estimated_cost_usd": f"${cost['total_cost']:.4f}",
                "cost_per_hour_usd": f"${cost['cost_per_hour']:.4f}",
            },
        }

        # Add per-type stats
        for call_type in self.calls_by_type:
            type_stats = self.get_stats_by_type(call_type)
            if type_stats.count > 0:
                report["by_type"][call_type] = {
                    "count": type_stats.count,
                    "avg_ms": f"{type_stats.avg_ms:.0f}",
                    "p95_ms": f"{type_stats.p95_ms:.0f}",
                }

        return report

    def print_report(self) -> None:
        """Print a formatted performance report."""
        report = self.get_full_report()

        print("\n" + "=" * 60)
        print("GEMINI PERFORMANCE REPORT")
        print("=" * 60)

        print(f"\nSUMMARY")
        print(f"  Total calls:      {report['summary']['total_calls']}")
        print(f"  Actual API calls: {report['summary']['actual_api_calls']}")
        print(f"  Cache rate:       {report['summary']['cache_rate']}")
        print(f"  Runtime:          {report['summary']['runtime_seconds']}s")
        print(f"  Calls/minute:     {report['summary']['calls_per_minute']}")

        print(f"\nLATENCY")
        print(f"  Average:   {report['latency']['average_ms']}ms")
        print(f"  P50:       {report['latency']['p50_ms']}ms")
        print(f"  P95:       {report['latency']['p95_ms']}ms")
        print(f"  P99:       {report['latency']['p99_ms']}ms")
        print(f"  Min/Max:   {report['latency']['min_ms']}ms / {report['latency']['max_ms']}ms")

        print(f"\nBY TYPE")
        for call_type, stats in report['by_type'].items():
            print(f"  {call_type}: {stats['count']} calls, avg {stats['avg_ms']}ms")

        print(f"\nALERTS")
        print(f"  Warnings (>500ms):  {report['alerts']['latency_warnings']}")
        print(f"  Critical (>1000ms): {report['alerts']['latency_criticals']}")

        print(f"\nCOST")
        print(f"  Estimated tokens: {report['cost']['estimated_tokens']}")
        print(f"  Total cost:       {report['cost']['estimated_cost_usd']}")
        print(f"  Cost/hour:        {report['cost']['cost_per_hour_usd']}")

        print("=" * 60 + "\n")

    def reset(self) -> None:
        """Reset all statistics."""
        self.calls.clear()
        self.total_calls = 0
        self.total_cached = 0
        self.total_latency_ms = 0
        self.total_tokens = 0
        self.start_time = time.time()
        self.latency_warnings = 0
        self.latency_criticals = 0
        for key in self.calls_by_type:
            self.calls_by_type[key] = []


# Global instance
_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get or create global performance monitor."""
    global _monitor
    if _monitor is None:
        _monitor = PerformanceMonitor()
    return _monitor


def record_gemini_call(
    call_type: str,
    latency_ms: float,
    cached: bool = False,
    success: bool = True,
    tokens: int = 500,  # Default estimate
) -> None:
    """Convenience function to record a call."""
    get_performance_monitor().record_call(
        call_type=call_type,
        latency_ms=latency_ms,
        cached=cached,
        success=success,
        tokens=tokens,
    )
