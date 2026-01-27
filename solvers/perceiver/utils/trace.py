"""Trace logging for debugging and analysis."""

import time
from typing import Any


class TraceLogger:
    """Captures all specialist calls and responses for debugging."""

    def __init__(self):
        self.entries: list[dict[str, Any]] = []
        self.start_time: float | None = None
        self._enabled: bool = True

    def reset(self):
        """Reset the logger for a new task."""
        self.entries = []
        self.start_time = time.time()

    def enable(self):
        """Enable logging."""
        self._enabled = True

    def disable(self):
        """Disable logging."""
        self._enabled = False

    def log(
        self,
        role: str,
        model: str,
        prompt: str,
        response: str,
        elapsed: float = 0.0,
        tokens: dict[str, int] | None = None,
    ):
        """Log an LLM call."""
        if not self._enabled:
            return

        timestamp = time.time() - self.start_time if self.start_time else 0
        self.entries.append({
            "timestamp": timestamp,
            "role": role,
            "model": model,
            "prompt": prompt[:1000],  # Truncate for storage
            "response": response[:1000],
            "elapsed": elapsed,
            "tokens": tokens or {},
        })

    def log_event(self, event_type: str, message: str, details: dict[str, Any] | None = None):
        """Log a non-LLM event."""
        if not self._enabled:
            return

        timestamp = time.time() - self.start_time if self.start_time else 0
        self.entries.append({
            "timestamp": timestamp,
            "event_type": event_type,
            "message": message,
            "details": details or {},
        })

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of all logged activity."""
        llm_calls = [e for e in self.entries if "role" in e]
        events = [e for e in self.entries if "event_type" in e]

        role_counts: dict[str, int] = {}
        role_times: dict[str, float] = {}

        for entry in llm_calls:
            role = entry["role"]
            role_counts[role] = role_counts.get(role, 0) + 1
            role_times[role] = role_times.get(role, 0) + entry.get("elapsed", 0)

        return {
            "total_entries": len(self.entries),
            "llm_calls": len(llm_calls),
            "events": len(events),
            "calls_by_role": role_counts,
            "time_by_role": role_times,
            "total_time": self.entries[-1]["timestamp"] if self.entries else 0,
        }

    def get_entries(self) -> list[dict[str, Any]]:
        """Get all log entries."""
        return self.entries.copy()


# Global trace logger instance
TRACE_LOGGER = TraceLogger()

