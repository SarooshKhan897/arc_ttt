"""Configuration for Triple Solver + Judge System.

Combines Arc-Solver (voting), Tool Solver (4-phase), and Claude V3 (iterative)
with a Judge LLM that picks the top 2 outputs.
"""

import os
import threading
from dataclasses import dataclass, field
from typing import Any
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# API Configuration
# =============================================================================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

REQUEST_TIMEOUT = 1800.0  # 30 min
MAX_WORKERS = 50

# Provider routing - restrict to anthropic only
PROVIDER_CONFIG = {"provider": {"only": ["anthropic"]}}

# =============================================================================
# Shared Hypothesis Model (Phase 0)
# =============================================================================

HYPOTHESIS_MODEL = "anthropic/claude-opus-4.6"
HYPOTHESIS_EXTRA_BODY = {"reasoning": {"enabled": True}}

# =============================================================================
# Judge Model
# =============================================================================

JUDGE_MODEL = "anthropic/claude-opus-4.6"
JUDGE_EXTRA_BODY = {"reasoning": {"enabled": True}}

# =============================================================================
# Arc-Solver Model Configs
# =============================================================================

ARC_SOLVER_MODELS = [
    {
        "id": "claude-opus-4.6",
        "model": "anthropic/claude-opus-4.6",
        "extra_body": {"reasoning": {"enabled": True}},
        "tries": 5,
    },
]

# =============================================================================
# Tool Solver Model
# =============================================================================

TOOL_SOLVER_MODEL = {
    "id": "claude-opus-4.6",
    "model": "anthropic/claude-opus-4.6",
    "extra_body": {"reasoning": {"enabled": True}},
}

# =============================================================================
# Claude V3 Solver Model
# =============================================================================

CLAUDE_V3_MODEL = "anthropic/claude-opus-4.6"
CLAUDE_V3_MAX_ITERATIONS = 25

# =============================================================================
# Perceiver/Differencer models
# =============================================================================

PERCEIVER_MODEL = "anthropic/claude-opus-4.6"
PERCEIVER_EXTRA_BODY = {"reasoning": {"enabled": True}}

# =============================================================================
# Cost Tracking
# =============================================================================

@dataclass
class UsageStats:
    """Accumulated usage statistics across all LLM calls."""
    total_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    total_reasoning_tokens: int = 0
    total_cached_tokens: int = 0
    is_byok: bool = False
    upstream_cost: float = 0.0
    upstream_prompt_cost: float = 0.0
    upstream_completion_cost: float = 0.0
    
    def add(self, usage: dict[str, Any] | None):
        """Add usage from a single response."""
        if usage is None:
            return
        self.total_calls += 1
        self.total_prompt_tokens += usage.get("prompt_tokens", 0)
        self.total_completion_tokens += usage.get("completion_tokens", 0)
        self.total_tokens += usage.get("total_tokens", 0)
        self.total_cost += usage.get("cost", 0.0) or 0.0
        
        completion_details = usage.get("completion_tokens_details", {})
        if completion_details:
            self.total_reasoning_tokens += completion_details.get("reasoning_tokens", 0)
        
        prompt_details = usage.get("prompt_tokens_details", {})
        if prompt_details:
            self.total_cached_tokens += prompt_details.get("cached_tokens", 0)
        
        if usage.get("is_byok"):
            self.is_byok = True
        cost_details = usage.get("cost_details", {})
        if cost_details:
            self.upstream_cost += cost_details.get("upstream_inference_cost", 0) or 0
            self.upstream_prompt_cost += cost_details.get("upstream_inference_prompt_cost", 0) or 0
            self.upstream_completion_cost += cost_details.get("upstream_inference_completions_cost", 0) or 0
    
    @property
    def effective_cost(self) -> float:
        return self.upstream_cost if self.is_byok else self.total_cost
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "total_reasoning_tokens": self.total_reasoning_tokens,
            "total_cached_tokens": self.total_cached_tokens,
            "is_byok": self.is_byok,
            "upstream_cost": self.upstream_cost,
            "upstream_prompt_cost": self.upstream_prompt_cost,
            "upstream_completion_cost": self.upstream_completion_cost,
            "effective_cost": self.effective_cost,
        }
    
    def __str__(self) -> str:
        cost_str = (
            f"  BYOK Cost: ${self.upstream_cost:.4f}\n"
            f"     - Prompt: ${self.upstream_prompt_cost:.4f}\n"
            f"     - Completion: ${self.upstream_completion_cost:.4f}"
        ) if self.is_byok else (
            f"  Total cost: ${self.total_cost:.4f}"
        )
        return (
            f"LLM Usage Summary:\n"
            f"  Total API calls: {self.total_calls:,}\n"
            f"  Prompt tokens: {self.total_prompt_tokens:,}\n"
            f"  Completion tokens: {self.total_completion_tokens:,}\n"
            f"  Reasoning tokens: {self.total_reasoning_tokens:,}\n"
            f"  Cached tokens: {self.total_cached_tokens:,}\n"
            f"  Total tokens: {self.total_tokens:,}\n"
            f"  BYOK: {self.is_byok}\n"
            f"{cost_str}"
        )


# Global usage tracking
_usage_stats = UsageStats()
_usage_lock = threading.Lock()


def get_usage_stats() -> UsageStats:
    return _usage_stats


def reset_usage_stats():
    global _usage_stats
    _usage_stats = UsageStats()


def record_usage(usage: dict[str, Any] | None):
    """Thread-safe recording of usage statistics."""
    with _usage_lock:
        _usage_stats.add(usage)
