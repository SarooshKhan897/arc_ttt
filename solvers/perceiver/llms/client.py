"""Shared API client for LLM calls - Sync version with proper usage tracking."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from solvers.perceiver.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    REQUEST_TIMEOUT,
)

# Max workers for ThreadPoolExecutor
MAX_WORKERS = 200

# =============================================================================
# Usage Tracking (supports both OpenRouter and BYOK costs)
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
            f"  💵 Upstream cost: ${self.upstream_cost:.4f}\n"
            f"     - Prompt: ${self.upstream_prompt_cost:.4f}\n"
            f"     - Completion: ${self.upstream_completion_cost:.4f}"
        ) if self.is_byok else (
            f"  💵 Total cost: ${self.total_cost:.4f}"
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


_usage_stats = UsageStats()
_usage_lock = threading.Lock()


def get_usage_stats() -> UsageStats:
    return _usage_stats


def reset_usage_stats():
    global _usage_stats
    _usage_stats = UsageStats()


def _record_usage(usage: dict[str, Any] | None):
    """Thread-safe recording of usage statistics."""
    with _usage_lock:
        _usage_stats.add(usage)


def get_and_reset_usage() -> dict[str, Any]:
    """Get current usage stats and reset for next task. Thread-safe."""
    global _usage_stats
    with _usage_lock:
        stats = _usage_stats.to_dict()
        _usage_stats = UsageStats()
    return stats


# =============================================================================
# Task-Level Cost Tracking
# =============================================================================

@dataclass
class TaskCostTracker:
    """Track costs for a single task."""
    task_id: str
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cost: float = 0.0
    upstream_cost: float = 0.0
    is_byok: bool = False
    
    def add(self, usage: dict[str, Any] | None):
        if usage is None:
            return
        self.calls += 1
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)
        self.total_tokens += usage.get("total_tokens", 0)
        self.cost += usage.get("cost", 0.0) or 0.0
        
        completion_details = usage.get("completion_tokens_details", {})
        if completion_details:
            self.reasoning_tokens += completion_details.get("reasoning_tokens", 0)
        
        if usage.get("is_byok"):
            self.is_byok = True
        cost_details = usage.get("cost_details", {})
        if cost_details:
            self.upstream_cost += cost_details.get("upstream_inference_cost", 0) or 0
    
    @property
    def effective_cost(self) -> float:
        return self.upstream_cost if self.is_byok else self.cost
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cost": self.cost,
            "upstream_cost": self.upstream_cost,
            "is_byok": self.is_byok,
            "effective_cost": self.effective_cost,
        }


_task_trackers: dict[str, TaskCostTracker] = {}
_task_tracker_lock = threading.Lock()
_current_task_id: str | None = None


class task_cost_context:
    """Context manager to track costs for a specific task."""
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.previous_task_id = None
    
    def __enter__(self):
        global _current_task_id
        self.previous_task_id = _current_task_id
        _current_task_id = self.task_id
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        global _current_task_id
        _current_task_id = self.previous_task_id
        return False


def get_current_task_id() -> str | None:
    return _current_task_id


def get_task_tracker(task_id: str) -> TaskCostTracker:
    if task_id not in _task_trackers:
        _task_trackers[task_id] = TaskCostTracker(task_id=task_id)
    return _task_trackers[task_id]


def get_all_task_trackers() -> dict[str, TaskCostTracker]:
    return _task_trackers.copy()


def reset_task_trackers():
    global _task_trackers
    _task_trackers = {}


def _record_task_usage(task_id: str | None, usage: dict[str, Any] | None):
    """Record usage for a specific task."""
    effective_task_id = task_id or _current_task_id
    if effective_task_id is None or usage is None:
        return
    with _task_tracker_lock:
        tracker = get_task_tracker(effective_task_id)
        tracker.add(usage)


# =============================================================================
# Global ThreadPoolExecutor
# =============================================================================

_executor: ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()


def get_executor() -> ThreadPoolExecutor:
    """Get or create the global ThreadPoolExecutor."""
    global _executor
    with _executor_lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    return _executor


def shutdown_executor():
    """Shutdown the global executor."""
    global _executor
    with _executor_lock:
        if _executor is not None:
            _executor.shutdown(wait=True)
            _executor = None


# =============================================================================
# Thread-Local Clients
# =============================================================================

_thread_local = threading.local()
_all_clients: list[OpenAI] = []
_clients_lock = threading.Lock()


def get_client() -> OpenAI:
    """Get thread-local OpenAI client with timeout."""
    if not hasattr(_thread_local, 'client') or _thread_local.client is None:
        client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
            timeout=REQUEST_TIMEOUT,
        )
        _thread_local.client = client
        with _clients_lock:
            _all_clients.append(client)
    return _thread_local.client


def close_client():
    """Close all thread-local clients."""
    global _all_clients
    with _clients_lock:
        for client in _all_clients:
            try:
                client.close()
            except Exception:
                pass
        _all_clients = []
    if hasattr(_thread_local, 'client'):
        _thread_local.client = None


# =============================================================================
# Retry Logic
# =============================================================================

RETRYABLE_ERRORS = (
    "rate_limit",
    "timeout",
    "overloaded",
    "502",
    "503",
    "504",
    "connection",
    "expecting value",  # JSON parsing error (truncated response)
    "jsondecode",       # JSONDecodeError
    "unterminated string",  # Incomplete JSON
    "invalid control character",  # Malformed JSON
)


def is_retryable_error(error: Exception) -> bool:
    error_str = str(error).lower()
    return any(keyword in error_str for keyword in RETRYABLE_ERRORS)


# =============================================================================
# Usage Extraction Helper
# =============================================================================

def _extract_usage_dict(usage: Any) -> dict[str, Any]:
    """Extract usage information from response, including BYOK costs."""
    if usage is None:
        return {}
    
    usage_dict = {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        "cost": getattr(usage, "cost", 0.0) or 0.0,
        "is_byok": getattr(usage, "is_byok", False),
    }
    
    if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
        usage_dict["completion_tokens_details"] = {
            "reasoning_tokens": getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0
        }
    if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
        usage_dict["prompt_tokens_details"] = {
            "cached_tokens": getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0
        }
    
    cost_details = getattr(usage, "cost_details", None)
    if cost_details and isinstance(cost_details, dict):
        usage_dict["cost_details"] = {
            "upstream_inference_cost": cost_details.get("upstream_inference_cost", 0) or 0,
            "upstream_inference_prompt_cost": cost_details.get("upstream_inference_prompt_cost", 0) or 0,
            "upstream_inference_completions_cost": cost_details.get("upstream_inference_completions_cost", 0) or 0,
        }
    
    return usage_dict


# =============================================================================
# LLM Call Functions
# =============================================================================

def call_llm(
    model: str,
    system_prompt: str,
    user_prompt: str,
    extra_body: dict[str, Any] | None = None,
    max_tokens: int | None = None,
    max_retries: int = 5,
    base_delay: float = 5.0,
    response_format: dict[str, Any] | None = None,
    task_id: str | None = None,
) -> tuple[str, float]:
    """
    Call an LLM with retry logic and usage tracking.

    Returns:
        (response_content, elapsed_seconds)
    """
    client = get_client()

    for attempt in range(max_retries):
        start_time = time.time()
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }

            # IMPORTANT: Include usage tracking in extra_body
            merged_extra_body = extra_body.copy() if extra_body else {}
            merged_extra_body["usage"] = {"include": True}
            kwargs["extra_body"] = merged_extra_body
            
            if max_tokens:
                kwargs["max_tokens"] = max_tokens
            if response_format:
                kwargs["response_format"] = response_format

            response = client.chat.completions.create(**kwargs)
            
            content = response.choices[0].message.content or ""
            elapsed = time.time() - start_time
            
            # Extract and record usage
            usage = getattr(response, "usage", None)
            usage_dict = _extract_usage_dict(usage)
            if usage_dict:
                _record_usage(usage_dict)
                _record_task_usage(task_id, usage_dict)

            return content, elapsed

        except Exception as e:
            elapsed = time.time() - start_time
            if is_retryable_error(e) and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
                continue
            raise

    raise RuntimeError(f"Max retries ({max_retries}) exceeded")


def call_llm_with_tools(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    extra_body: dict[str, Any] | None = None,
    max_tokens: int = 120000,
    max_retries: int = 5,
    base_delay: float = 5.0,
    response_format: dict[str, Any] | None = None,
    task_id: str | None = None,
    tool_choice: dict[str, Any] | str | None = None,
) -> tuple[dict[str, Any], float]:
    """
    Call an LLM with tools (function calling).

    Returns:
        (response_dict with tool_calls and content, elapsed_seconds)
    """
    client = get_client()

    for attempt in range(max_retries):
        start_time = time.time()
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "tools": tools,
                "max_tokens": max_tokens,
            }
            
            if tool_choice:
                kwargs["tool_choice"] = tool_choice
            
            if response_format:
                kwargs["response_format"] = response_format

            # IMPORTANT: Include usage tracking in extra_body
            merged_extra_body = extra_body.copy() if extra_body else {}
            merged_extra_body["usage"] = {"include": True}
            kwargs["extra_body"] = merged_extra_body

            response = client.chat.completions.create(**kwargs)
            
            elapsed = time.time() - start_time

            # Extract and record usage
            usage = getattr(response, "usage", None)
            usage_dict = _extract_usage_dict(usage)
            if usage_dict:
                _record_usage(usage_dict)
                _record_task_usage(task_id, usage_dict)

            choice = response.choices[0]
            message = choice.message
            
            result = {
                "content": message.content or "",
                "tool_calls": [],
                "finish_reason": choice.finish_reason,
            }
            
            if message.tool_calls:
                for tc in message.tool_calls:
                    result["tool_calls"].append({
                        "id": tc.id,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    })

            return result, elapsed

        except Exception as e:
            if is_retryable_error(e) and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
                continue
            raise

    raise RuntimeError(f"Max retries ({max_retries}) exceeded")


def call_llm_with_history(
    model: str,
    messages: list[dict[str, str]],
    extra_body: dict[str, Any] | None = None,
    max_tokens: int | None = None,
    max_retries: int = 5,
    base_delay: float = 5.0,
    task_id: str | None = None,
    response_format: dict[str, Any] | None = None,
) -> tuple[str, float]:
    """
    Call an LLM with a full message history.

    Returns:
        (response_content, elapsed_seconds)
    """
    client = get_client()

    for attempt in range(max_retries):
        start_time = time.time()
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
            }

            # IMPORTANT: Include usage tracking in extra_body
            merged_extra_body = extra_body.copy() if extra_body else {}
            merged_extra_body["usage"] = {"include": True}
            kwargs["extra_body"] = merged_extra_body
            
            if max_tokens:
                kwargs["max_tokens"] = max_tokens
            if response_format:
                kwargs["response_format"] = response_format

            response = client.chat.completions.create(**kwargs)
            
            content = response.choices[0].message.content or ""
            elapsed = time.time() - start_time
            
            # Extract and record usage
            usage = getattr(response, "usage", None)
            usage_dict = _extract_usage_dict(usage)
            if usage_dict:
                _record_usage(usage_dict)
                _record_task_usage(task_id, usage_dict)

            return content, elapsed

        except Exception as e:
            elapsed = time.time() - start_time
            if is_retryable_error(e) and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
                continue
            raise

    raise RuntimeError(f"Max retries ({max_retries}) exceeded")
