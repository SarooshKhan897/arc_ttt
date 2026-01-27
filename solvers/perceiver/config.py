"""Configuration for the ARC Solver."""

import os
from enum import Enum
from typing import Any

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# API Configuration
# =============================================================================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# Request timeout
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "1800.0"))  # 30 min

# Concurrency
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "100"))


# =============================================================================
# Role Definitions
# =============================================================================

class Role(Enum):
    """Specialist roles in the solving pipeline."""
    PERCEIVER = "perceiver"       # Structured grid analysis
    DIFFERENCER = "differencer"   # I/O comparison
    SOLVER = "solver"             # Code generation
    VERIFIER = "verifier"         # Solution validation
    SELF_VERIFIER = "self_verifier"  # Self-verification


# =============================================================================
# Model Configurations
# =============================================================================

# Default model assignments by role
ROLE_MODELS = {
    Role.PERCEIVER: os.getenv("PERCEIVER_MODEL", "google/gemini-3-pro-preview"),
    Role.DIFFERENCER: os.getenv("DIFFERENCER_MODEL", "google/gemini-3-pro-preview"),
    Role.SOLVER: os.getenv("SOLVER_MODEL", "openai/gpt-5.2"),
    Role.VERIFIER: os.getenv("VERIFIER_MODEL", "openai/gpt-5.2"),
    Role.SELF_VERIFIER: os.getenv("SELF_VERIFIER_MODEL", "openai/gpt-5.2"),
}

# Role-specific extra configurations (reasoning effort, etc.)
ROLE_CONFIGS: dict[Role, dict[str, Any]] = {
    Role.PERCEIVER: {
        "extra_body": {"reasoning": {"effort": "xhigh"}},
    },
    Role.DIFFERENCER: {
        "extra_body": {"reasoning": {"effort": "xhigh"}},
    },
}

# Model-specific configurations for the solver models
SOLVER_MODELS = [
    {
        "id": "gpt-5.2",
        "model": "openai/gpt-5.2",
        "extra_body": {"reasoning": {"effort": "xhigh"}},
        "max_tokens": 120000,
        "tries": 5,
    },
    {
        "id": "gpt-5.2-high",
        "model": "openai/gpt-5.2",
        "extra_body": {"reasoning": {"effort": "high"}},
        "max_tokens": 120000,
        "tries": 5,
    },
    {
        "id": "claude-opus-4.5",
        "model": "anthropic/claude-opus-4.5",
        "extra_body": {"reasoning": {"effort": "high"}},
        "max_tokens": 120000,
        "tries": 10,
    },
    {
        "id": "gemini-pro",
        "model": "google/gemini-3-pro-preview",
        "extra_body": {"reasoning": {"effort": "xhigh"}},
        "max_tokens": None,
        "tries": 10,
    },
]

# Default model for tool-based solver
TOOL_SOLVER_MODEL_ID = os.getenv("TOOL_SOLVER_MODEL", "gpt-5.2")

# =============================================================================
# Solver Settings
# =============================================================================

MIN_SOLUTIONS_REQUIRED = 2  # Stop when we have this many high-confidence solutions
MIN_CONFIDENCE_SCORE = 90   # Minimum score to count as high-confidence

# Model ranking for fallback selection (lower index = higher priority)
# Used when no self-verified solutions, pick training-passed by this order
MODEL_RANK = [
    "gpt-5.2",           # Highest priority (xhigh reasoning)
    "gpt-5.2-high",      # Second priority (high reasoning fallback)
    "gemini-flash",      # Third priority
    "gemini-pro",        # Fourth priority
]


# =============================================================================
# Helper Functions
# =============================================================================

def get_model_config(model_id: str) -> dict[str, Any] | None:
    """Get configuration for a specific solver model."""
    for cfg in SOLVER_MODELS:
        if cfg["id"] == model_id:
            return cfg
    return None


def get_role_model(role: Role) -> str:
    """Get the model assigned to a specific role."""
    return ROLE_MODELS.get(role, "openai/gpt-5.2")


def get_role_extra_body(role: Role) -> dict[str, Any] | None:
    """Get extra_body config for a specific role (e.g., reasoning effort)."""
    config = ROLE_CONFIGS.get(role, {})
    return config.get("extra_body")

