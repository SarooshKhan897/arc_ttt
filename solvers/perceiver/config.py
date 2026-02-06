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

# Provider routing - restrict to anthropic only
PROVIDER_CONFIG = {"provider": {"only": ["anthropic"]}}


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
    Role.PERCEIVER: os.getenv("PERCEIVER_MODEL", "anthropic/claude-opus-4.6"),
    Role.DIFFERENCER: os.getenv("DIFFERENCER_MODEL", "anthropic/claude-opus-4.6"),
    Role.SOLVER: os.getenv("SOLVER_MODEL", "anthropic/claude-opus-4.6"),
    Role.VERIFIER: os.getenv("VERIFIER_MODEL", "anthropic/claude-opus-4.6"),
    Role.SELF_VERIFIER: os.getenv("SELF_VERIFIER_MODEL", "anthropic/claude-opus-4.6"),
}

# Role-specific extra configurations (reasoning effort, etc.)
ROLE_CONFIGS: dict[Role, dict[str, Any]] = {
    Role.PERCEIVER: {
        "extra_body": {"reasoning": {"enabled": True}},
    },
    Role.DIFFERENCER: {
        "extra_body": {"reasoning": {"enabled": True}},
    },
}

# Model-specific configurations for the solver models
SOLVER_MODELS = [
    {
        "id": "claude-opus-4.6",
        "model": "anthropic/claude-opus-4.6",
        "extra_body": {"reasoning": {"enabled": True}},
        "tries": 5,
    },
]

# Default model for tool-based solver
TOOL_SOLVER_MODEL_ID = os.getenv("TOOL_SOLVER_MODEL", "claude-opus-4.6")

# =============================================================================
# Solver Settings
# =============================================================================

MIN_SOLUTIONS_REQUIRED = 2  # Stop when we have this many high-confidence solutions
MIN_CONFIDENCE_SCORE = 90   # Minimum score to count as high-confidence

# Model ranking for fallback selection (lower index = higher priority)
# Used when no self-verified solutions, pick training-passed by this order
MODEL_RANK = [
    "claude-opus-4.6",         # Highest priority
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
    return ROLE_MODELS.get(role, "anthropic/claude-opus-4.6")


def get_role_extra_body(role: Role) -> dict[str, Any] | None:
    """Get extra_body config for a specific role (e.g., reasoning effort)."""
    config = ROLE_CONFIGS.get(role, {})
    return config.get("extra_body")
