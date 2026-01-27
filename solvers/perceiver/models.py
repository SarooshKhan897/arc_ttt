"""Data models for the ARC Solver."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pydantic import BaseModel


# =============================================================================
# Grid Types
# =============================================================================

GRID = list[list[int]]

# Color names for display
COLOR_NAMES = [
    'black(0)', 'blue(1)', 'red(2)', 'green(3)', 'yellow(4)',
    'gray(5)', 'magenta(6)', 'orange(7)', 'azure(8)', 'maroon(9)'
]


def color_name(c: int) -> str:
    """Get human-readable color name."""
    return COLOR_NAMES[c] if 0 <= c < len(COLOR_NAMES) else f"color({c})"


# =============================================================================
# Task Models
# =============================================================================

class Example(BaseModel):
    """A single training example with input and output grids."""
    input: GRID
    output: GRID


class TestInput(BaseModel):
    """A test input (may or may not have ground truth output)."""
    input: GRID
    output: GRID | None = None


class Task(BaseModel):
    """An ARC task with training examples and test inputs."""
    task_id: str = ""
    train: list[Example]
    test: list[TestInput]

    @property
    def n_train(self) -> int:
        return len(self.train)

    @property
    def n_test(self) -> int:
        return len(self.test)


# =============================================================================
# Perception Models
# =============================================================================

@dataclass
class EnhancedObject:
    """A detected object in a grid with enhanced analysis."""
    pixels: set[tuple[int, int]]
    color: int
    bounding_box: tuple[int, int, int, int]  # (min_row, min_col, max_row, max_col)
    size: int
    is_rectangle: bool
    centroid: tuple[float, float]
    neighbors: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "color": color_name(int(self.color)),
            "size": int(self.size),
            "bounding_box": tuple(int(x) for x in self.bounding_box),
            "is_rectangle": bool(self.is_rectangle),
            "centroid": tuple(float(x) for x in self.centroid),
        }


@dataclass
class GridPerception:
    """Structured output from the Perceiver."""
    grid_shape: tuple[int, int]
    background_color: int
    objects: list[EnhancedObject]
    symmetry: dict[str, bool]
    patterns: list[str]
    color_counts: dict[int, int]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "grid_shape": tuple(int(x) for x in self.grid_shape),
            "background_color": color_name(int(self.background_color)),
            "objects": [obj.to_dict() for obj in self.objects],
            "symmetry": {k: bool(v) for k, v in self.symmetry.items()},
            "patterns": list(self.patterns),
            "color_counts": {color_name(int(k)): int(v) for k, v in self.color_counts.items()},
        }


@dataclass
class TransformDelta:
    """Structured output from the Differencer."""
    object_changes: list[dict[str, Any]]
    color_changes: list[dict[str, Any]]
    size_change: tuple[int, int]
    structural_changes: list[str]
    constants: list[str]
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "object_changes": self.object_changes,
            "color_changes": self.color_changes,
            "size_change": tuple(int(x) for x in self.size_change),
            "structural_changes": list(self.structural_changes),
            "constants": list(self.constants),
            "summary": str(self.summary),
        }


# =============================================================================
# Solution Models
# =============================================================================

@dataclass
class SolutionCandidate:
    """A candidate solution from one solving attempt."""
    code: str
    explanation: str
    model_id: str
    verifier_score: int
    verifier_verdict: str
    self_verify_decision: str = "PENDING"
    attempts: int = 1
    # Now stores results for ALL test inputs
    test_results: list[np.ndarray] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_id": self.model_id,
            "verifier_score": self.verifier_score,
            "verifier_verdict": self.verifier_verdict,
            "self_verify_decision": self.self_verify_decision,
            "attempts": self.attempts,
            "num_test_results": len(self.test_results),
        }


# =============================================================================
# Result Models
# =============================================================================

class TaskResult(BaseModel):
    """Result of solving a single task with multiple test inputs."""
    task_id: str
    # predictions[i] = list of predictions for test_input[i] (2 attempts each)
    predictions: list[list[GRID]]  # [test_idx][attempt_idx]
    n_test_cases: int = 1
    n_correct: int = 0  # Number of test cases where any attempt was correct
    score: float = 0.0  # n_correct / n_test_cases (fractional)
    per_test_correct: list[bool] = []  # Per-test correctness
    error: str | None = None
    solve_info: dict[str, Any] = {}


# =============================================================================
# Evaluator
# =============================================================================

class ArcEvaluator:
    """Evaluator for ARC predictions following official scoring."""

    @staticmethod
    def exact_match(pred: np.ndarray, target: np.ndarray) -> bool:
        """Check if prediction exactly matches target."""
        return np.array_equal(pred, target)

    @staticmethod
    def cell_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
        """Calculate percentage of cells that match."""
        if pred.shape != target.shape:
            return 0.0
        return float(np.mean(pred == target))

    @staticmethod
    def evaluate_single_test(
        attempts: list[np.ndarray],
        ground_truth: np.ndarray
    ) -> tuple[bool, float]:
        """
        Evaluate multiple attempts for a SINGLE test case.
        A test case is correct if ANY attempt is correct.

        Args:
            attempts: List of prediction attempts for one test case
            ground_truth: The expected output

        Returns:
            (any_attempt_correct, best_accuracy)
        """
        if not attempts:
            return False, 0.0

        exact = any(ArcEvaluator.exact_match(p, ground_truth) for p in attempts)
        accuracies = [ArcEvaluator.cell_accuracy(p, ground_truth) for p in attempts]
        return exact, max(accuracies)

    @staticmethod
    def evaluate_task(
        predictions_per_test: list[list[np.ndarray]],
        ground_truths: list[np.ndarray]
    ) -> tuple[int, int, float, list[bool]]:
        """
        Evaluate predictions for ALL test cases in a task.
        Official ARC scoring: score = correct_tests / total_tests

        Args:
            predictions_per_test: predictions_per_test[i] = list of attempts for test i
            ground_truths: ground_truths[i] = expected output for test i

        Returns:
            (n_correct, n_total, score, per_test_correct)
        """
        if not ground_truths:
            return 0, 0, 0.0, []

        n_total = len(ground_truths)
        per_test_correct = []

        for i, gt in enumerate(ground_truths):
            attempts = predictions_per_test[i] if i < len(predictions_per_test) else []
            is_correct, _ = ArcEvaluator.evaluate_single_test(attempts, gt)
            per_test_correct.append(is_correct)

        n_correct = sum(per_test_correct)
        score = n_correct / n_total

        return n_correct, n_total, score, per_test_correct

    # Legacy method for backwards compatibility
    @staticmethod
    def evaluate_predictions(
        predictions: list[np.ndarray],
        ground_truth: np.ndarray
    ) -> tuple[bool, float]:
        """Legacy: Evaluate predictions against single ground truth."""
        return ArcEvaluator.evaluate_single_test(predictions, ground_truth)

