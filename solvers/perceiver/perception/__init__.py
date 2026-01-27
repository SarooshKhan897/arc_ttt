"""Perception module - grid analysis and comparison."""

from solvers.perceiver.perception.objects import ObjectPreprocessor, perceive_grid_fast
from solvers.perceiver.perception.perceiver import perceive, perceive_batch, perceive_task, format_hypotheses_for_solver
from solvers.perceiver.perception.differencer import difference, difference_batch
from solvers.perceiver.perception.analyzer import (
    GridAnalysis,
    TransformAnalysis,
    ExampleAnalysis,
    TaskAnalysis,
    analyze_grid,
    analyze_transform,
    analyze_example,
    analyze_task,
    quick_grid_stats,
    quick_transform_stats,
)
# Hypothesizer functions moved into perceiver.py as perceive_task()

__all__ = [
    # Objects
    "ObjectPreprocessor",
    "perceive_grid_fast",
    # Perceiver
    "perceive",
    "perceive_batch",
    "perceive_task",
    "format_hypotheses_for_solver",
    # Differencer
    "difference",
    "difference_batch",
    # Analyzer
    "GridAnalysis",
    "TransformAnalysis",
    "ExampleAnalysis",
    "TaskAnalysis",
    "analyze_grid",
    "analyze_transform",
    "analyze_example",
    "analyze_task",
    "quick_grid_stats",
    "quick_transform_stats",
]

