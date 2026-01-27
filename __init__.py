"""
ARC Ensemble - Multi-solver system for ARC-AGI puzzles.

Solvers:
- perceiver: Perception-based exhaustive solver
- phased: 4-phase structured tool-calling solver  
- iterative: Iterative refinement solver

The orchestrator runs all solvers and uses a judge to pick the best outputs.
"""

__version__ = "1.0.0"
