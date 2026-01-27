"""
Solver Wrappers - Thin wrappers that call the underlying solvers.

Each wrapper:
1. Handles setup and configuration
2. Calls the underlying solver
3. Captures and returns usage stats
4. Formats output consistently
"""

# Import wrapper functions for easy access
from wrappers.perceiver import run_perceiver_solver
from wrappers.phased import run_phased_solver
from wrappers.iterative import run_iterative_solver

__all__ = [
    "run_perceiver_solver",
    "run_phased_solver", 
    "run_iterative_solver",
]
