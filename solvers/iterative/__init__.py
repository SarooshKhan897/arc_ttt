"""
Iterative Solver - Uses iterative refinement with code execution.

This solver:
1. Analyzes the task using code
2. Generates hypotheses via LLM
3. Iteratively refines through tool calls
4. Executes Python code to verify solutions
"""

# Imported directly by wrappers to avoid circular imports
__all__ = []
