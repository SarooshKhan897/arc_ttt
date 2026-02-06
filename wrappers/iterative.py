"""
Iterative Solver Wrapper

Uses iterative refinement with code execution:
- Code-based task analysis
- LLM-based hypothesis generation
- Phase 0 structured observation
- Iterative tool-calling workflow with code execution
"""

import sys
import os
from typing import Any

import numpy as np

# Add solvers to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    CLAUDE_V3_MODEL,
    CLAUDE_V3_MAX_ITERATIONS,
)
from solvers.iterative.solver import ARCSolver


def run_iterative_solver(
    task_data: dict,
    task_id: str = "unknown",
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run the iterative solver independently.
    
    This solver has its OWN complete pipeline:
    - Code-based task analysis
    - LLM-based hypothesis generation
    - Phase 0 structured observation
    - Iterative tool-calling workflow with code execution
    
    Args:
        task_data: Task with 'train' and 'test' keys
        task_id: Task identifier
        verbose: Whether to print progress
        
    Returns:
        Dict with:
        - output: The predicted grid (or None if failed)
        - info: Additional information about the solve
        - usage: Token/cost usage stats
    """
    if verbose:
        print(f"  🔵 Iterative: Starting (iterative refinement solver)...")
    
    try:
        # Create solver with full config
        solver = ARCSolver(
            api_key=OPENROUTER_API_KEY,
            model=CLAUDE_V3_MODEL,
            base_url=OPENROUTER_BASE_URL,
            max_iterations=CLAUDE_V3_MAX_ITERATIONS,
            verbose=verbose,
            hypothesis_model=CLAUDE_V3_MODEL,
        )
        
        # Run the solver
        result = solver.solve(task_data, return_history=False)
        
        if result and result.get("success"):
            # Get the answer
            answer = result.get("answer")
            
            output = None
            num_tests = 1
            if answer is not None:
                # Handle different answer formats
                if isinstance(answer, np.ndarray):
                    output = answer.tolist()
                elif isinstance(answer, list) and len(answer) > 0:
                    if isinstance(answer[0], list) and len(answer[0]) > 0:
                        if isinstance(answer[0][0], list):
                            # List of grids (multiple tests)
                            output = answer
                            num_tests = len(answer)
                        else:
                            # Single grid as list of rows
                            output = answer
                    else:
                        output = answer
            
            # Get usage stats from solver
            usage = solver.get_usage()
            
            if verbose:
                print(f"  ✓ Iterative: Solution found")
                print(f"    Iterations: {result.get('iterations', 'N/A')}")
                print(f"    Tool calls: {result.get('tool_calls_made', 'N/A')}")
                if num_tests > 1:
                    print(f"    {num_tests} test outputs generated")
            
            return {
                "output": output,
                "info": {
                    "solver": "iterative",
                    "model": CLAUDE_V3_MODEL,
                    "iterations": result.get("iterations", 0),
                    "tool_calls_made": result.get("tool_calls_made", 0),
                    "rule": result.get("rule", ""),
                    "success": True,
                    "num_tests": num_tests,
                },
                "usage": usage,
            }
        else:
            # Try attempt_1 or attempt_2 as fallback
            attempt_1 = result.get("attempt_1") if result else None
            attempt_2 = result.get("attempt_2") if result else None
            
            fallback = attempt_1 or attempt_2
            if fallback and fallback.get("answer"):
                output = fallback["answer"]
                if isinstance(output, np.ndarray):
                    output = output.tolist()
                
                # Handle multi-test outputs
                num_tests = 1
                if isinstance(output, list) and len(output) > 0:
                    if isinstance(output[0], list) and len(output[0]) > 0:
                        if isinstance(output[0][0], list):
                            num_tests = len(output)
                
                # Get usage stats from solver
                usage = solver.get_usage()
                
                if verbose:
                    print(f"  ⚠️ Iterative: Using fallback solution")
                    if num_tests > 1:
                        print(f"    {num_tests} test outputs generated")
                
                return {
                    "output": output,
                    "info": {
                        "solver": "iterative",
                        "model": CLAUDE_V3_MODEL,
                        "success": True,
                        "fallback": True,
                        "num_tests": num_tests,
                    },
                    "usage": usage,
                }
            
            # Get usage stats even on failure
            usage = solver.get_usage()
            
            if verbose:
                print(f"  ✗ Iterative: No solution found")
            
            return {
                "output": None,
                "info": {
                    "solver": "iterative",
                    "success": False,
                    "error": "No solution found"
                },
                "usage": usage,
            }
            
    except Exception as e:
        if verbose:
            print(f"  ✗ Iterative error: {e}")
            import traceback
            traceback.print_exc()
        
        return {
            "output": None,
            "info": {
                "solver": "iterative",
                "success": False,
                "error": str(e)
            },
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "total_cost": 0},
        }
