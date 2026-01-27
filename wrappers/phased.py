"""
Phased Solver Wrapper

Uses 4-phase structured tool-calling approach:
1. OBSERVE - Document observations
2. HYPOTHESIZE - Formulate hypotheses
3. VERIFY - Verify on training examples
4. IMPLEMENT - Write code and predict
"""

import sys
import os
from typing import Any

import numpy as np

# Add solvers to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from solvers.perceiver.solve.tool_solver import solve_task_with_tools
from solvers.perceiver.llms.client import get_and_reset_usage
from config import TOOL_SOLVER_MODEL


def run_phased_solver(
    task_data: dict,
    task_id: str = "unknown",
    verbose: bool = True,
    # Pre-computed perception (optional - if provided, skips perception stage)
    perceptions: list[dict] | None = None,
    deltas: list[dict] | None = None,
    test_perception: list[dict] | None = None,
    hypotheses: list[dict] | None = None,
    observations: dict | None = None,
    key_insight: str | None = None,
) -> dict[str, Any]:
    """
    Run the phased solver.
    
    If perceptions/deltas/hypotheses are provided, uses them directly.
    Otherwise, runs the full perception pipeline.
    
    Args:
        task_data: Task with 'train' and 'test' keys
        task_id: Task identifier
        verbose: Whether to print progress
        perceptions: Pre-computed perceptions (optional)
        deltas: Pre-computed deltas (optional)
        test_perception: Pre-computed test perceptions (optional)
        hypotheses: Pre-computed hypotheses (optional)
        observations: Pre-computed observations (optional)
        key_insight: Pre-computed key insight (optional)
        
    Returns:
        Dict with:
        - output: The predicted grid from CODE execution (or None if failed)
        - info: Additional information about the solve
        - usage: Token/cost usage stats
    """
    if verbose:
        print(f"  🔶 Phased: Starting (4-phase structured solver)...")
    
    # Reset usage tracking before this solver run
    _ = get_and_reset_usage()
    
    # Determine if we should run perceiver pipeline
    use_shared_perception = perceptions is not None and deltas is not None
    run_perceiver = not use_shared_perception
    
    if use_shared_perception and verbose:
        print(f"    ✓ Using shared perception ({len(perceptions)} examples)")
    
    try:
        # Run the solver
        predictions, info = solve_task_with_tools(
            task_data=task_data,
            task_id=task_id,
            perceptions=perceptions,
            deltas=deltas,
            test_perception=test_perception,
            hypotheses=hypotheses,
            observations=observations,
            key_insight=key_insight,
            ground_truths=None,
            run_perceiver=run_perceiver,
            verbose=verbose,
        )
        
        if info.get("success") and predictions:
            # Get ALL predictions
            outputs = []
            for pred in predictions:
                if isinstance(pred, np.ndarray):
                    outputs.append(pred.tolist())
                else:
                    outputs.append(pred)
            
            # Return single output if only 1 test, otherwise list
            output = outputs[0] if len(outputs) == 1 else outputs
            
            # Get usage stats for this solver run
            usage = get_and_reset_usage()
            
            if verbose:
                print(f"  ✓ Phased: Solution found")
                print(f"    Phases: {info.get('phases_completed', [])}")
                if len(outputs) > 1:
                    print(f"    {len(outputs)} test outputs generated")
            
            return {
                "output": output,
                "info": {
                    "solver": "phased",
                    "model": TOOL_SOLVER_MODEL["id"],
                    "phases_completed": info.get("phases_completed", []),
                    "code_matches_prediction": info.get("code_matches_prediction"),
                    "success": True,
                    "num_tests": len(outputs),
                },
                "usage": usage,
            }
        else:
            # Get usage stats even on failure
            usage = get_and_reset_usage()
            
            if verbose:
                print(f"  ✗ Phased: No solution found")
            
            return {
                "output": None,
                "info": {
                    "solver": "phased",
                    "success": False,
                    "error": info.get("error", "Unknown error") if info else "No result"
                },
                "usage": usage,
            }
            
    except Exception as e:
        # Get usage stats even on error
        usage = get_and_reset_usage()
        
        if verbose:
            print(f"  ✗ Phased error: {e}")
            import traceback
            traceback.print_exc()
        
        return {
            "output": None,
            "info": {
                "solver": "phased",
                "success": False,
                "error": str(e)
            },
            "usage": usage,
        }
