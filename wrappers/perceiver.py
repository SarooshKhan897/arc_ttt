"""
Perceiver Solver Wrapper

Uses perception-based exhaustive solver approach:
- Perceiver pipeline (perceive_batch for all grids)
- Differencer (difference_batch for deltas)
- Task perceiver (perceive_task for hypotheses + key insight)
- Voting/exhaustive solver approach
"""

import sys
import os
from typing import Any

import numpy as np

# Add solvers to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from solvers.perceiver.solve.solver import solve_single_exhaustive
from solvers.perceiver.perception import perceive_batch, difference_batch, perceive_task
from solvers.perceiver.llms.client import get_and_reset_usage
from config import ARC_SOLVER_MODELS


def run_perceiver_solver(
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
    Run the perceiver solver.
    
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
        - output: The predicted grid (or None if failed)
        - info: Additional information about the solve
        - usage: Token/cost usage stats
    """
    if verbose:
        print(f"  🔷 Perceiver: Starting (perception-based exhaustive solver)...")
    
    # Reset usage tracking before this solver run
    _ = get_and_reset_usage()
    
    try:
        train = task_data['train']
        test_cases = task_data['test']
        test_inputs = [np.array(t['input']) for t in test_cases]
        n_examples = len(train)
        
        # Check if we have pre-computed perception
        use_shared_perception = perceptions is not None and deltas is not None
        
        if use_shared_perception:
            if verbose:
                print(f"    ✓ Using shared perception ({len(perceptions)} examples)")
            # Use provided values
            solver_hypotheses = hypotheses
            solver_observations = observations
            solver_key_insight = key_insight
            solver_test_perception = test_perception
        else:
            # =====================================================================
            # Step 1: Perceive all grids (training + ALL test inputs)
            # =====================================================================
            if verbose:
                print(f"    👁️ Perceiving grids...")
            
            all_grids = []
            for pair in train:
                all_grids.append(np.array(pair['input']))
                all_grids.append(np.array(pair['output']))
            # Add ALL test inputs
            for ti in test_inputs:
                all_grids.append(ti)
            
            all_perceptions = perceive_batch(all_grids, verbose=False)
            
            # Organize perceptions
            perceptions = []
            for i in range(n_examples):
                perceptions.append({
                    'input': all_perceptions[i * 2],
                    'output': all_perceptions[i * 2 + 1],
                })
            
            # Test perceptions - ALL of them
            test_perceptions_list = all_perceptions[n_examples * 2:]
            solver_test_perception = test_perceptions_list if test_perceptions_list else None
            
            if verbose:
                print(f"    ✓ Perceived {len(all_grids)} grids")
            
            # =====================================================================
            # Step 2: Compute deltas
            # =====================================================================
            if verbose:
                print(f"    🔍 Computing deltas...")
            
            pairs = [(np.array(p['input']), np.array(p['output'])) for p in train]
            perc_pairs = [(perceptions[i]['input'], perceptions[i]['output']) for i in range(n_examples)]
            deltas = difference_batch(pairs, perc_pairs, verbose=False)
            
            if verbose:
                print(f"    ✓ Computed {len(deltas)} deltas")
            
            # =====================================================================
            # Step 3: Perceiver generates transformation hypotheses
            # =====================================================================
            if verbose:
                print(f"    🔮 Generating hypotheses...")
            
            task_perception = perceive_task(task_data, verbose=False)
            raw_hypotheses = task_perception.get("transformation_hypotheses", [])
            solver_observations = task_perception.get("observations")
            solver_key_insight = task_perception.get("key_insight", "")
            
            # Convert to format expected by solver
            solver_hypotheses = []
            for h in raw_hypotheses:
                solver_hypotheses.append({
                    "rank": h.get("rank", 1),
                    "confidence": h.get("confidence", "MEDIUM"),
                    "rule": h.get("rule", ""),
                    "evidence": h.get("evidence", "")
                })
            
            if verbose:
                print(f"    ✓ Generated {len(solver_hypotheses)} hypotheses")
                if solver_key_insight:
                    print(f"    💡 Key insight: {solver_key_insight[:60]}...")
        
        # =====================================================================
        # Step 4: Run the solver
        # =====================================================================
        if verbose:
            print(f"    🚀 Running solver...")
        
        model_config = ARC_SOLVER_MODELS[0]
        
        candidates = solve_single_exhaustive(
            task_data=task_data,
            model_config=model_config,
            perceptions=perceptions,
            deltas=deltas,
            test_perception=solver_test_perception if use_shared_perception else solver_test_perception,
            hypotheses=solver_hypotheses if solver_hypotheses else None,
            observations=solver_observations,
            key_insight=solver_key_insight if solver_key_insight else None,
            max_tries=5,
            verbose=verbose,
            early_stop=True,
            skip_self_verify=True,
        )
        
        if candidates:
            # Get best candidate
            best = candidates[0]
            
            # Extract ALL test outputs
            outputs = []
            if best.test_results:
                for test_result in best.test_results:
                    if isinstance(test_result, np.ndarray):
                        outputs.append(test_result.tolist())
                    else:
                        outputs.append(test_result)
            
            # Return single output if only 1 test, otherwise list
            output = outputs[0] if len(outputs) == 1 else outputs
            
            # Get usage stats for this solver run
            usage = get_and_reset_usage()
            
            if verbose:
                print(f"  ✓ Perceiver: Solution found (score: {best.verifier_score})")
                if len(outputs) > 1:
                    print(f"    {len(outputs)} test outputs generated")
            
            return {
                "output": output,
                "info": {
                    "solver": "perceiver",
                    "model": model_config["id"],
                    "verifier_score": best.verifier_score,
                    "explanation": best.explanation[:200] if best.explanation else "",
                    "success": True,
                    "num_tests": len(outputs),
                },
                "usage": usage,
            }
        else:
            # Get usage stats even on failure
            usage = get_and_reset_usage()
            
            if verbose:
                print(f"  ✗ Perceiver: No solution found")
            
            return {
                "output": None,
                "info": {
                    "solver": "perceiver",
                    "success": False,
                    "error": "No passing solution found"
                },
                "usage": usage,
            }
            
    except Exception as e:
        # Get usage stats even on error
        usage = get_and_reset_usage()
        
        if verbose:
            print(f"  ✗ Perceiver error: {e}")
            import traceback
            traceback.print_exc()
        
        return {
            "output": None,
            "info": {
                "solver": "perceiver",
                "success": False,
                "error": str(e)
            },
            "usage": usage,
        }
