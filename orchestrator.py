"""
ARC Ensemble Orchestrator

Runs multiple independent solvers and uses a judge to pick the best outputs.

Solvers (each run twice = 6 candidates):
- Perceiver: Perception-based exhaustive solver
- Phased: 4-phase structured tool-calling solver
- Iterative: Iterative refinement solver

Flow:
1. Run all 3 solvers TWICE each (6 total candidates)
2. Filter to distinct outputs based on test output grids
3. If >2 distinct → use judge to pick top 2
4. If ≤2 distinct → use them directly as attempts
"""

import json
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np

from config import reset_usage_stats, record_usage  # For backwards compatibility
from judge import run_judge
from wrappers.perceiver import run_perceiver_solver
from wrappers.phased import run_phased_solver
from wrappers.iterative import run_iterative_solver

# Import perception functions for shared perception
from solvers.perceiver.perception import perceive_batch, difference_batch, perceive_task


# =============================================================================
# Helper Functions
# =============================================================================

def grid_to_hash(grid: list | np.ndarray | None) -> str:
    """Convert grid to a hashable string for deduplication."""
    if grid is None:
        return "NONE"
    if isinstance(grid, np.ndarray):
        grid = grid.tolist()
    return hashlib.md5(json.dumps(grid, sort_keys=True).encode()).hexdigest()


def get_distinct_outputs_with_votes(outputs: dict[str, dict]) -> tuple[dict[str, dict], dict[str, int], dict[str, list[str]]]:
    """
    Filter outputs to distinct grids and count votes for each.
    
    Returns:
        - distinct: dict mapping representative solver_name to output_data
        - vote_counts: dict mapping representative solver_name to vote count
        - voters: dict mapping representative solver_name to list of all solvers that voted for it
    """
    hash_to_solver = {}  # hash -> first solver that produced it (representative)
    hash_to_voters = {}  # hash -> list of all solvers that produced it
    
    # Process in consistent order - both runs of each solver
    solver_order = [
        "perceiver_1", "perceiver_2",
        "phased_1", "phased_2",
        "iterative_1", "iterative_2"
    ]
    
    for solver_name in solver_order:
        if solver_name not in outputs:
            continue
        output = outputs[solver_name].get("output")
        h = grid_to_hash(output)
        
        if h == "NONE":
            continue  # Skip None outputs
            
        if h not in hash_to_solver:
            hash_to_solver[h] = solver_name
            hash_to_voters[h] = []
        
        hash_to_voters[h].append(solver_name)
    
    # Build return values
    distinct = {}
    vote_counts = {}
    voters = {}
    
    for h, representative in hash_to_solver.items():
        distinct[representative] = outputs[representative]
        vote_counts[representative] = len(hash_to_voters[h])
        voters[representative] = hash_to_voters[h]
    
    return distinct, vote_counts, voters


def get_distinct_outputs(outputs: dict[str, dict]) -> dict[str, dict]:
    """
    Legacy wrapper - returns just distinct outputs for backward compatibility.
    """
    distinct, _, _ = get_distinct_outputs_with_votes(outputs)
    return distinct


def build_hypothesis_summary_for_judge(outputs: dict[str, dict]) -> dict:
    """
    Build a minimal hypothesis summary from solver outputs for the judge.
    Since each solver runs independently, we extract key info from their results.
    """
    summary = {
        "solvers_run": list(outputs.keys()),
        "key_patterns": [],
        "audit_rule": {
            "description": "Judge should evaluate based on correctness, pattern matching, and consistency",
            "checkpoints": [
                "Output dimensions match expected pattern",
                "Colors are valid (from training outputs)",
                "Structure follows transformation rule",
                "Edge cases handled correctly"
            ]
        }
    }
    return summary


def get_base_solver_name(solver_name: str) -> str:
    """Extract base solver name (e.g., 'perceiver_1' -> 'arc_solver')."""
    if solver_name.endswith("_1") or solver_name.endswith("_2"):
        return solver_name[:-2]
    return solver_name


def aggregate_usage(outputs: dict[str, dict], judge_usage: dict | None = None) -> dict[str, Any]:
    """
    Aggregate usage stats from all solver outputs + judge.
    
    Args:
        outputs: Dict mapping solver_name to output data (each has 'usage' key)
        judge_usage: Optional usage dict from judge LLM calls
        
    Returns:
        Aggregated usage dict
    """
    total = {
        "total_calls": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "total_cost": 0.0,
        "total_reasoning_tokens": 0,
        "total_cached_tokens": 0,
        "is_byok": False,
        "upstream_cost": 0.0,
        "upstream_prompt_cost": 0.0,
        "upstream_completion_cost": 0.0,
    }
    
    # Aggregate from all solver outputs
    for solver_name, output_data in outputs.items():
        usage = output_data.get("usage", {})
        if usage:
            total["total_calls"] += usage.get("total_calls", 0)
            total["total_prompt_tokens"] += usage.get("total_prompt_tokens", 0) or usage.get("prompt_tokens", 0)
            total["total_completion_tokens"] += usage.get("total_completion_tokens", 0) or usage.get("completion_tokens", 0)
            total["total_tokens"] += usage.get("total_tokens", 0)
            total["total_cost"] += usage.get("total_cost", 0.0) or usage.get("effective_cost", 0.0)
            total["total_reasoning_tokens"] += usage.get("total_reasoning_tokens", 0)
            total["total_cached_tokens"] += usage.get("total_cached_tokens", 0)
            if usage.get("is_byok"):
                total["is_byok"] = True
            total["upstream_cost"] += usage.get("upstream_cost", 0.0)
            total["upstream_prompt_cost"] += usage.get("upstream_prompt_cost", 0.0)
            total["upstream_completion_cost"] += usage.get("upstream_completion_cost", 0.0)
    
    # Add judge usage if provided
    if judge_usage:
        total["total_calls"] += judge_usage.get("total_calls", 1)
        total["total_prompt_tokens"] += judge_usage.get("prompt_tokens", 0)
        total["total_completion_tokens"] += judge_usage.get("completion_tokens", 0)
        total["total_tokens"] += judge_usage.get("total_tokens", 0)
        total["total_cost"] += judge_usage.get("cost", 0.0)
        if judge_usage.get("is_byok"):
            total["is_byok"] = True
        cost_details = judge_usage.get("cost_details", {})
        if cost_details:
            total["upstream_cost"] += cost_details.get("upstream_inference_cost", 0.0)
            total["upstream_prompt_cost"] += cost_details.get("upstream_inference_prompt_cost", 0.0)
            total["upstream_completion_cost"] += cost_details.get("upstream_inference_completions_cost", 0.0)
    
    # Calculate effective cost
    total["effective_cost"] = total["upstream_cost"] if total["is_byok"] else total["total_cost"]
    
    return total


# =============================================================================
# Main Orchestrator
# =============================================================================

def solve_with_judge(
    task_data: dict,
    task_id: str = "unknown",
    ground_truths: list[np.ndarray] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Solve an ARC task using triple solver x2 + judge approach.
    
    Flow:
    1. Run Perceiver, Phased, Iterative each TWICE in parallel (6 total)
    2. Filter to distinct outputs based on test output grids
    3. If >2 distinct → use judge to pick top 2
    4. If ≤2 distinct → use them directly as attempts
    
    Args:
        task_data: Task with 'train' and 'test' keys
        task_id: Task identifier
        ground_truths: Optional ground truth outputs for verification
        verbose: Whether to print progress
        
    Returns:
        Dict with:
        - attempt_1: First submission (from top-rated solver)
        - attempt_2: Second submission (from second-rated solver)
        - all_outputs: Dict of all solver outputs
        - ratings: Judge ratings for each solver
        - scores: If ground_truths provided, scores for each attempt
    """
    start_time = time.time()
    reset_usage_stats()  # Reset global stats for backwards compatibility
    
    # Always print task start (minimal)
    print(f"[{task_id}] Starting...", end=" ", flush=True)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"TRIPLE SOLVER x2 + JUDGE: Task {task_id}")
        print(f"{'='*60}")
    
    # Reset usage stats
    reset_usage_stats()
    
    # =========================================================================
    # Run ALL Solvers in Parallel
    # - Iterative×2 start immediately
    # - Perception runs once in parallel, then perceiver/phased use shared results
    # =========================================================================
    
    if verbose:
        print(f"\n🚀 Running all solvers in parallel...")
        print(f"    Iterative×2 start immediately")
        print(f"    Perception runs once, shared by Perceiver×2 + Phased×2")
    
    phase1_start = time.time()
    
    outputs = {}
    solver_times = {}
    
    # Shared perception results (will be populated by perception thread)
    shared_perception = {"ready": False, "data": None}
    import threading
    perception_lock = threading.Lock()
    perception_done = threading.Event()
    
    def run_shared_perception():
        """Run perception once, store results for perceiver/phased solvers."""
        perc_start = time.time()
        
        train = task_data['train']
        test_cases = task_data['test']
        test_inputs = [np.array(t['input']) for t in test_cases]
        n_examples = len(train)
        
        if verbose:
            print(f"  👁️ Perception: Starting shared pipeline...")
        
        # Step 1: Perceive all grids
        all_grids = []
        for pair in train:
            all_grids.append(np.array(pair['input']))
            all_grids.append(np.array(pair['output']))
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
        test_perception = all_perceptions[n_examples * 2:] or None
        
        # Step 2: Compute deltas
        pairs = [(np.array(p['input']), np.array(p['output'])) for p in train]
        perc_pairs = [(perceptions[i]['input'], perceptions[i]['output']) for i in range(n_examples)]
        deltas = difference_batch(pairs, perc_pairs, verbose=False)
        
        # Step 3: Generate hypotheses
        task_perception_result = perceive_task(task_data, verbose=False)
        raw_hypotheses = task_perception_result.get("transformation_hypotheses", [])
        observations = task_perception_result.get("observations")
        key_insight = task_perception_result.get("key_insight", "")
        
        # Convert to solver format
        hypotheses = []
        for h in raw_hypotheses:
            hypotheses.append({
                "rank": h.get("rank", 1),
                "confidence": h.get("confidence", "MEDIUM"),
                "rule": h.get("rule", ""),
                "evidence": h.get("evidence", "")
            })
        
        perc_time = time.time() - perc_start
        if verbose:
            print(f"  ✓ Perception: Done in {perc_time:.1f}s (shared by 4 solvers)")
        
        # Store results and signal completion
        with perception_lock:
            shared_perception["data"] = {
                "perceptions": perceptions,
                "deltas": deltas,
                "test_perception": test_perception,
                "hypotheses": hypotheses,
                "observations": observations,
                "key_insight": key_insight,
            }
            shared_perception["ready"] = True
        perception_done.set()
    
    def run_solver_with_shared_perception(solver_func, solver_name):
        """Wait for shared perception, then run solver."""
        # Wait for perception to complete
        perception_done.wait()
        
        # Get shared perception data
        with perception_lock:
            data = shared_perception["data"]
        
        if verbose:
            print(f"  🔷 {solver_name}: Using shared perception, starting solver...")
        
        return solver_func(
            task_data, task_id, verbose,
            data["perceptions"], data["deltas"], data["test_perception"],
            data["hypotheses"], data["observations"], data["key_insight"]
        )
    
    with ThreadPoolExecutor(max_workers=7) as executor:  # 7 = 6 solvers + 1 perception
        futures = {}
        
        # Start perception thread first
        perception_future = executor.submit(run_shared_perception)
        
        # Iterative x2 - start IMMEDIATELY (no perception needed)
        futures[executor.submit(run_iterative_solver, task_data, task_id, verbose)] = "iterative_1"
        futures[executor.submit(run_iterative_solver, task_data, task_id, verbose)] = "iterative_2"
        
        # Perceiver x2 - wait for perception, then run
        futures[executor.submit(run_solver_with_shared_perception, run_perceiver_solver, "perceiver_1")] = "perceiver_1"
        futures[executor.submit(run_solver_with_shared_perception, run_perceiver_solver, "perceiver_2")] = "perceiver_2"
        
        # Phased x2 - wait for perception, then run
        futures[executor.submit(run_solver_with_shared_perception, run_phased_solver, "phased_1")] = "phased_1"
        futures[executor.submit(run_solver_with_shared_perception, run_phased_solver, "phased_2")] = "phased_2"
        
        completed_solvers = []
        for future in as_completed(futures):
            solver_name = futures[future]
            try:
                result = future.result()
                outputs[solver_name] = result
                solver_times[solver_name] = time.time() - phase1_start
                # Minimal progress: show solver completed with checkmark or X
                has_output = result.get("output") is not None
                completed_solvers.append(solver_name.split("_")[0][0].upper())  # P, I, or P
                if not verbose:
                    print("✓" if has_output else "✗", end="", flush=True)
            except Exception as e:
                if verbose:
                    print(f"  ⚠️ {solver_name} failed: {e}")
                else:
                    print("✗", end="", flush=True)
                outputs[solver_name] = {
                    "output": None,
                    "info": {"solver": solver_name, "success": False, "error": str(e)}
                }
                solver_times[solver_name] = time.time() - phase1_start
        
        # Wait for perception to complete (should already be done)
        perception_future.result()
    
    phase1_time = time.time() - phase1_start
    
    if verbose:
        print(f"\n  ⏱️ Phase 1 completed in {phase1_time:.1f}s")
        print(f"    Perceiver #1: {solver_times.get('perceiver_1', 0):.1f}s")
        print(f"    Perceiver #2: {solver_times.get('perceiver_2', 0):.1f}s")
        print(f"    Phased #1: {solver_times.get('phased_1', 0):.1f}s")
        print(f"    Phased #2: {solver_times.get('phased_2', 0):.1f}s")
        print(f"    Iterative #1: {solver_times.get('iterative_1', 0):.1f}s")
        print(f"    Iterative #2: {solver_times.get('iterative_2', 0):.1f}s")
        
        # Show output summary
        print(f"\n  📋 Solver outputs:")
        for name in ["perceiver_1", "perceiver_2", "phased_1", "phased_2", "iterative_1", "iterative_2"]:
            data = outputs.get(name, {})
            output = data.get("output")
            if output:
                if isinstance(output, list) and output:
                    if isinstance(output[0], list) and output[0]:
                        if isinstance(output[0][0], list):
                            # Multi-test output
                            shape = f"{len(output)} tests"
                        else:
                            shape = f"{len(output)}×{len(output[0])}"
                    else:
                        shape = f"len={len(output)}"
                else:
                    shape = "?"
                print(f"    {name}: ✓ {shape}")
            else:
                print(f"    {name}: ✗ (no output)")
    
    # =========================================================================
    # PHASE 2: Filter Distinct Outputs + Conditional Judge
    # =========================================================================
    
    if verbose:
        print(f"\n🔍 PHASE 2: Filtering distinct outputs and counting votes...")
    
    phase2_start = time.time()
    
    # Get distinct outputs with vote counts
    distinct_outputs, vote_counts, voters = get_distinct_outputs_with_votes(outputs)
    num_distinct = len(distinct_outputs)
    total_with_output = len([k for k,v in outputs.items() if v.get('output') is not None])
    
    if verbose:
        print(f"  • {num_distinct} distinct outputs from {total_with_output}/6 total")
        # Sort by vote count for display
        sorted_by_votes = sorted(vote_counts.items(), key=lambda x: -x[1])
        for solver_name, votes in sorted_by_votes:
            voter_list = ", ".join(voters[solver_name])
            print(f"    ✓ {solver_name}: {votes} votes [{voter_list}]")
    
    # =========================================================================
    # 2-TIER SELECTION SYSTEM
    # Tier 1: Voting - If one output has clear majority (≥4 votes), use it
    # Tier 2: Judge - Only when votes are close/ambiguous
    # =========================================================================
    
    judge_result = None
    judge_used = False
    
    # Sort candidates by vote count (descending)
    sorted_candidates = sorted(vote_counts.items(), key=lambda x: -x[1])
    
    if num_distinct == 0:
        # No outputs at all
        if verbose:
            print(f"  ⚠️ No valid outputs from any solver!")
        top_2 = []
        
    elif num_distinct == 1:
        # Only 1 distinct output - use it for both attempts
        if verbose:
            print(f"  ⚡ Tier 1: Only 1 distinct output - using it for both attempts")
        solver_name = list(distinct_outputs.keys())[0]
        top_2 = [solver_name, solver_name]
        
    elif num_distinct == 2:
        # Exactly 2 distinct outputs - sort by votes
        if verbose:
            print(f"  ⚡ Tier 1: 2 distinct outputs - ordering by vote count")
        top_2 = [c[0] for c in sorted_candidates]
        
    else:
        # More than 2 distinct outputs - use 2-tier voting + judge system
        top_votes = sorted_candidates[0][1]
        second_votes = sorted_candidates[1][1] if len(sorted_candidates) > 1 else 0
        third_votes = sorted_candidates[2][1] if len(sorted_candidates) > 2 else 0
        
        # =====================================================================
        # TIER 1: VOTING - Check if votes give us clear winners
        # =====================================================================
        
        # Case A: Top candidate has clear lead over ALL others
        if top_votes > second_votes:
            # Top candidate wins attempt_1 by votes
            attempt_1_winner = sorted_candidates[0][0]
            
            if verbose:
                print(f"  🗳️ Tier 1: Top candidate wins by votes ({top_votes} vs {second_votes})")
            
            # For attempt_2: check if 2nd place also has clear lead over 3rd
            if second_votes > third_votes:
                # 2nd place wins attempt_2 by votes too - no judge needed!
                if verbose:
                    print(f"  🗳️ Tier 1: Second candidate also wins by votes ({second_votes} vs {third_votes})")
                top_2 = [attempt_1_winner, sorted_candidates[1][0]]
            else:
                # Tie for 2nd place - use judge to pick attempt_2 from tied candidates
                tied_for_second = [c[0] for c in sorted_candidates[1:] if c[1] == second_votes]
                
                if verbose:
                    print(f"  🧑‍⚖️ Tier 2: Tie for 2nd place ({len(tied_for_second)} candidates with {second_votes} votes) - using judge...")
                
                remaining_candidates = {k: v for k, v in distinct_outputs.items() if k in tied_for_second}
                hypothesis_summary = build_hypothesis_summary_for_judge(outputs)
                judge_result = run_judge(task_data, hypothesis_summary, remaining_candidates, verbose=verbose)
                judge_used = True
                
                judge_top = judge_result.get("top_2", tied_for_second[:1])
                top_2 = [attempt_1_winner, judge_top[0] if judge_top else sorted_candidates[1][0]]
        
        # Case B: Tie for 1st place - need judge to break tie
        else:
            tied_for_first = [c[0] for c in sorted_candidates if c[1] == top_votes]
            
            if verbose:
                print(f"  🧑‍⚖️ Tier 2: Tie for 1st place ({len(tied_for_first)} candidates with {top_votes} votes each) - using judge...")
            
            # Only send tied candidates to judge
            tied_candidates = {k: v for k, v in distinct_outputs.items() if k in tied_for_first}
            hypothesis_summary = build_hypothesis_summary_for_judge(outputs)
            judge_result = run_judge(task_data, hypothesis_summary, tied_candidates, verbose=verbose)
            judge_used = True
            
            judge_top_2 = judge_result.get("top_2", tied_for_first[:2])
            
            # If judge only picked 1 (or tied candidates < 2), fill from remaining by votes
            if len(judge_top_2) >= 2:
                top_2 = judge_top_2[:2]
            elif len(judge_top_2) == 1:
                # Need to pick attempt_2 from non-tied candidates by votes
                non_tied = [c[0] for c in sorted_candidates if c[0] not in tied_for_first]
                top_2 = [judge_top_2[0], non_tied[0] if non_tied else sorted_candidates[1][0]]
            else:
                top_2 = tied_for_first[:2]
        
        if verbose and top_2:
            print(f"  ✅ Final picks: attempt_1={top_2[0]} ({vote_counts.get(top_2[0], 0)} votes), attempt_2={top_2[1]} ({vote_counts.get(top_2[1], 0)} votes)")
    
    phase2_time = time.time() - phase2_start
    
    if verbose:
        print(f"  ⏱️ Phase 2 completed in {phase2_time:.1f}s")
    
    # =========================================================================
    # PHASE 3: Extract Top 2 Submissions
    # =========================================================================
    
    attempt_1 = None
    attempt_2 = None
    attempt_1_source = None
    attempt_2_source = None
    
    # Get top 2 outputs
    for i, solver_name in enumerate(top_2):
        if solver_name in outputs:
            output = outputs[solver_name].get("output")
            if output is not None:
                if i == 0:
                    attempt_1 = output
                    attempt_1_source = solver_name
                elif i == 1:
                    attempt_2 = output
                    attempt_2_source = solver_name
    
    # If either is missing, try to get from remaining outputs
    if attempt_1 is None or attempt_2 is None:
        for solver_name, data in outputs.items():
            output = data.get("output")
            if output is not None:
                if attempt_1 is None:
                    attempt_1 = output
                    attempt_1_source = solver_name
                elif attempt_2 is None and solver_name != attempt_1_source:
                    attempt_2 = output
                    attempt_2_source = solver_name
    
    # =========================================================================
    # PHASE 4: Score if ground truths provided
    # =========================================================================
    
    scores = {}
    if ground_truths:
        n_tests = len(ground_truths)
        
        for attempt_num, (attempt, source) in enumerate([(attempt_1, attempt_1_source), (attempt_2, attempt_2_source)], 1):
            if attempt is None:
                scores[f"attempt_{attempt_num}"] = {"correct": False, "source": source, "tests_correct": 0, "total_tests": n_tests}
                continue
            
            # Handle multi-test outputs: attempt can be a single grid or list of grids
            if n_tests == 1:
                # Single test case
                attempt_arr = np.array(attempt)
                gt_arr = np.array(ground_truths[0])
                if attempt_arr.shape == gt_arr.shape and np.array_equal(attempt_arr, gt_arr):
                    scores[f"attempt_{attempt_num}"] = {"correct": True, "source": source, "tests_correct": 1, "total_tests": 1}
                else:
                    scores[f"attempt_{attempt_num}"] = {"correct": False, "source": source, "tests_correct": 0, "total_tests": 1}
            else:
                # Multiple test cases - attempt should be a list of grids
                # Check if attempt is a list of grids (not just a single grid)
                is_list_of_grids = (
                    isinstance(attempt, list) and 
                    len(attempt) > 0 and 
                    isinstance(attempt[0], list) and 
                    len(attempt[0]) > 0 and
                    isinstance(attempt[0][0], list)
                )
                
                if not is_list_of_grids:
                    # Single grid provided for multi-test task - only compare with first GT
                    attempt_arr = np.array(attempt)
                    gt_arr = np.array(ground_truths[0])
                    tests_correct = 1 if (attempt_arr.shape == gt_arr.shape and np.array_equal(attempt_arr, gt_arr)) else 0
                    scores[f"attempt_{attempt_num}"] = {
                        "correct": tests_correct == n_tests,  # Only fully correct if all tests pass
                        "source": source,
                        "tests_correct": tests_correct,
                        "total_tests": n_tests,
                        "partial": tests_correct > 0 and tests_correct < n_tests
                    }
                else:
                    # List of grids - compare each
                    tests_correct = 0
                    for i, gt in enumerate(ground_truths):
                        if i < len(attempt):
                            attempt_arr = np.array(attempt[i])
                            gt_arr = np.array(gt)
                            if attempt_arr.shape == gt_arr.shape and np.array_equal(attempt_arr, gt_arr):
                                tests_correct += 1
                    
                    scores[f"attempt_{attempt_num}"] = {
                        "correct": tests_correct == n_tests,
                        "source": source,
                        "tests_correct": tests_correct,
                        "total_tests": n_tests,
                        "partial": tests_correct > 0 and tests_correct < n_tests
                    }
    
    # =========================================================================
    # Build Result
    # =========================================================================
    
    total_time = time.time() - start_time
    
    # Aggregate usage from all solver outputs + judge
    judge_usage = judge_result.get("usage") if judge_result else None
    usage = aggregate_usage(outputs, judge_usage)
    
    # Update global tracker for backwards compatibility with get_usage_stats()
    # Transform aggregated keys to match UsageStats.add() expected format
    usage_for_tracker = {
        "prompt_tokens": usage.get("total_prompt_tokens", 0),
        "completion_tokens": usage.get("total_completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "cost": usage.get("total_cost", 0.0),
        "is_byok": usage.get("is_byok", False),
        "completion_tokens_details": {
            "reasoning_tokens": usage.get("total_reasoning_tokens", 0),
        },
        "prompt_tokens_details": {
            "cached_tokens": usage.get("total_cached_tokens", 0),
        },
        "cost_details": {
            "upstream_inference_cost": usage.get("upstream_cost", 0.0),
            "upstream_inference_prompt_cost": usage.get("upstream_prompt_cost", 0.0),
            "upstream_inference_completions_cost": usage.get("upstream_completion_cost", 0.0),
        },
    }
    record_usage(usage_for_tracker)
    
    result = {
        "attempt_1": attempt_1,
        "attempt_2": attempt_2,
        "attempt_1_source": attempt_1_source,
        "attempt_2_source": attempt_2_source,
        "all_outputs": outputs,
        "distinct_outputs": list(distinct_outputs.keys()),
        "vote_counts": vote_counts,
        "voters": voters,
        "num_distinct": num_distinct,
        "total_candidates": 6,
        "judge_used": judge_used,
        "ratings": judge_result.get("ratings", []) if judge_result else [],
        "top_2": top_2,
        "scores": scores,
        "timing": {
            "phase_1": phase1_time,
            "phase_2": phase2_time,
            "total": total_time,
            "solver_times": solver_times,
        },
        "usage": usage,
    }
    
    # Minimal completion print (always)
    if not verbose:
        print(f" | {num_distinct}d {'J' if judge_used else ''} {total_time:.0f}s")
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"RESULT SUMMARY")
        print(f"{'='*60}")
        print(f"  Total candidates: 6 (3 solvers × 2 runs)")
        print(f"  Distinct outputs: {num_distinct}/6")
        print(f"  Judge used: {'Yes' if judge_used else 'No (skipped)'}")
        print(f"  Attempt 1: {attempt_1_source} {'✓' if attempt_1 else '✗'}")
        print(f"  Attempt 2: {attempt_2_source} {'✓' if attempt_2 else '✗'}")
        
        if scores:
            print(f"\n  Scores:")
            for k, v in scores.items():
                tests_correct = v.get('tests_correct', 0)
                total_tests = v.get('total_tests', 1)
                if v.get('correct'):
                    status = f"✓ CORRECT ({tests_correct}/{total_tests})"
                elif v.get('partial'):
                    status = f"⚠️ PARTIAL ({tests_correct}/{total_tests})"
                else:
                    status = f"✗ wrong ({tests_correct}/{total_tests})"
                print(f"    {k}: {status} ({v.get('source')})")
        
        print(f"\n  Total time: {total_time:.1f}s")
        print(f"  Total cost: ${usage.get('effective_cost', 0):.4f}")
        print(f"{'='*60}\n")
    
    return result


def solve_task_with_judge_sync(
    task_data: dict,
    task_id: str = "unknown",
    ground_truths: list[np.ndarray] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Synchronous wrapper for solve_with_judge."""
    return solve_with_judge(task_data, task_id, ground_truths, verbose)


# =============================================================================
# Batch Processing
# =============================================================================

def solve_batch_with_judge(
    tasks: list[tuple[str, dict]],
    ground_truths_map: dict[str, list[np.ndarray]] | None = None,
    max_concurrent: int = 10,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """
    Solve multiple tasks with the judge system.
    
    Args:
        tasks: List of (task_id, task_data) tuples
        ground_truths_map: Optional dict mapping task_id to ground truths
        max_concurrent: Max tasks to run in parallel
        verbose: Whether to print progress
        
    Returns:
        List of results
    """
    results = []
    
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {}
        
        for task_id, task_data in tasks:
            gt = ground_truths_map.get(task_id) if ground_truths_map else None
            future = executor.submit(
                solve_with_judge,
                task_data,
                task_id,
                gt,
                verbose,
            )
            futures[future] = task_id
        
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                result = future.result()
                result["task_id"] = task_id
                results.append(result)
            except Exception as e:
                if verbose:
                    print(f"Task {task_id} failed: {e}")
                results.append({
                    "task_id": task_id,
                    "error": str(e),
                    "attempt_1": None,
                    "attempt_2": None,
                })
    
    return results


if __name__ == "__main__":
    # Test with a simple task
    test_task = {
        "train": [
            {"input": [[0, 0, 0], [0, 1, 0], [0, 0, 0]], "output": [[0, 0, 0], [0, 0, 1], [0, 0, 0]]},
            {"input": [[0, 0, 0], [1, 0, 0], [0, 0, 0]], "output": [[0, 0, 0], [0, 1, 0], [0, 0, 0]]},
        ],
        "test": [
            {"input": [[0, 0, 0], [0, 0, 1], [0, 0, 0]]}
        ]
    }
    
    result = solve_with_judge(test_task, task_id="test_001")
    
    print("\n" + "="*60)
    print("FINAL RESULT")
    print("="*60)
    print(f"Attempt 1 ({result['attempt_1_source']}): {result['attempt_1']}")
    print(f"Attempt 2 ({result['attempt_2_source']}): {result['attempt_2']}")
