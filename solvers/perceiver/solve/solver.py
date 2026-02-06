"""Core solver - exhaustive solution collection with smart fallback."""

import time
from typing import Any
from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np

from solvers.perceiver.config import SOLVER_MODELS, MODEL_RANK, MIN_CONFIDENCE_SCORE
from solvers.perceiver.llms.client import call_llm
from solvers.perceiver.models import SolutionCandidate
from solvers.perceiver.solve.executor import execute_transform, parse_llm_response, test_on_examples
from solvers.perceiver.solve.prompt import SOLVER_SYSTEM, generate_prompt, generate_feedback_prompt
from solvers.perceiver.verification.self_verifier import self_verify
from solvers.perceiver.utils.trace import TRACE_LOGGER

# Fixed self-verification model (always claude-opus-4.6 with adaptive reasoning)
SELF_VERIFY_MODEL = "anthropic/claude-opus-4.6"
SELF_VERIFY_EXTRA_BODY = {"reasoning": {"enabled": True}}


# =============================================================================
# Deduplication Helpers
# =============================================================================

def outputs_are_equal(results1: list[np.ndarray], results2: list[np.ndarray]) -> bool:
    """Check if two solutions produce identical output arrays for all test inputs."""
    if len(results1) != len(results2):
        return False
    for arr1, arr2 in zip(results1, results2):
        if arr1 is None or arr2 is None:
            if arr1 is not arr2:
                return False
            continue
        if arr1.shape != arr2.shape:
            return False
        if not np.array_equal(arr1, arr2):
            return False
    return True


def get_distinct_solutions(
    candidates: list[SolutionCandidate], 
    target: int
) -> list[SolutionCandidate]:
    """
    Select up to `target` distinct solutions (by output arrays), 
    prioritized by verifier score.
    
    Returns candidates sorted by score, ensuring no two have identical outputs.
    """
    # Sort by score first (highest to lowest)
    sorted_candidates = sorted(candidates, key=lambda c: -c.verifier_score)
    
    distinct: list[SolutionCandidate] = []
    for cand in sorted_candidates:
        # Check if this candidate's outputs are distinct from all selected ones
        is_duplicate = False
        for selected in distinct:
            if outputs_are_equal(cand.test_results, selected.test_results):
                is_duplicate = True
                break
        
        if not is_duplicate:
            distinct.append(cand)
            if len(distinct) >= target:
                break
    
    return distinct


def count_distinct_high_confidence(
    candidates: list[SolutionCandidate],
    min_score: int
) -> int:
    """Count how many distinct high-confidence solutions we have."""
    high_conf = [c for c in candidates if c.verifier_score >= min_score]
    distinct = get_distinct_solutions(high_conf, len(high_conf))
    return len(distinct)


# =============================================================================
# Alternative Approach Feedback
# =============================================================================

def generate_alternative_feedback(
    original_prompt: str,
    code: str,
    explanation: str,
    attempt_num: int,
    previous_approaches: list[str],
) -> str:
    """
    Generate feedback encouraging a fundamentally different approach.
    Used when the current solution passes but we want more alternatives.
    """
    prev_summary = ""
    if previous_approaches:
        prev_summary = "\n".join(f"  {i+1}. {approach[:100]}..." 
                                  for i, approach in enumerate(previous_approaches[-3:]))
    
    feedback_section = f"""

============================================================
✅ SOLUTION {attempt_num} PASSED - BUT TRY A DIFFERENT APPROACH
============================================================

Your solution works! However, we want to explore alternative approaches.

Your current approach:
```python
{code[:500]}...
```

{f"Previous approaches you've tried:{chr(10)}{prev_summary}" if prev_summary else ""}

NOW: Try a FUNDAMENTALLY DIFFERENT approach to solve this puzzle.
- Don't just tweak your current solution
- Think of a completely different way to interpret the pattern
- Consider alternative transformation strategies from the pattern taxonomy
- What if you approached this from the opposite direction?

Provide a NEW solution with a different methodology.
"""
    return original_prompt + feedback_section


# =============================================================================
# Single Model Solver (with Early Stop option)
# =============================================================================

def solve_single_exhaustive(
    task_data: dict[str, Any],
    model_config: dict[str, Any],
    perceptions: list[dict[str, Any]] | None = None,
    deltas: list[dict[str, Any]] | None = None,
    test_perception: dict[str, Any] | None = None,
    hypotheses: list[dict[str, Any]] | None = None,
    observations: dict[str, Any] | None = None,
    key_insight: str | None = None,
    max_tries: int | None = None,
    verbose: bool = True,
    early_stop: bool = False,
    skip_self_verify: bool = False,
) -> list[SolutionCandidate]:
    """
    Solve a task with optional early stopping.
    
    Flow (exhaustive mode - early_stop=False):
    1. Run all tries, collecting solutions that pass training
    2. At the end, self-verify ALL solutions in parallel
    3. Return all candidates with their scores
    
    Flow (early stop mode - early_stop=True):
    1. Run until ONE solution passes training, then stop
    2. Optionally self-verify just that one solution
    3. Return immediately

    Args:
        task_data: Task with 'train' and 'test' keys
        model_config: Model configuration dict
        perceptions: Pre-computed perceptions
        deltas: Pre-computed deltas
        test_perception: Perception of test input
        hypotheses: Pre-computed transformation hypotheses
        observations: Task-level observations
        key_insight: Key insight about the puzzle
        max_tries: Override default tries from config
        verbose: Whether to print progress
        early_stop: If True, stop after first passing solution
        skip_self_verify: If True, skip self-verification entirely

    Returns:
        List of SolutionCandidates that passed training (with self-verify scores if not skipped)
    """
    model_id = model_config["id"]
    model = model_config["model"]
    extra_body = model_config.get("extra_body")
    max_tokens = model_config.get("max_tokens")
    tries = max_tries or model_config.get("tries", 5)

    train_examples = task_data['train']
    test_inputs = [np.array(t['input']) for t in task_data['test']]

    # Generate initial prompt
    prompt = generate_prompt(
        task_data=task_data,
        perceptions=perceptions,
        deltas=deltas,
        test_perception=test_perception,
        hypotheses=hypotheses,
        observations=observations,
        key_insight=key_insight,
    )

    mode_str = "early-stop" if early_stop else f"{tries} tries"
    if verbose:
        print(f"     🚀 [{model_id}] Starting ({mode_str})...")

    # Collect solutions that pass training (don't verify yet)
    passing_solutions: list[dict[str, Any]] = []
    previous_approaches: list[str] = []

    # =========================================================================
    # PHASE 1: Generate solutions
    # =========================================================================
    for attempt in range(1, tries + 1):
        try:
            # Call the model
            response, elapsed = call_llm(
                model=model,
                system_prompt=SOLVER_SYSTEM,
                user_prompt=prompt,
                extra_body=extra_body,
                max_tokens=max_tokens,
            )

            TRACE_LOGGER.log(f"solver_{model_id}", model, prompt[:500], response[:500], elapsed)

            # Parse response
            parsed = parse_llm_response(response)
            if not parsed['code']:
                if verbose:
                    print(f"     [{model_id}] Try {attempt}/{tries}: ✗ No code")
                prompt = generate_feedback_prompt(prompt, "", ["No valid code found"], attempt)
                continue

            # Test on training examples
            all_passed, feedback_messages = test_on_examples(parsed['code'], train_examples)

            if not all_passed:
                if verbose:
                    print(f"     [{model_id}] Try {attempt}/{tries}: ✗ Failed training")
                prompt = generate_feedback_prompt(prompt, parsed['code'], feedback_messages, attempt)
                continue

            # Passed training! Execute on test inputs
            if verbose:
                print(f"     [{model_id}] Try {attempt}/{tries}: ✓ Passed training!")

            test_results = [execute_transform(parsed['code'], ti) for ti in test_inputs]
            
            passing_solutions.append({
                'code': parsed['code'],
                'explanation': parsed['explanation'] or '',
                'test_results': test_results,
                'attempt': attempt,
            })

            # EARLY STOP: If enabled, break after first passing solution
            if early_stop:
                if verbose:
                    print(f"     [{model_id}] ⚡ Early stop - got first passing solution")
                break

            # Track approach and continue with alternative feedback
            previous_approaches.append(parsed['explanation'] or f"Approach {attempt}")
            
            if attempt < tries:
                prompt = generate_alternative_feedback(
                    prompt, parsed['code'], parsed['explanation'] or '',
                    attempt, previous_approaches
                )

        except Exception as e:
            if verbose:
                print(f"     [{model_id}] Try {attempt}/{tries}: ✗ Error - {str(e)[:50]}")

    if verbose:
        print(f"     [{model_id}] 📋 {len(passing_solutions)} solution(s) passed training")

    if not passing_solutions:
        return []

    # =========================================================================
    # PHASE 2: Self-Verification (optional)
    # =========================================================================
    if skip_self_verify:
        # Skip self-verification - return with default score
        if verbose:
            print(f"     [{model_id}] ⏭️ Skipping self-verification")
        
        return [
            SolutionCandidate(
                code=sol['code'],
                explanation=sol['explanation'],
                model_id=model_id,
                verifier_score=75,  # Default passing score
                verifier_verdict="SKIPPED",
                self_verify_decision="SKIPPED",
                attempts=sol['attempt'],
                test_results=sol['test_results'],
            )
            for sol in passing_solutions
        ]
    
    if verbose:
        print(f"     [{model_id}] 🔍 Self-verifying {len(passing_solutions)} solution(s)...")

    def verify_one(sol: dict[str, Any]) -> SolutionCandidate:
        """Self-verify a single solution and return a SolutionCandidate."""
        sv_result = self_verify(
            model=SELF_VERIFY_MODEL,
            model_id="claude-opus-4.6",
            extra_body=SELF_VERIFY_EXTRA_BODY,
            max_tokens=None,
            code=sol['code'],
            explanation=sol['explanation'],
            train_examples=train_examples,
            test_inputs=test_inputs,
            test_outputs=sol['test_results'],
            hypotheses=hypotheses,
            key_insight=key_insight,
            observations=observations,
            perceptions=perceptions,
            verbose=False,
        )
        
        sv_decision = sv_result.get('decision', 'UNSURE')
        sv_score = sv_result.get('score', 50)
        
        return SolutionCandidate(
            code=sol['code'],
            explanation=sol['explanation'],
            model_id=model_id,
            verifier_score=sv_score,
            verifier_verdict=sv_decision,
            self_verify_decision=sv_decision,
            attempts=sol['attempt'],
            test_results=sol['test_results'],
        )

    # Run verifications in parallel with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(len(passing_solutions), 10)) as executor:
        all_candidates = list(executor.map(verify_one, passing_solutions))

    if verbose:
        high_conf = [c for c in all_candidates if c.verifier_score >= MIN_CONFIDENCE_SCORE]
        print(f"     [{model_id}] 📊 Finished: {len(all_candidates)} verified "
              f"({len(high_conf)} with score {MIN_CONFIDENCE_SCORE}+)")

    return all_candidates


# =============================================================================
# Multi-Model Solver (Exhaustive Primary + Smart Fallback)
# =============================================================================

def solve_with_models(
    task_data: dict[str, Any],
    perceptions: list[dict[str, Any]] | None = None,
    deltas: list[dict[str, Any]] | None = None,
    test_perception: dict[str, Any] | None = None,
    hypotheses: list[dict[str, Any]] | None = None,
    observations: dict[str, Any] | None = None,
    key_insight: str | None = None,
    models: list[dict[str, Any]] | None = None,
    primary_model_id: str = "claude-opus-4.6",
    target_solutions: int = 2,
    verbose: bool = True,
) -> list[SolutionCandidate]:
    """
    Solve a task with exhaustive primary model + smart fallback.

    Strategy:
    1. Run primary model (claude-opus-4.6) exhaustively - collect ALL solutions
    2. Self-verify all solutions, filter for score >= 90
    3. If we have 2+ high-confidence solutions, pick top 2 by score
    4. If < 2, run fallback models until we have 2 total (max 10 tries each)

    Args:
        task_data: Task with 'train' and 'test' keys
        perceptions: Pre-computed perceptions
        deltas: Pre-computed deltas
        test_perception: Perception of test input
        hypotheses: Pre-computed transformation hypotheses
        observations: Task-level observations
        key_insight: Key insight about the puzzle
        models: List of model configs (defaults to SOLVER_MODELS)
        primary_model_id: ID of the primary model (default: claude-opus-4.6)
        target_solutions: Number of high-confidence solutions needed (default: 2)
        verbose: Whether to print progress

    Returns:
        List of top SolutionCandidates (up to target_solutions, sorted by score)
    """
    if models is None:
        models = SOLVER_MODELS

    # Separate primary model from fallback models
    primary_model = None
    fallback_models = []
    for cfg in models:
        if cfg["id"] == primary_model_id:
            primary_model = cfg
        else:
            fallback_models.append(cfg)

    if primary_model is None:
        primary_model = models[0]
        fallback_models = models[1:]

    # =========================================================================
    # PHASE 1: Exhaustive Primary Model
    # =========================================================================
    if verbose:
        print(f"  🎯 Phase 1: Running [{primary_model['id']}] exhaustively...")

    all_candidates = solve_single_exhaustive(
        task_data=task_data,
        model_config=primary_model,
        perceptions=perceptions,
        deltas=deltas,
        test_perception=test_perception,
        hypotheses=hypotheses,
        observations=observations,
        key_insight=key_insight,
        verbose=verbose,
    )

    # =========================================================================
    # PHASE 2: Filter for High-Confidence DISTINCT Solutions
    # =========================================================================
    high_confidence = [c for c in all_candidates if c.verifier_score >= MIN_CONFIDENCE_SCORE]
    
    # Get distinct solutions (by output arrays)
    distinct_high_confidence = get_distinct_solutions(high_confidence, target_solutions)
    
    if verbose:
        print(f"\n  📊 Phase 1 Results: {len(all_candidates)} solutions, "
              f"{len(high_confidence)} high-confidence ({MIN_CONFIDENCE_SCORE}+), "
              f"{len(distinct_high_confidence)} distinct")

    # Check if we have enough DISTINCT solutions
    if len(distinct_high_confidence) >= target_solutions:
        if verbose:
            print(f"  ✅ Got {len(distinct_high_confidence)} distinct high-confidence solutions from primary model!")
            for i, c in enumerate(distinct_high_confidence[:target_solutions]):
                print(f"     {i+1}. score={c.verifier_score}, attempts={c.attempts}")
        return distinct_high_confidence[:target_solutions]

    # =========================================================================
    # PHASE 3: Run Fallback Models (Need More DISTINCT Solutions)
    # =========================================================================
    if not fallback_models:
        if verbose:
            print(f"  ⚠️ Only {len(distinct_high_confidence)} distinct high-confidence solution(s), no fallbacks configured")
        # Return what we have (even if less than target), ensuring distinct outputs
        all_distinct = get_distinct_solutions(all_candidates, target_solutions)
        return all_distinct

    needed = target_solutions - len(distinct_high_confidence)
    if verbose:
        print(f"\n  🔄 Phase 2: Need {needed} more distinct high-confidence solution(s), "
              f"running {len(fallback_models)} fallback models...")

    # Shared state for fallback coordination
    collected_from_fallbacks: list[SolutionCandidate] = []
    stop_event = threading.Event()
    lock = threading.Lock()
    MAX_FALLBACK_TRIES = 10

    def run_fallback_model(model_config: dict[str, Any]) -> None:
        """Run a fallback model, stop when we have enough high-confidence solutions."""
        model_id = model_config["id"]
        model = model_config["model"]
        extra_body = model_config.get("extra_body")
        max_tokens = model_config.get("max_tokens")
        
        train_examples = task_data['train']
        test_inputs = [np.array(t['input']) for t in task_data['test']]

        # Generate prompt
        prompt = generate_prompt(
            task_data=task_data,
            perceptions=perceptions,
            deltas=deltas,
            test_perception=test_perception,
            hypotheses=hypotheses,
            observations=observations,
            key_insight=key_insight,
        )

        if verbose:
            print(f"     🚀 [{model_id}] Starting (up to {MAX_FALLBACK_TRIES} tries, "
                  f"stopping when we have {target_solutions} total high-confidence)...")

        for attempt in range(1, MAX_FALLBACK_TRIES + 1):
            # Check if we should stop
            if stop_event.is_set():
                if verbose:
                    print(f"     [{model_id}] Stopping - enough solutions collected")
                return

            try:
                response, elapsed = call_llm(
                    model=model,
                    system_prompt=SOLVER_SYSTEM,
                    user_prompt=prompt,
                    extra_body=extra_body,
                    max_tokens=max_tokens,
                )

                parsed = parse_llm_response(response)
                if not parsed['code']:
                    prompt = generate_feedback_prompt(prompt, "", ["No valid code found"], attempt)
                    continue

                all_passed, feedback_messages = test_on_examples(parsed['code'], train_examples)
                if not all_passed:
                    prompt = generate_feedback_prompt(prompt, parsed['code'], feedback_messages, attempt)
                    continue

                # Passed training - execute and self-verify
                test_results = [execute_transform(parsed['code'], ti) for ti in test_inputs]

                sv_result = self_verify(
                    model=SELF_VERIFY_MODEL,
                    model_id="claude-opus-4.6",
                    extra_body=SELF_VERIFY_EXTRA_BODY,
                    max_tokens=None,
                    code=parsed['code'],
                    explanation=parsed['explanation'] or '',
                    train_examples=train_examples,
                    test_inputs=test_inputs,
                    test_outputs=test_results,
                    hypotheses=hypotheses,
                    key_insight=key_insight,
                    observations=observations,
                    perceptions=perceptions,
                    verbose=False,
                )

                sv_score = sv_result.get('score', 50)
                sv_decision = sv_result.get('decision', 'UNSURE')

                candidate = SolutionCandidate(
                    code=parsed['code'],
                    explanation=parsed['explanation'] or '',
                    model_id=model_id,
                    verifier_score=sv_score,
                    verifier_verdict=sv_decision,
                    self_verify_decision=sv_decision,
                    attempts=attempt,
                    test_results=test_results,
                )

                is_high_confidence = sv_score >= MIN_CONFIDENCE_SCORE

                with lock:
                    collected_from_fallbacks.append(candidate)
                    
                    if is_high_confidence:
                        # Count total DISTINCT high-confidence (primary + fallbacks)
                        all_high_conf = distinct_high_confidence + [
                            c for c in collected_from_fallbacks 
                            if c.verifier_score >= MIN_CONFIDENCE_SCORE
                        ]
                        total_distinct = len(get_distinct_solutions(all_high_conf, target_solutions))
                        
                        confidence_tag = "🔥"
                        if verbose:
                            print(f"     [{model_id}] {confidence_tag} High-confidence solution! "
                                  f"(score={sv_score}, total distinct high-conf={total_distinct})")
                        
                        if total_distinct >= target_solutions:
                            if verbose:
                                print(f"  🎯 Got {target_solutions} distinct high-confidence solutions!")
                            stop_event.set()
                            return
                    else:
                        if verbose:
                            print(f"     [{model_id}] ✅ Solution (score={sv_score}, below threshold)")

                # Continue trying for alternatives
                prompt = generate_alternative_feedback(
                    prompt, parsed['code'], parsed['explanation'] or '',
                    attempt, []
                )

            except Exception as e:
                if verbose:
                    print(f"     [{model_id}] Attempt {attempt}: Error - {str(e)[:50]}")

        if verbose:
            print(f"     [{model_id}] Exhausted {MAX_FALLBACK_TRIES} tries")

    # Launch fallback models in parallel with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(fallback_models)) as executor:
        list(executor.map(run_fallback_model, fallback_models))

    # =========================================================================
    # FINAL: Combine and Return Top DISTINCT Solutions
    # =========================================================================
    all_high_conf = high_confidence + [
        c for c in collected_from_fallbacks 
        if c.verifier_score >= MIN_CONFIDENCE_SCORE
    ]
    
    # Get distinct high-confidence solutions first
    final_selection = get_distinct_solutions(all_high_conf, target_solutions)
    
    # If we still don't have enough distinct solutions, add from lower-confidence
    if len(final_selection) < target_solutions:
        remaining = all_candidates + collected_from_fallbacks
        remaining = [c for c in remaining if c not in final_selection]
        # Get distinct from remaining
        remaining_distinct = get_distinct_solutions(remaining, target_solutions - len(final_selection))
        # Filter to only add those distinct from what we already have
        for cand in remaining_distinct:
            is_dup = any(outputs_are_equal(cand.test_results, sel.test_results) for sel in final_selection)
            if not is_dup:
                final_selection.append(cand)
                if len(final_selection) >= target_solutions:
                    break

    if verbose:
        print(f"\n  📊 Final Results ({len(final_selection)} distinct solutions):")
        for i, c in enumerate(final_selection):
            conf_tag = "🔥" if c.verifier_score >= MIN_CONFIDENCE_SCORE else "✅"
            print(f"     {i+1}. {conf_tag} [{c.model_id}] score={c.verifier_score}")

    return final_selection


# =============================================================================
# Phoenix Mode: Early Stop Per Model (Each model stops at first passing solution)
# =============================================================================

def solve_with_early_stop(
    task_data: dict[str, Any],
    perceptions: list[dict[str, Any]] | None = None,
    deltas: list[dict[str, Any]] | None = None,
    test_perception: dict[str, Any] | None = None,
    hypotheses: list[dict[str, Any]] | None = None,
    observations: dict[str, Any] | None = None,
    key_insight: str | None = None,
    models: list[dict[str, Any]] | None = None,
    skip_self_verify: bool = True,
    verbose: bool = True,
    stop_event: "threading.Event" = None,  # Signal to stop early from external source
) -> list[SolutionCandidate]:
    """
    Solve a task with multiple models in parallel.
    Each model stops as soon as IT finds a passing solution (early stop per model).
    Returns one solution from each model that found one.
    
    Then uses voting to pick the best solution.

    Args:
        task_data: Task with 'train' and 'test' keys
        perceptions: Pre-computed perceptions
        deltas: Pre-computed deltas  
        test_perception: Perception of test input
        hypotheses: Pre-computed transformation hypotheses
        observations: Task-level observations
        key_insight: Key insight about the puzzle
        models: List of model configs (defaults to SOLVER_MODELS)
        skip_self_verify: If True, skip self-verification (faster)
        verbose: Whether to print progress
        stop_event: Optional threading.Event to signal early stop from external source

    Returns:
        List of SolutionCandidates (one per model that found a solution), best first
    """
    # Check for early stop signal at start
    if stop_event and stop_event.is_set():
        if verbose:
            print("🛑 Early stop signal received - another solver found solution")
        return []
    
    if models is None:
        models = SOLVER_MODELS

    if verbose:
        print(f"  ⚡ Running {len(models)} models in parallel (early-stop per model)...")

    # Collect one solution from each model
    all_candidates: list[SolutionCandidate] = []
    lock = threading.Lock()

    def run_model(model_config: dict[str, Any]) -> None:
        """Run a single model until it finds ONE passing solution."""
        model_id = model_config["id"]
        
        try:
            candidates = solve_single_exhaustive(
                task_data=task_data,
                model_config=model_config,
                perceptions=perceptions,
                deltas=deltas,
                test_perception=test_perception,
                hypotheses=hypotheses,
                observations=observations,
                key_insight=key_insight,
                verbose=verbose,
                early_stop=True,  # Stop at first passing solution for THIS model
                skip_self_verify=skip_self_verify,
            )

            if candidates:
                with lock:
                    all_candidates.extend(candidates)
                    if verbose:
                        print(f"  ✅ [{model_id}] Found solution (attempt {candidates[0].attempts})")
            else:
                if verbose:
                    print(f"  ⚠️ [{model_id}] No solution found")
        
        except Exception as e:
            if verbose:
                print(f"  ❌ [{model_id}] Error: {str(e)[:50]}")

    # Run all models in parallel
    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        list(executor.map(run_model, models))

    if verbose:
        print(f"  📊 Got {len(all_candidates)} solution(s) from {len(models)} models")

    if not all_candidates:
        return []

    # Vote on the best solution (most common output)
    # Group by output and pick most common
    from collections import Counter
    
    def output_key(candidate: SolutionCandidate) -> str:
        """Create hashable key from outputs."""
        return str([r.tolist() if hasattr(r, 'tolist') else r for r in candidate.test_results])
    
    output_counts = Counter(output_key(c) for c in all_candidates)
    most_common_key = output_counts.most_common(1)[0][0]
    
    # Find candidate with most common output, preferring higher-ranked models
    best_candidate = None
    for c in all_candidates:
        if output_key(c) == most_common_key:
            if best_candidate is None:
                best_candidate = c
            else:
                # Prefer by model rank
                try:
                    curr_rank = MODEL_RANK.index(c.model_id)
                except ValueError:
                    curr_rank = len(MODEL_RANK)
                try:
                    best_rank = MODEL_RANK.index(best_candidate.model_id)
                except ValueError:
                    best_rank = len(MODEL_RANK)
                if curr_rank < best_rank:
                    best_candidate = c

    if verbose:
        vote_count = output_counts[most_common_key]
        print(f"  🗳️ Winner: [{best_candidate.model_id}] with {vote_count}/{len(all_candidates)} votes")

    # Return best candidate first, then others
    result = [best_candidate] + [c for c in all_candidates if c != best_candidate]
    return result


# =============================================================================
# Legacy Wrapper (for backwards compatibility)
# =============================================================================

def solve_single(
    task_data: dict[str, Any],
    model_config: dict[str, Any],
    **kwargs,
) -> tuple[SolutionCandidate | None, SolutionCandidate | None]:
    """
    Legacy wrapper - returns (best_self_verified, best_training_passed).
    Use solve_single_exhaustive for new code.
    """
    candidates = solve_single_exhaustive(task_data, model_config, **kwargs)
    
    if not candidates:
        return (None, None)
    
    # Best by score
    candidates.sort(key=lambda c: -c.verifier_score)
    best = candidates[0]
    
    # Return as (self_verified, training_passed)
    if best.self_verify_decision == "CORRECT":
        return (best, best)
    else:
        return (None, best)
