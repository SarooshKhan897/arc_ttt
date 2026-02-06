"""Self-Verifier - model verifies its own output with full context."""

import json
import re
from typing import Any

import numpy as np

from solvers.perceiver.llms.client import call_llm
from solvers.perceiver.utils.grid import grid_to_text

# =============================================================================
# Self-Verification Prompt (Enhanced with Perception Context)
# =============================================================================

SELF_VERIFICATION_PROMPT = """You wrote code to solve an ARC puzzle. I ran your code on the TEST input(s).

Your job is to verify: Does your output match the transformation pattern from the training examples?

═══════════════════════════════════════════════════════════════════
CONTEXT FROM PERCEPTION ANALYSIS
═══════════════════════════════════════════════════════════════════
{perception_context}

═══════════════════════════════════════════════════════════════════
YOUR SOLUTION
═══════════════════════════════════════════════════════════════════

## Your Understanding of the Pattern
{explanation}

## Your Code
```python
{code}
```

═══════════════════════════════════════════════════════════════════
TRAINING EXAMPLES (Ground Truth)
═══════════════════════════════════════════════════════════════════
{training_pairs}

═══════════════════════════════════════════════════════════════════
YOUR TEST OUTPUT(S) TO VERIFY ({num_tests} test input(s))
═══════════════════════════════════════════════════════════════════
{test_results}

═══════════════════════════════════════════════════════════════════
VERIFICATION CHECKLIST
═══════════════════════════════════════════════════════════════════

1. **HYPOTHESIS MATCH**: Does your output match the key insight/hypothesis?
   - Key insight: {key_insight}
   - Your output should demonstrate this exact transformation

2. **SHAPE VERIFICATION**: 
   - Training output shapes: {training_output_shapes}
   - Your output shapes should follow the same pattern

3. **COLOR VERIFICATION**: 
   - Training output colors: {training_output_colors}
   - Your output colors: {your_output_colors}
   - Any unexpected colors? Any missing expected colors?

4. **STRUCTURAL VERIFICATION**:
   - Do your outputs have the same structure as training outputs?
   - Same symmetry? Same object placement? Same patterns?

5. **COMMON ERROR CHECK**:
   - Off-by-one error (shifted by 1 row/column)?
   - Wrong rotation/reflection direction?
   - Foreground/background swapped?
   - Partial application (rule applied to some but not all)?
   - Wrong anchor point?
   - Edge case at boundaries?

═══════════════════════════════════════════════════════════════════
YOUR RESPONSE (ALL THREE REQUIRED)
═══════════════════════════════════════════════════════════════════

**VERDICT** (must be one of):
- CORRECT: Output matches the pattern. The transformation is applied correctly.
- WRONG: Output does NOT match the expected pattern. [Explain what's wrong]
- UNSURE: Something looks off but I'm not certain. [Explain concern]

**SCORE** (0-100):
- 90-100: Highly confident the outputs are correct
- 70-89: Likely correct but minor concerns
- 50-69: Uncertain, could go either way
- 30-49: Likely has issues
- 0-29: Almost certainly wrong

**FEEDBACK**: If WRONG or UNSURE, explain what the output SHOULD look like.

Format your response as:
VERDICT: [CORRECT/WRONG/UNSURE]
SCORE: [0-100]
FEEDBACK: [Your explanation]"""


# =============================================================================
# Context Formatting Helpers
# =============================================================================

def _format_perception_context(
    hypotheses: list[dict[str, Any]] | None,
    key_insight: str | None,
    observations: dict[str, Any] | None,
    perceptions: list[dict[str, Any]] | None,
) -> str:
    """Format perception context for the verifier."""
    parts = []
    
    # Key insight (most important)
    if key_insight:
        parts.append(f"💡 KEY INSIGHT: {key_insight}")
    
    # Top hypotheses
    if hypotheses:
        parts.append("\n🎯 TOP TRANSFORMATION HYPOTHESES:")
        for h in hypotheses[:5]:  # Top 5
            rank = h.get("rank", "?")
            conf = h.get("confidence", "?")
            rule = h.get("rule", "")
            conf_icon = "🟢" if conf == "HIGH" else "🟡" if conf == "MEDIUM" else "🔴"
            parts.append(f"  {conf_icon} #{rank} [{conf}]: {rule}")
    
    # Observations
    if observations:
        parts.append("\n📋 PERCEIVER OBSERVATIONS:")
        if observations.get("common_input_features"):
            parts.append(f"  Input patterns: {', '.join(observations['common_input_features'])}")
        if observations.get("common_output_features"):
            parts.append(f"  Output patterns: {', '.join(observations['common_output_features'])}")
        if observations.get("size_pattern"):
            parts.append(f"  Size behavior: {observations['size_pattern']}")
        if observations.get("color_changes"):
            parts.append(f"  Color behavior: {observations['color_changes']}")
    
    # Object summary from perceptions (condensed)
    if perceptions:
        total_input_objects = 0
        total_output_objects = 0
        for perc in perceptions:
            if isinstance(perc, dict):
                inp_perc = perc.get('input', perc)
                out_perc = perc.get('output', {})
                total_input_objects += len(inp_perc.get('objects', []))
                total_output_objects += len(out_perc.get('objects', []))
        
        if total_input_objects > 0 or total_output_objects > 0:
            parts.append(f"\n🔍 OBJECTS DETECTED:")
            parts.append(f"  Training inputs: {total_input_objects} total objects")
            parts.append(f"  Training outputs: {total_output_objects} total objects")
    
    if not parts:
        return "(No perception context available)"
    
    return '\n'.join(parts)


def _format_hypotheses_for_check(hypotheses: list[dict[str, Any]] | None) -> str:
    """Format hypotheses as a checklist for verification."""
    if not hypotheses:
        return "No specific hypotheses to verify against"
    
    top = hypotheses[0] if hypotheses else {}
    return top.get("rule", "Pattern identified from training examples")


# =============================================================================
# Self-Verify Function (Synchronous)
# =============================================================================

def self_verify(
    model: str,
    model_id: str,
    extra_body: dict[str, Any] | None,
    max_tokens: int | None,
    code: str,
    explanation: str,
    train_examples: list[dict[str, Any]],
    test_inputs: list[np.ndarray],
    test_outputs: list[np.ndarray],
    # NEW: Perception context
    hypotheses: list[dict[str, Any]] | None = None,
    key_insight: str | None = None,
    observations: dict[str, Any] | None = None,
    perceptions: list[dict[str, Any]] | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Have the model verify its own output(s) with full perception context.

    Args:
        model: Model identifier
        model_id: Short model ID
        extra_body: Model-specific parameters
        max_tokens: Max tokens for response
        code: The generated transform code
        explanation: The solver's explanation of the pattern
        train_examples: All training examples
        test_inputs: ALL test input grids
        test_outputs: ALL corresponding outputs from the model
        hypotheses: Ranked transformation hypotheses from perceiver
        key_insight: Key insight about the puzzle
        observations: Task-level observations from perceiver
        perceptions: Per-example perceptions (objects, relationships)
        verbose: Whether to print progress

    Returns:
        Dict with 'decision' (CORRECT/WRONG/UNSURE), 'score', 'feedback'
    """
    # Format perception context
    perception_context = _format_perception_context(
        hypotheses=hypotheses,
        key_insight=key_insight,
        observations=observations,
        perceptions=perceptions,
    )
    
    # Format training pairs
    training_pairs = ""
    training_shapes = []
    training_colors = set()

    for i, ex in enumerate(train_examples):
        inp = np.array(ex['input'])
        out = np.array(ex['output'])
        training_shapes.append(f"{out.shape}")
        training_colors.update(np.unique(out).tolist())

        training_pairs += f"\n--- Training Pair {i+1} ---\n"
        training_pairs += f"Input ({inp.shape[0]}×{inp.shape[1]}):\n{grid_to_text(inp)}\n"
        training_pairs += f"Output ({out.shape[0]}×{out.shape[1]}):\n{grid_to_text(out)}\n"

    # Format ALL test results
    n_tests = len(test_inputs)
    test_results_str = ""
    all_output_colors = set()
    your_output_shapes = []
    
    for i, (test_input, test_output) in enumerate(zip(test_inputs, test_outputs)):
        all_output_colors.update(np.unique(test_output).tolist())
        your_output_shapes.append(f"{test_output.shape}")
        
        if n_tests > 1:
            test_results_str += f"\n--- Test {i+1}/{n_tests} ---\n"
        
        test_results_str += f"Input ({test_input.shape[0]}×{test_input.shape[1]}):\n"
        test_results_str += f"{grid_to_text(test_input)}\n"
        test_results_str += f"\nYour Output ({test_output.shape[0]}×{test_output.shape[1]}):\n"
        test_results_str += f"{grid_to_text(test_output)}\n"

    # Get key insight for verification (fallback to top hypothesis)
    verify_insight = key_insight
    if not verify_insight and hypotheses:
        verify_insight = hypotheses[0].get("rule", "Pattern from training")
    if not verify_insight:
        verify_insight = "Pattern identified from training examples"

    # Build prompt
    prompt = SELF_VERIFICATION_PROMPT.format(
        num_tests=n_tests,
        perception_context=perception_context,
        explanation=explanation if explanation else "Pattern identified from examples",
        code=code if code else "# Code not provided",
        training_pairs=training_pairs,
        test_results=test_results_str,
        training_output_shapes=", ".join(training_shapes),
        training_output_colors=str(training_colors),
        your_output_colors=str(all_output_colors),
        key_insight=verify_insight,
    )

    # Use HIGH reasoning for self-verification (always)
    verify_extra_body = {"reasoning": {"enabled": True}}

    response, elapsed = call_llm(
        model=model,
        system_prompt="You are verifying your own solution to an ARC puzzle. Be critical and thorough. Check against the key insight and hypotheses provided.",
        user_prompt=prompt,
        extra_body=verify_extra_body,
        max_tokens=max_tokens,
    )

    # Parse verdict and score
    result = {
        "decision": "UNSURE",
        "score": 50,  # Default score
        "feedback": "",
        "raw_response": response[:500],
    }

    response_upper = response.upper()

    # Extract VERDICT
    if "VERDICT:" in response_upper:
        verdict_idx = response_upper.find("VERDICT:")
        verdict_section = response_upper[verdict_idx:verdict_idx + 50]
        if "CORRECT" in verdict_section:
            result["decision"] = "CORRECT"
        elif "WRONG" in verdict_section:
            result["decision"] = "WRONG"
        else:
            result["decision"] = "UNSURE"
    else:
        # Fallback parsing
        if "CORRECT" in response_upper and "WRONG" not in response_upper[:response_upper.find("CORRECT") if "CORRECT" in response_upper else 0]:
            result["decision"] = "CORRECT"
        elif "WRONG" in response_upper:
            result["decision"] = "WRONG"

    # Extract SCORE
    score_match = re.search(r'SCORE:\s*(\d+)', response, re.IGNORECASE)
    if score_match:
        result["score"] = min(100, max(0, int(score_match.group(1))))
    else:
        # Infer score from verdict if not explicitly given
        if result["decision"] == "CORRECT":
            result["score"] = 90
        elif result["decision"] == "WRONG":
            result["score"] = 20
        else:
            result["score"] = 50

    # Extract FEEDBACK
    feedback_match = re.search(r'FEEDBACK:\s*(.+?)(?:\n\n|$)', response, re.IGNORECASE | re.DOTALL)
    if feedback_match:
        result["feedback"] = feedback_match.group(1).strip()[:500]
    else:
        # Try to extract any explanation after verdict
        if result["decision"] == "WRONG":
            wrong_idx = response_upper.find("WRONG")
            result["feedback"] = response[wrong_idx:wrong_idx + 300].strip()
        elif result["decision"] == "UNSURE":
            unsure_idx = response_upper.find("UNSURE") if "UNSURE" in response_upper else 0
            result["feedback"] = response[unsure_idx:unsure_idx + 300].strip()

    if verbose:
        emoji = "✓" if result["decision"] == "CORRECT" else "⚠️"
        print(f"    🔍 Self-verify: {emoji} {result['decision']} (score={result['score']})")

    return result
