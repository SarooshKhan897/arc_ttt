"""Verifier - evaluates solution quality using an LLM."""

import re
from typing import Any

import numpy as np

from solvers.perceiver.config import Role, get_role_model
from solvers.perceiver.llms.client import call_llm
from solvers.perceiver.utils.grid import grid_to_text

# =============================================================================
# System Prompt
# =============================================================================

VERIFIER_SYSTEM = """You are a code reviewer for ARC-AGI puzzle solutions.
Your role: Evaluate if the code logic is sound and likely to generalize to similar inputs.
Focus on correctness and practical applicability, not theoretical edge cases."""


VERIFIER_PROMPT = """## Task Context
A solver produced code that passes {num_examples} training examples.
Evaluate if the approach is sound and likely to work on ALL {num_tests} test input(s).

## Solver's Explanation:
{explanation}

## Solver's Code:
```python
{code}
```

## Training Examples:
{examples}

## Test Inputs ({num_tests} total):
{test_inputs}

## Review Focus:
1. Does the code logic match the explained pattern?
2. Are there any obvious bugs or typos?
3. Will it handle ALL test inputs correctly?

## Output Format:
VERDICT: [APPROVE / NEEDS_REVISION]
CONFIDENCE: [HIGH / MEDIUM / LOW]
SCORE: [0-100] (how likely the code works correctly on ALL tests)
FEEDBACK: [Brief note if NEEDS_REVISION]"""


# =============================================================================
# Verifier Function
# =============================================================================

async def verify(
    code: str,
    explanation: str,
    train_examples: list[dict[str, Any]],
    test_inputs: list[np.ndarray],
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Verify a solution using an LLM.

    Args:
        code: The transform code
        explanation: The solver's explanation
        train_examples: Training examples
        test_inputs: ALL test input grids
        verbose: Whether to print progress

    Returns:
        Dict with 'score', 'verdict', 'confidence', 'feedback'
    """
    # Format examples
    examples_str = ""
    for i, ex in enumerate(train_examples):
        inp = np.array(ex['input'])
        out = np.array(ex['output'])
        examples_str += f"\nExample {i+1}:\n"
        examples_str += f"Input ({inp.shape[0]}x{inp.shape[1]}):\n{grid_to_text(inp)}\n"
        examples_str += f"Output ({out.shape[0]}x{out.shape[1]}):\n{grid_to_text(out)}\n"

    # Format ALL test inputs
    n_tests = len(test_inputs)
    test_inputs_str = ""
    for i, test_input in enumerate(test_inputs):
        if n_tests > 1:
            test_inputs_str += f"\nTest Input {i+1}/{n_tests} ({test_input.shape[0]}x{test_input.shape[1]}):\n"
        else:
            test_inputs_str += f"({test_input.shape[0]}x{test_input.shape[1]}):\n"
        test_inputs_str += f"{grid_to_text(test_input)}\n"

    # Build prompt
    prompt = VERIFIER_PROMPT.format(
        num_examples=len(train_examples),
        num_tests=n_tests,
        explanation=explanation if explanation else "No explanation provided",
        code=code,
        examples=examples_str,
        test_inputs=test_inputs_str,
    )

    # Call LLM
    model = get_role_model(Role.VERIFIER)
    response, elapsed = await call_llm(
        model=model,
        system_prompt=VERIFIER_SYSTEM,
        user_prompt=prompt,
        extra_body={"reasoning": {"enabled": True}},
    )

    # Parse response
    result = {
        "score": 70,  # Default
        "verdict": "APPROVE",
        "confidence": "MEDIUM",
        "feedback": "",
        "raw_response": response[:500],
    }

    # Extract SCORE
    score_match = re.search(r'SCORE:\s*(\d+)', response)
    if score_match:
        result["score"] = int(score_match.group(1))

    # Extract VERDICT
    if "NEEDS_REVISION" in response.upper():
        result["verdict"] = "NEEDS_REVISION"
    elif "APPROVE" in response.upper():
        result["verdict"] = "APPROVE"

    # Extract CONFIDENCE
    if "HIGH" in response.upper() and "CONFIDENCE" in response.upper():
        result["confidence"] = "HIGH"
    elif "LOW" in response.upper() and "CONFIDENCE" in response.upper():
        result["confidence"] = "LOW"

    # Extract FEEDBACK
    feedback_match = re.search(r'FEEDBACK:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
    if feedback_match:
        result["feedback"] = feedback_match.group(1).strip()

    if verbose:
        print(f"    📊 Verifier: score={result['score']}, verdict={result['verdict']}")

    return result

