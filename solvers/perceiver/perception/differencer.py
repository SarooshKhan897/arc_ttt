"""Differencer - compares input/output pairs to extract transformation deltas."""

import json
import re
import time
from typing import Any
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from solvers.perceiver.config import Role, get_role_model, get_role_extra_body
from solvers.perceiver.llms.client import call_llm
from solvers.perceiver.perception.objects import compare_grids_fast
from solvers.perceiver.utils.grid import grid_to_text

# =============================================================================
# JSON Schema for Structured Outputs
# =============================================================================

DIFFERENCER_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "differencer_output",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "object_changes": {
                    "type": "array",
                    "description": "List of changes to objects (moved, deleted, created, scaled, rotated, etc.)",
                    "items": {"type": "string"}
                },
                "color_changes": {
                    "type": "array",
                    "description": "List of color-related changes (recoloring, new colors, color removals)",
                    "items": {"type": "string"}
                },
                "structural_changes": {
                    "type": "array",
                    "description": "Structural changes to the grid itself (size changes, tiling, merging)",
                    "items": {"type": "string"}
                },
                "constants": {
                    "type": "array",
                    "description": "Things that stayed the same (preserved elements)",
                    "items": {"type": "string"}
                },
                "summary": {
                    "type": "string",
                    "description": "Brief one-sentence summary of the transformation"
                }
            },
            "required": ["object_changes", "color_changes", "structural_changes", "constants", "summary"],
            "additionalProperties": False
        }
    }
}

# =============================================================================
# System Prompt
# =============================================================================

DIFFERENCER_SYSTEM = """You are the DIFFERENCER specialist in an ARC puzzle solving system.
Your ONLY job is to describe WHAT CHANGED between input and output - NO hypothesis about WHY.

Be PRECISE and SPECIFIC. Focus on OBSERVABLE changes, not interpretations.
Your output will be parsed as structured JSON."""


# =============================================================================
# Differencer Function
# =============================================================================

def difference(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    input_perception: dict[str, Any] | None = None,
    output_perception: dict[str, Any] | None = None,
    verbose: bool = False,
    max_retries: int = 3,
    retry_delay: float = 10.0,
) -> dict[str, Any]:
    """
    Call Differencer to compare input/output and extract delta using structured outputs.

    Args:
        input_grid: The input grid
        output_grid: The output grid
        input_perception: Optional pre-computed perception of input
        output_perception: Optional pre-computed perception of output
        verbose: Whether to print progress
        max_retries: Number of retry attempts on failure
        retry_delay: Delay in seconds between retries

    Returns:
        Structured delta dictionary
    """
    # Get code-based delta
    code_delta = compare_grids_fast(input_grid, output_grid)

    # Build prompt
    user_prompt = f"""Compare these INPUT and OUTPUT grids. What changed?

INPUT GRID ({input_grid.shape[0]}x{input_grid.shape[1]}):
{grid_to_text(input_grid)}

OUTPUT GRID ({output_grid.shape[0]}x{output_grid.shape[1]}):
{grid_to_text(output_grid)}

CODE-DETECTED CHANGES:
- Size change: {code_delta.size_change}
- Color changes: {code_delta.color_changes}
- Preserved: {code_delta.constants}
"""

    # Add perception context if available
    if input_perception:
        user_prompt += f"""
INPUT PERCEPTION:
{json.dumps(input_perception, indent=2)}
"""

    if output_perception:
        user_prompt += f"""
OUTPUT PERCEPTION:
{json.dumps(output_perception, indent=2)}
"""

    user_prompt += "\nProvide your analysis of what changed between input and output."

    # Call LLM with structured output format
    model = get_role_model(Role.DIFFERENCER)
    extra_body = get_role_extra_body(Role.DIFFERENCER)
    
    for attempt in range(max_retries):
        try:
            response, elapsed = call_llm(
                model=model,
                system_prompt=DIFFERENCER_SYSTEM,
                user_prompt=user_prompt,
                extra_body=extra_body,
                response_format=DIFFERENCER_SCHEMA,
            )
            
            # With structured outputs, response should be valid JSON
            delta = json.loads(response)
            
            if verbose:
                n_changes = len(delta.get('object_changes', []))
                print(f"     Differencer: {n_changes} object changes detected")
            
            return delta
            
        except (json.JSONDecodeError, Exception) as e:
            if attempt < max_retries - 1:
                if verbose:
                    print(f"     Differencer: Error ({str(e)[:50]}), retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(retry_delay)
                continue
            else:
                if verbose:
                    print("     Differencer: Failed after retries, using code fallback")

    # Fallback to code-based delta
    fallback = {
        "object_changes": [],
        "color_changes": code_delta.color_changes if hasattr(code_delta, 'color_changes') else [],
        "structural_changes": [],
        "constants": code_delta.constants if hasattr(code_delta, 'constants') else [],
        "summary": f"Size change: {code_delta.size_change}",
    }

    return fallback


def difference_batch(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    perceptions: list[tuple[dict, dict]] | None = None,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Compute deltas for multiple input/output pairs in parallel using ThreadPoolExecutor."""
    if perceptions is None:
        perceptions = [(None, None)] * len(pairs)

    def process_pair(args):
        (inp, out), (inp_perc, out_perc) = args
        return difference(inp, out, inp_perc, out_perc, verbose=False)

    with ThreadPoolExecutor(max_workers=min(len(pairs), 10)) as executor:
        results = list(executor.map(process_pair, zip(pairs, perceptions)))

    if verbose:
        print(f"     ✓ Computed {len(results)} deltas")

    return results
