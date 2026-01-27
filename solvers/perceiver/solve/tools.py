"""
ARC Solver with 4-Phase Tool Calls

This module implements a structured, tool-based approach to solving ARC puzzles.
The solver is broken into 4 sequential phases, each enforced as a tool call:

1. OBSERVE - Document observations for each example
2. HYPOTHESIZE - Formulate transformation hypotheses
3. VERIFY - Mentally verify hypothesis on all examples
4. IMPLEMENT - Write code and predict output grids

Each phase produces detailed natural language output that feeds into the next phase.
"""

import json
from typing import Any

import numpy as np

from solvers.perceiver.utils.grid import grid_to_text


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "observe_examples",
            "description": """PHASE 1: OBSERVE - Carefully analyze each input→output training example.

This is the FIRST step in solving an ARC puzzle. You MUST call this tool before any other.

For EACH training example, document your observations about:
- Dimensions: Are input/output the same size? Scaled? Cropped? Dynamically computed?
- Colors: Which colors appear, disappear, change, or remain fixed?
- Objects: What discrete "things" exist? (connected regions, shapes, lines, patterns)
- Spatial relationships: Distances, alignment, containment, symmetry between objects
- Information flow: What in the input determines what in the output?
- Connections: How are shapes connected to each other? How do connections change?

Be EXHAUSTIVE. The quality of your observations determines solution success.
Look for patterns that are CONSISTENT across ALL examples.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "example_observations": {
                        "type": "array",
                        "description": "Array of observations, one per training example",
                        "items": {
                            "type": "object",
                            "properties": {
                                "example_number": {
                                    "type": "integer",
                                    "description": "Which example (1, 2, 3, etc.)"
                                },
                                "dimensions": {
                                    "type": "string",
                                    "description": "Input shape, output shape, and relationship (same/scaled/cropped/etc.)"
                                },
                                "colors_analysis": {
                                    "type": "string",
                                    "description": "Which colors exist in input vs output. What changes? What's preserved?"
                                },
                                "objects_identified": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of discrete objects/shapes found in the input"
                                },
                                "spatial_relationships": {
                                    "type": "string",
                                    "description": "How objects relate to each other spatially (adjacent, contained, aligned, etc.)"
                                },
                                "transformation_observed": {
                                    "type": "string",
                                    "description": "What specific changes happen from input to output in this example?"
                                },
                                "key_insight": {
                                    "type": "string",
                                    "description": "The most important observation about this example"
                                }
                            },
                            "required": ["example_number", "dimensions", "colors_analysis", "objects_identified", "transformation_observed", "key_insight"]
                        }
                    },
                    "cross_example_patterns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Patterns that are CONSISTENT across ALL examples"
                    },
                    "invariants": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Things that are ALWAYS true (always same size, always preserves colors, etc.)"
                    }
                },
                "required": ["example_observations", "cross_example_patterns", "invariants"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "hypothesize_rule",
            "description": """PHASE 2: HYPOTHESIZE - Formulate transformation hypotheses based on observations.

This is the SECOND step. You MUST have called observe_examples first.

Based on your observations, formulate hypotheses about the transformation rule:
- What is the SIMPLEST rule that explains ALL examples?
- What does the output "know" that only the input could "tell" it?
- Is the transformation per-pixel, per-object, global, or compositional?
- Does one object serve as a template/reference for another?

Provide your TOP 3 hypotheses ranked by likelihood.
The best hypothesis is usually the SIMPLEST one that explains everything.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "hypotheses": {
                        "type": "array",
                        "description": "Your ranked hypotheses (best first)",
                        "minItems": 1,
                        "maxItems": 3,
                        "items": {
                            "type": "object",
                            "properties": {
                                "rank": {
                                    "type": "integer",
                                    "description": "1 = most likely, 2 = second most likely, etc."
                                },
                                "rule_description": {
                                    "type": "string",
                                    "description": "Clear, precise description of the transformation rule"
                                },
                                "transformation_type": {
                                    "type": "string",
                                    "enum": ["per_pixel", "per_object", "global", "compositional"],
                                    "description": "What type of transformation is this?"
                                },
                                "step_by_step": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Break down the rule into sequential steps"
                                },
                                "why_this_works": {
                                    "type": "string",
                                    "description": "Why does this hypothesis explain all the examples?"
                                },
                                "potential_issues": {
                                    "type": "string",
                                    "description": "Any edge cases or potential problems with this hypothesis"
                                }
                            },
                            "required": ["rank", "rule_description", "transformation_type", "step_by_step", "why_this_works"]
                        }
                    },
                    "chosen_hypothesis": {
                        "type": "integer",
                        "description": "Which hypothesis rank (1, 2, or 3) you will proceed with"
                    },
                    "reasoning_for_choice": {
                        "type": "string",
                        "description": "Why you chose this hypothesis over the others"
                    }
                },
                "required": ["hypotheses", "chosen_hypothesis", "reasoning_for_choice"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "verify_hypothesis",
            "description": """PHASE 3: VERIFY - Mentally execute your hypothesis on EVERY training example.

This is the THIRD step. You MUST have called hypothesize_rule first.

⚠️ CRITICAL: Before writing ANY code, you must verify your hypothesis works on ALL examples.

For EACH training example, mentally apply your chosen rule and check:
- Does the output SIZE match exactly?
- Does EVERY PIXEL match the expected output?
- Are there ANY exceptions or mismatches?

If ANY example fails verification, you must either:
1. Revise your hypothesis and re-verify
2. Choose a different hypothesis from Phase 2

DO NOT proceed to implementation until ALL examples pass verification.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "verification_results": {
                        "type": "array",
                        "description": "Verification result for each training example",
                        "items": {
                            "type": "object",
                            "properties": {
                                "example_number": {
                                    "type": "integer",
                                    "description": "Which example (1, 2, 3, etc.)"
                                },
                                "mental_trace": {
                                    "type": "string",
                                    "description": "Step-by-step trace of applying the rule to this input"
                                },
                                "predicted_output_description": {
                                    "type": "string",
                                    "description": "Describe what your rule produces"
                                },
                                "matches_expected": {
                                    "type": "boolean",
                                    "description": "Does your prediction match the expected output EXACTLY?"
                                },
                                "discrepancies": {
                                    "type": "string",
                                    "description": "If not matching, what's different? (empty string if matches)"
                                }
                            },
                            "required": ["example_number", "mental_trace", "predicted_output_description", "matches_expected", "discrepancies"]
                        }
                    },
                    "all_examples_pass": {
                        "type": "boolean",
                        "description": "Do ALL examples pass verification?"
                    },
                    "hypothesis_revision": {
                        "type": "string",
                        "description": "If any failed, describe how you're revising the hypothesis. Empty if all passed."
                    },
                    "confidence_level": {
                        "type": "string",
                        "enum": ["HIGH", "MEDIUM", "LOW"],
                        "description": "How confident are you in this hypothesis after verification?"
                    },
                    "ready_to_implement": {
                        "type": "boolean",
                        "description": "Are you ready to proceed to implementation? Only true if all_examples_pass is true."
                    }
                },
                "required": ["verification_results", "all_examples_pass", "confidence_level", "ready_to_implement"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "implement_solution",
            "description": """PHASE 4: IMPLEMENT - Write the transform function AND predict the output grid(s).

This is the FINAL step. You MUST have called verify_hypothesis first with all examples passing.

Provide BOTH:
1. The Python transform function
2. The predicted output grid(s) for each test input

We will verify BOTH the code execution AND the grid prediction - they should match.

Implementation requirements:
- Function signature: def transform(grid: np.ndarray) -> np.ndarray
- Use only numpy and scipy.ndimage
- Handle edge cases (empty masks, boundary objects, no matches)
- Use np.clip() for coordinate safety
- Always return out.astype(int)
- All output values must be integers 0-9""",
            "parameters": {
                "type": "object",
                "properties": {
                    "explanation": {
                        "type": "string",
                        "description": "Brief explanation of the transformation rule (2-3 sentences)"
                    },
                    "python_code": {
                        "type": "string",
                        "description": "Complete Python code with the transform function. Include necessary imports."
                    },
                    "predicted_outputs": {
                        "type": "array",
                        "description": "Predicted output grid for each test input",
                        "items": {
                            "type": "object",
                            "properties": {
                                "test_number": {
                                    "type": "integer",
                                    "description": "Which test input (1, 2, etc.)"
                                },
                                "predicted_grid": {
                                    "type": "array",
                                    "description": "The predicted output as a 2D array of integers 0-9",
                                    "items": {
                                        "type": "array",
                                        "items": {"type": "integer", "minimum": 0, "maximum": 9}
                                    }
                                },
                                "output_dimensions": {
                                    "type": "string",
                                    "description": "Expected dimensions of output (e.g., '5x5')"
                                },
                                "reasoning": {
                                    "type": "string",
                                    "description": "Brief explanation of how you derived this specific output"
                                }
                            },
                            "required": ["test_number", "predicted_grid", "output_dimensions", "reasoning"]
                        }
                    },
                    "edge_cases_handled": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of edge cases your code handles"
                    }
                },
                "required": ["explanation", "python_code", "predicted_outputs"]
            }
        }
    }
]


# =============================================================================
# TOOL CALLING SYSTEM PROMPT
# =============================================================================

TOOL_SYSTEM_PROMPT = """You are an expert in solving ARC-AGI puzzles. You solve puzzles through a rigorous 4-phase process using tool calls.

════════════════════════════════════════════════════════════════════
MANDATORY WORKFLOW - YOU MUST FOLLOW THESE STEPS IN ORDER
════════════════════════════════════════════════════════════════════

You have 4 tools available, and you MUST call them in this exact sequence:

1. observe_examples    → Document what you see in each example
2. hypothesize_rule    → Formulate and rank transformation hypotheses  
3. verify_hypothesis   → Mentally verify hypothesis on ALL examples
4. implement_solution  → Write code AND predict output grids

⚠️ CRITICAL RULES:
- You CANNOT skip phases or call them out of order
- You CANNOT call implement_solution until verify_hypothesis confirms all examples pass
- If verification fails, revise hypothesis and re-verify before implementing
- Your predicted_grid in implement_solution MUST match what your code produces

════════════════════════════════════════════════════════════════════
ARC PUZZLE CONSTRAINTS
════════════════════════════════════════════════════════════════════

GRID SPEC:
- 2D arrays, 1×1 to 30×30
- Colors are integers 0-9 ONLY:
    0=black  1=blue   2=red     3=green   4=yellow
    5=gray   6=magenta 7=orange  8=azure   9=maroon
- ANY value outside 0-9 = immediate failure

OUTPUT REQUIREMENTS:
- Function: def transform(grid: np.ndarray) -> np.ndarray
- Use numpy/scipy.ndimage only
- Return 2D int array (always end with `return out.astype(int)`)
- NO test code, NO examples, NO __main__

════════════════════════════════════════════════════════════════════
COMMON PATTERNS TO LOOK FOR
════════════════════════════════════════════════════════════════════

GEOMETRIC: rotate, flip, translate, scale
TILING: repeat, mirror-tile, stack
CROPPING: extract by color/size/shape, crop to content
FILL: flood fill, fill enclosed regions, checkerboard
COLOR: swap, replace, map colors
OBJECT: copy, move, align, sort, merge, delete
MASKING: apply mask, overlay, composite
SYMMETRY: complete partial symmetry, reflect
CONNECTIVITY: connect same color, separate components

════════════════════════════════════════════════════════════════════
COMMON MISTAKES TO AVOID
════════════════════════════════════════════════════════════════════

❌ Overfitting to one example (rule must work on ALL)
❌ Off-by-one errors (r_max+1 for slicing)
❌ Coordinate confusion (numpy = row, col NOT x, y)
❌ Hardcoding values that should be computed
❌ Wrong connectivity (4-connected vs 8-connected)
❌ Forgetting edge cases (empty results, boundary objects)

Now, analyze the puzzle and call the tools in sequence."""


# =============================================================================
# PROMPT GENERATION
# =============================================================================

def _format_objects_compact(objects: list[dict[str, Any]], max_objects: int = 8) -> str:
    """Format detected objects in a compact, decision-relevant way."""
    if not objects:
        return "  (no objects detected)"
    
    lines = []
    for i, obj in enumerate(objects[:max_objects]):
        obj_id = obj.get('id', i + 1)
        color = obj.get('color', 'unknown')
        shape = obj.get('shape', 'unknown')
        size = obj.get('size', '?')
        pos = obj.get('position', '')
        special = obj.get('special', '')
        
        desc = f"  #{obj_id}: {color} {shape} (size={size})"
        if pos:
            desc += f" @ {pos}"
        if special:
            desc += f" ★{special}"
        lines.append(desc)
    
    if len(objects) > max_objects:
        lines.append(f"  ... and {len(objects) - max_objects} more objects")
    
    return '\n'.join(lines)


def _format_delta(delta: dict[str, Any]) -> str:
    """Format a transformation delta compactly."""
    lines = []
    
    if delta.get('summary'):
        lines.append(f"     Summary: {delta['summary']}")
    
    if delta.get('size_change'):
        lines.append(f"     Size: {delta['size_change']}")
    
    if delta.get('color_changes'):
        for cc in delta['color_changes'][:3]:
            lines.append(f"     Color: {cc}")
    
    if delta.get('object_changes'):
        for oc in delta['object_changes'][:5]:
            lines.append(f"     Object: {oc}")
    
    return '\n'.join(lines) if lines else "     (no significant changes detected)"


def _format_hypotheses_section(
    hypotheses: list[dict[str, Any]],
    key_insight: str | None = None
) -> str:
    """Format hypotheses from perceiver for the prompt."""
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("🔮 TRANSFORMATION HYPOTHESES (from Perceiver)")
    lines.append("=" * 60)
    
    if key_insight:
        lines.append(f"\n💡 KEY INSIGHT: {key_insight}")
    
    lines.append("\nRanked hypotheses (use these to guide your analysis):")
    
    for h in hypotheses[:5]:
        rank = h.get("rank", "?")
        conf = h.get("confidence", "?")
        rule = h.get("rule", "No rule")
        evidence = h.get("evidence", "")
        
        lines.append(f"\n  #{rank} [{conf}]: {rule}")
        if evidence:
            lines.append(f"      Evidence: {evidence[:100]}...")
    
    lines.append("\n⚠️ Use these hypotheses as HINTS, but verify them yourself in Phase 1-3.")
    
    return '\n'.join(lines)


def _format_observations(observations: dict[str, Any]) -> str:
    """Format task-level observations."""
    lines = []
    lines.append("\n" + "─" * 60)
    lines.append("📋 PERCEIVER OBSERVATIONS")
    lines.append("─" * 60)
    
    if observations.get('common_input_features'):
        lines.append("\nCommon Input Features:")
        for f in observations['common_input_features'][:5]:
            lines.append(f"  • {f}")
    
    if observations.get('common_output_features'):
        lines.append("\nCommon Output Features:")
        for f in observations['common_output_features'][:5]:
            lines.append(f"  • {f}")
    
    if observations.get('size_pattern'):
        lines.append(f"\nSize Pattern: {observations['size_pattern']}")
    
    if observations.get('color_changes'):
        lines.append(f"Color Changes: {observations['color_changes']}")
    
    return '\n'.join(lines)


def generate_tool_prompt(
    task_data: dict[str, Any],
    perceptions: list[dict[str, Any]] | None = None,
    deltas: list[dict[str, Any]] | None = None,
    test_perception: dict[str, Any] | list[dict[str, Any]] | None = None,
    hypotheses: list[dict[str, Any]] | None = None,
    observations: dict[str, Any] | None = None,
    key_insight: str | None = None,
) -> str:
    """
    Generate the user prompt for tool-based solving.
    
    This provides the puzzle data AND perceiver analysis that the model will use
    with the 4-phase tool approach.
    
    Args:
        task_data: Task with 'train' and 'test' keys
        perceptions: Per-example perceptions (objects, relationships, patterns)
        deltas: Per-example transformation deltas
        test_perception: Perception(s) of test input(s)
        hypotheses: Ranked transformation hypotheses from perceiver
        observations: Task-level observations from perceiver
        key_insight: The key insight about the puzzle
    
    Returns:
        The complete prompt string
    """
    parts = []
    train_examples = task_data['train']
    test_inputs = task_data['test']
    
    parts.append("╔" + "═" * 58 + "╗")
    parts.append("║              ARC PUZZLE - SOLVE WITH TOOLS              ║")
    parts.append("╚" + "═" * 58 + "╝")
    parts.append("")
    parts.append("Use the 4 tools in sequence: observe → hypothesize → verify → implement")
    parts.append("")
    
    # Quick summary
    parts.append("┌─ QUICK SUMMARY ─────────────────────────────────────────┐")
    parts.append(f"│  Training Examples: {len(train_examples)}")
    parts.append(f"│  Test Cases: {len(test_inputs)}")
    
    size_info = []
    for idx, pair in enumerate(train_examples):
        inp = np.array(pair['input'])
        out = np.array(pair['output'])
        size_info.append(f"Ex{idx+1}: {inp.shape}→{out.shape}")
    parts.append(f"│  Size Changes: {' | '.join(size_info)}")
    parts.append("└─────────────────────────────────────────────────────────┘")
    
    # =================================================================
    # PERCEIVER HYPOTHESES (if available)
    # =================================================================
    if hypotheses:
        parts.append(_format_hypotheses_section(hypotheses, key_insight))
    
    if observations:
        parts.append(_format_observations(observations))
    
    # =================================================================
    # TRAINING EXAMPLES with Perception Data
    # =================================================================
    parts.append("\n" + "=" * 60)
    parts.append(f"TRAINING EXAMPLES ({len(train_examples)} total)")
    parts.append("=" * 60)
    
    for idx, pair in enumerate(train_examples):
        inp = np.array(pair['input'])
        out = np.array(pair['output'])
        
        parts.append(f"\n{'─'*60}")
        parts.append(f"EXAMPLE {idx + 1}")
        parts.append(f"{'─'*60}")
        
        # INPUT
        parts.append(f"\nINPUT ({inp.shape[0]}×{inp.shape[1]}):")
        parts.append(grid_to_text(inp))
        
        # Input perception (objects, relationships)
        if perceptions and idx < len(perceptions):
            perc = perceptions[idx]
            inp_perc = perc.get('input', perc) if 'input' in perc else perc
            
            inp_objects = inp_perc.get('objects', [])
            if inp_objects:
                parts.append(f"\n  🔍 DETECTED OBJECTS ({len(inp_objects)}):")
                parts.append(_format_objects_compact(inp_objects))
            
            inp_patterns = inp_perc.get('global_patterns', [])
            if inp_patterns:
                parts.append(f"\n  ✨ PATTERNS:")
                for p in inp_patterns[:5]:
                    parts.append(f"    • {p}")
        
        # OUTPUT
        parts.append(f"\nOUTPUT ({out.shape[0]}×{out.shape[1]}):")
        parts.append(grid_to_text(out))
        
        # Output perception
        if perceptions and idx < len(perceptions):
            perc = perceptions[idx]
            out_perc = perc.get('output', {})
            
            out_objects = out_perc.get('objects', [])
            if out_objects:
                parts.append(f"\n  🔍 OUTPUT OBJECTS ({len(out_objects)}):")
                parts.append(_format_objects_compact(out_objects))
        
        # Delta (transformation analysis)
        if deltas and idx < len(deltas):
            delta = deltas[idx]
            if delta.get('summary') or delta.get('object_changes'):
                parts.append(f"\n  📝 TRANSFORMATION DELTA:")
                parts.append(_format_delta(delta))
    
    # =================================================================
    # TEST INPUTS with Perception
    # =================================================================
    parts.append("\n" + "=" * 60)
    parts.append(f"TEST INPUT{'S' if len(test_inputs) > 1 else ''} ({len(test_inputs)} total)")
    parts.append("=" * 60)
    
    for idx, test_case in enumerate(test_inputs):
        test_input = np.array(test_case['input'])
        
        parts.append(f"\nTEST {idx + 1} ({test_input.shape[0]}×{test_input.shape[1]}):")
        parts.append(grid_to_text(test_input))
        
        # Test perception
        if test_perception:
            if isinstance(test_perception, list) and idx < len(test_perception):
                tp = test_perception[idx]
            elif isinstance(test_perception, dict):
                tp = test_perception
            else:
                tp = None
            
            if tp:
                tp_objects = tp.get('objects', [])
                if tp_objects:
                    parts.append(f"\n  🔍 TEST OBJECTS ({len(tp_objects)}):")
                    parts.append(_format_objects_compact(tp_objects))
    
    parts.append("\n" + "=" * 60)
    parts.append("BEGIN SOLVING - Call observe_examples first")
    parts.append("(Use the perceiver hypotheses above as hints, but verify them yourself)")
    parts.append("=" * 60)
    
    return '\n'.join(parts)


# =============================================================================
# TOOL RESPONSE HANDLERS
# =============================================================================

def handle_observe_response(response: dict[str, Any]) -> str:
    """Format the observation tool response for the next phase."""
    lines = [
        "",
        "✓ PHASE 1 COMPLETE: Observations recorded",
        "",
        "Key patterns identified:",
    ]
    
    for pattern in response.get("cross_example_patterns", []):
        lines.append(f"  • {pattern}")
    
    if response.get("invariants"):
        lines.append("\nInvariants (must hold in solution):")
        for inv in response["invariants"]:
            lines.append(f"  ✓ {inv}")
    
    lines.append("\n→ Now call hypothesize_rule to formulate transformation hypotheses")
    
    return '\n'.join(lines)


def handle_hypothesize_response(response: dict[str, Any]) -> str:
    """Format the hypothesis tool response for the next phase."""
    lines = [
        "",
        "✓ PHASE 2 COMPLETE: Hypotheses formulated",
        "",
    ]
    
    chosen = response.get("chosen_hypothesis", 1)
    for hyp in response.get("hypotheses", []):
        marker = "→" if hyp["rank"] == chosen else " "
        lines.append(f"{marker} Hypothesis #{hyp['rank']}: {hyp['rule_description'][:80]}...")
    
    lines.append(f"\nChosen: Hypothesis #{chosen}")
    lines.append(f"Reasoning: {response.get('reasoning_for_choice', 'N/A')}")
    lines.append("\n→ Now call verify_hypothesis to verify on ALL examples")
    
    return '\n'.join(lines)


def handle_verify_response(response: dict[str, Any]) -> str:
    """Format the verification tool response for the next phase."""
    all_pass = response.get("all_examples_pass", False)
    ready = response.get("ready_to_implement", False)
    
    lines = [
        "",
        "✓ PHASE 3 COMPLETE: Verification finished",
        "",
        "Results:"
    ]
    
    for result in response.get("verification_results", []):
        status = "✓" if result.get("matches_expected") else "✗"
        lines.append(f"  {status} Example {result['example_number']}: {'PASS' if result.get('matches_expected') else 'FAIL'}")
        if not result.get("matches_expected") and result.get("discrepancies"):
            lines.append(f"      Discrepancy: {result['discrepancies']}")
    
    lines.append(f"\nAll examples pass: {all_pass}")
    lines.append(f"Confidence: {response.get('confidence_level', 'N/A')}")
    
    if ready:
        lines.append("\n✓ Ready to implement!")
        lines.append("→ Now call implement_solution with code AND predicted output grids")
    else:
        lines.append("\n⚠️ NOT ready to implement - revise hypothesis first")
        if response.get("hypothesis_revision"):
            lines.append(f"Revision needed: {response['hypothesis_revision']}")
    
    return '\n'.join(lines)


def handle_implement_response(response: dict[str, Any]) -> dict[str, Any]:
    """
    Process the implementation response.
    
    Returns both the code and predicted grids for verification.
    """
    return {
        "explanation": response.get("explanation", ""),
        "code": response.get("python_code", ""),
        "predicted_outputs": [
            {
                "test_number": pred["test_number"],
                "grid": np.array(pred["predicted_grid"]),
                "reasoning": pred.get("reasoning", "")
            }
            for pred in response.get("predicted_outputs", [])
        ],
        "edge_cases": response.get("edge_cases_handled", [])
    }


# =============================================================================
# VALIDATION
# =============================================================================

def validate_tool_sequence(tool_calls: list[str]) -> tuple[bool, str]:
    """
    Validate that tools were called in the correct sequence.
    
    Returns (is_valid, error_message)
    """
    expected_sequence = ["observe_examples", "hypothesize_rule", "verify_hypothesis", "implement_solution"]
    
    if not tool_calls:
        return False, "No tools called"
    
    for i, (called, expected) in enumerate(zip(tool_calls, expected_sequence)):
        if called != expected:
            return False, f"Expected {expected} at step {i+1}, got {called}"
    
    if len(tool_calls) < 4:
        return False, f"Incomplete sequence: only {len(tool_calls)}/4 phases completed"
    
    return True, "Valid sequence"


def compare_code_and_grid(
    code: str,
    predicted_grid: np.ndarray,
    test_input: np.ndarray
) -> dict[str, Any]:
    """
    Execute the code and compare with the predicted grid.
    
    Returns comparison results.
    """
    import traceback
    
    result = {
        "code_executed": False,
        "code_output": None,
        "predicted_grid": predicted_grid,
        "code_matches_prediction": False,
        "error": None
    }
    
    try:
        # Execute the code
        exec_globals = {"np": np}
        try:
            from scipy import ndimage
            exec_globals["ndimage"] = ndimage
        except ImportError:
            pass
        
        exec(code, exec_globals)
        
        if "transform" not in exec_globals:
            result["error"] = "No transform function defined"
            return result
        
        transform = exec_globals["transform"]
        code_output = transform(test_input.copy())
        
        result["code_executed"] = True
        result["code_output"] = code_output
        
        # Compare
        if code_output.shape == predicted_grid.shape:
            result["code_matches_prediction"] = np.array_equal(code_output, predicted_grid)
        else:
            result["error"] = f"Shape mismatch: code={code_output.shape}, predicted={predicted_grid.shape}"
    
    except Exception as e:
        result["error"] = f"Execution error: {str(e)}\n{traceback.format_exc()}"
    
    return result
