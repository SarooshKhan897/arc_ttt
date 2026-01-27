"""Perceiver - structured grid analysis using LLM."""

import json
import re
import time
from typing import Any
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from solvers.perceiver.config import Role, get_role_model, get_role_extra_body
from solvers.perceiver.llms.client import call_llm
from solvers.perceiver.perception.objects import perceive_grid_fast, ObjectPreprocessor
from solvers.perceiver.utils.grid import grid_to_text
from solvers.perceiver.models import color_name

# =============================================================================
# JSON Schemas for Structured Outputs
# =============================================================================

PERCEIVER_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "perception_output",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "objects": {
                    "type": "array",
                    "description": "List of detected objects in the grid",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer", "description": "Unique object identifier"},
                            "color": {"type": "string", "description": "Color of the object"},
                            "shape": {"type": "string", "description": "Shape description (rectangle, L-shape, etc.)"},
                            "size": {"type": "integer", "description": "Number of cells"},
                            "position": {"type": "string", "description": "Position in grid (e.g., top-left, center)"},
                            "special": {"type": "string", "description": "Any special characteristics"}
                        },
                        "required": ["id", "color", "shape", "size", "position", "special"],
                        "additionalProperties": False
                    }
                },
                "relationships": {
                    "type": "array",
                    "description": "Relationships between objects",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "description": "Relationship type: adjacent, contained, aligned, or other"},
                            "obj1": {"type": "integer", "description": "First object ID (for adjacent relationships)"},
                            "obj2": {"type": "integer", "description": "Second object ID (for adjacent relationships)"},
                            "direction": {"type": "string", "description": "Direction of adjacency (left, right, above, below)"},
                            "inner": {"type": "integer", "description": "Inner object ID (for contained relationships)"},
                            "outer": {"type": "integer", "description": "Outer object ID (for contained relationships)"},
                            "objects": {"type": "array", "items": {"type": "integer"}, "description": "Object IDs (for aligned relationships)"},
                            "axis": {"type": "string", "description": "Alignment axis: horizontal or vertical"}
                        },
                        "required": ["type"],
                        "additionalProperties": False
                    }
                },
                "global_patterns": {
                    "type": "array",
                    "description": "Global patterns observed in the grid",
                    "items": {"type": "string"}
                },
                "notable_features": {
                    "type": "array",
                    "description": "Notable features worth mentioning",
                    "items": {"type": "string"}
                }
            },
            "required": ["objects", "relationships", "global_patterns", "notable_features"],
            "additionalProperties": False
        }
    }
}

TASK_PERCEIVER_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "task_perception_output",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "observations": {
                    "type": "object",
                    "description": "Observations about input/output patterns",
                    "properties": {
                        "common_input_features": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Features seen in all inputs"
                        },
                        "common_output_features": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Features seen in all outputs"
                        },
                        "size_pattern": {
                            "type": "string",
                            "description": "How size changes: same size, grows, shrinks, varies"
                        },
                        "color_changes": {
                            "type": "string",
                            "description": "Color behavior: none, recoloring, new colors added, colors removed"
                        }
                    },
                    "required": ["common_input_features", "common_output_features", "size_pattern", "color_changes"],
                    "additionalProperties": False
                },
                "transformation_hypotheses": {
                    "type": "array",
                    "description": "Exactly 5 transformation hypotheses ranked by likelihood",
                    "items": {
                        "type": "object",
                        "properties": {
                            "rank": {"type": "integer", "description": "Rank 1-5, 1 being most likely"},
                            "confidence": {"type": "string", "description": "HIGH, MEDIUM, or LOW"},
                            "rule": {"type": "string", "description": "Clear, specific description of the transformation rule"},
                            "evidence": {"type": "string", "description": "How this explains all training examples"}
                        },
                        "required": ["rank", "confidence", "rule", "evidence"],
                        "additionalProperties": False
                    }
                },
                "key_insight": {
                    "type": "string",
                    "description": "The single most important observation about this puzzle"
                }
            },
            "required": ["observations", "transformation_hypotheses", "key_insight"],
            "additionalProperties": False
        }
    }
}

# =============================================================================
# System Prompt
# =============================================================================

PERCEIVER_SYSTEM = """You are the PERCEIVER specialist in an ARC puzzle solving system.
Your ONLY job is to describe what you see in the grid - NO hypothesis generation.

Be PRECISE and EXHAUSTIVE. Describe EVERYTHING you observe.
Do NOT suggest what the transformation might be - just describe the grid.

Your output will be parsed as structured JSON."""


# =============================================================================
# Perceiver Function
# =============================================================================

def perceive(
    grid: np.ndarray,
    verbose: bool = False,
    max_retries: int = 3,
    retry_delay: float = 10.0,
) -> dict[str, Any]:
    """
    Call Perceiver to get structured grid representation using structured outputs.

    Args:
        grid: The grid to analyze
        verbose: Whether to print progress
        max_retries: Number of retry attempts on failure
        retry_delay: Delay in seconds between retries

    Returns:
        Structured perception dictionary
    """
    # Get code-based perception as fallback
    code_perception = perceive_grid_fast(grid)

    # Get all patterns for context
    all_patterns = {
        "tiling": ObjectPreprocessor.detect_tiling(grid),
        "frame": ObjectPreprocessor.detect_frame(grid),
    }

    # Build prompt
    user_prompt = f"""Analyze this grid and provide structured analysis.

GRID ({grid.shape[0]}x{grid.shape[1]}):
{grid_to_text(grid)}

CODE-DETECTED OBJECTS ({len(code_perception.objects)} found):
{json.dumps([obj.to_dict() for obj in code_perception.objects[:10]], indent=2)}

CODE-DETECTED PATTERNS:
- Symmetry: {code_perception.symmetry}
- Tiling: {all_patterns['tiling']}
- Frame: {all_patterns['frame']}

Provide your complete analysis. Include any objects, relationships, or patterns the code may have missed."""

    # Call LLM with structured output format
    model = get_role_model(Role.PERCEIVER)
    extra_body = get_role_extra_body(Role.PERCEIVER)
    
    for attempt in range(max_retries):
        try:
            response, elapsed = call_llm(
                model=model,
                system_prompt=PERCEIVER_SYSTEM,
                user_prompt=user_prompt,
                extra_body=extra_body,
                response_format=PERCEIVER_SCHEMA,
            )
            
            # With structured outputs, response should be valid JSON
            perception = json.loads(response)
            
            if verbose:
                n_objects = len(perception.get('objects', []))
                n_relations = len(perception.get('relationships', []))
                print(f"     Perceiver: {n_objects} objects, {n_relations} relations")
            
            # Add code-detected patterns as additional context
            perception["tiling"] = all_patterns['tiling']
            perception["frame"] = all_patterns['frame']
            
            return perception
            
        except (json.JSONDecodeError, Exception) as e:
            if attempt < max_retries - 1:
                if verbose:
                    print(f"     Perceiver: Error ({str(e)[:50]}), retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(retry_delay)
                continue
            else:
                if verbose:
                    print(f"     Perceiver: Failed after retries, using code fallback")

    # Fallback to code-based perception
    global_patterns_list = []
    if code_perception.symmetry.get("horizontal"):
        global_patterns_list.append("grid has horizontal symmetry")
    if code_perception.symmetry.get("vertical"):
        global_patterns_list.append("grid has vertical symmetry")
    if all_patterns['tiling'].get('is_tiled'):
        global_patterns_list.append("objects form a tiled/repeating pattern")
    if code_perception.patterns:
        global_patterns_list.extend(code_perception.patterns)
    
    fallback = {
        "objects": [
            {
                "id": i,
                "color": color_name(obj.color),
                "shape": "rectangle" if obj.is_rectangle else "irregular",
                "size": obj.size,
                "position": f"row {obj.bounding_box[0]}-{obj.bounding_box[2]}, col {obj.bounding_box[1]}-{obj.bounding_box[3]}",
                "special": "",
            }
            for i, obj in enumerate(code_perception.objects)
        ],
        "relationships": [],
        "global_patterns": global_patterns_list,
        "notable_features": [],
        "tiling": all_patterns['tiling'],
        "frame": all_patterns['frame'],
    }

    return fallback


def perceive_batch(
    grids: list[np.ndarray],
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Perceive multiple grids in parallel using ThreadPoolExecutor."""
    with ThreadPoolExecutor(max_workers=min(len(grids), 10)) as executor:
        results = list(executor.map(lambda g: perceive(g, verbose=False), grids))
    if verbose:
        print(f"     ✓ Perceived {len(results)} grids")
    return results


# =============================================================================
# Task-Level Perception with Transformation Hypotheses
# =============================================================================

TASK_PERCEIVER_SYSTEM = """You are the PERCEIVER specialist in an ARC puzzle solving system.

Your job is to analyze ALL training examples together and identify:
1. What objects/patterns exist in the grids
2. What transformations occur between input and output
3. Generate EXACTLY 5 POSSIBLE TRANSFORMATION HYPOTHESES (ranked by likelihood)

RULES FOR HYPOTHESES:
- Each hypothesis must be DIFFERENT - explore various interpretations
- Be SPECIFIC: "move objects" is bad, "move each colored shape 2 cells right" is good
- Rank 1 = your best guess, Rank 5 = least likely but still plausible
- Use evidence from multiple examples to support each hypothesis

Your output will be parsed as structured JSON."""


def perceive_task(
    task_data: dict[str, Any],
    verbose: bool = False,
    max_retries: int = 3,
    retry_delay: float = 10.0,
) -> dict[str, Any]:
    """
    Perceive an entire task and generate transformation hypotheses using structured outputs.
    
    This analyzes ALL training examples together to identify the transformation rule.
    
    Args:
        task_data: Task with 'train' and 'test' keys
        verbose: Whether to print progress
        max_retries: Number of retry attempts on failure
        retry_delay: Delay in seconds between retries
        
    Returns:
        Dict with 'observations', 'transformation_hypotheses', 'key_insight'
    """
    train_examples = task_data['train']
    
    # Build prompt with all examples
    prompt_parts = []
    prompt_parts.append("Analyze these training examples to identify the transformation rule.\n")
    
    for idx, pair in enumerate(train_examples):
        inp = np.array(pair['input'])
        out = np.array(pair['output'])
        
        prompt_parts.append(f"\n{'='*50}")
        prompt_parts.append(f"EXAMPLE {idx + 1}")
        prompt_parts.append(f"{'='*50}")
        
        prompt_parts.append(f"\nINPUT ({inp.shape[0]}x{inp.shape[1]}):")
        prompt_parts.append(grid_to_text(inp))
        
        prompt_parts.append(f"\nOUTPUT ({out.shape[0]}x{out.shape[1]}):")
        prompt_parts.append(grid_to_text(out))
        
        # Quick stats
        inp_colors = set(np.unique(inp).tolist())
        out_colors = set(np.unique(out).tolist())
        prompt_parts.append(f"\nStats: Input colors={inp_colors}, Output colors={out_colors}")
        prompt_parts.append(f"Size change: {inp.shape} → {out.shape}")
    
    # Add test input for context
    test_input = np.array(task_data['test'][0]['input'])
    prompt_parts.append(f"\n{'='*50}")
    prompt_parts.append("TEST INPUT (for context)")
    prompt_parts.append(f"{'='*50}")
    prompt_parts.append(f"\nTEST ({test_input.shape[0]}x{test_input.shape[1]}):")
    prompt_parts.append(grid_to_text(test_input))
    
    prompt_parts.append("""
    
Now analyze all examples and provide:
1. observations (common patterns in inputs and outputs)
2. transformation_hypotheses (EXACTLY 5, ranked from most to least likely)
3. key_insight (the single most important observation)
""")
    
    user_prompt = '\n'.join(prompt_parts)
    
    # Call LLM with structured output format
    model = get_role_model(Role.PERCEIVER)
    extra_body = get_role_extra_body(Role.PERCEIVER)
    
    if verbose:
        print("  👁️ Perceiver analyzing task...")
    
    for attempt in range(max_retries):
        try:
            response, elapsed = call_llm(
                model=model,
                system_prompt=TASK_PERCEIVER_SYSTEM,
                user_prompt=user_prompt,
                extra_body=extra_body,
                response_format=TASK_PERCEIVER_SCHEMA,
            )
            
            # With structured outputs, response should be valid JSON
            parsed = json.loads(response)
            
            result = {
                "observations": parsed.get("observations", {}),
                "transformation_hypotheses": parsed.get("transformation_hypotheses", []),
                "key_insight": parsed.get("key_insight", ""),
                "raw_response": response,
            }
            
            if verbose:
                n_hyp = len(result["transformation_hypotheses"])
                print(f"     ✓ Generated {n_hyp} transformation hypotheses")
                if result["key_insight"]:
                    insight = result["key_insight"][:60]
                    print(f"     💡 Key insight: {insight}...")
            
            return result
            
        except (json.JSONDecodeError, Exception) as e:
            if attempt < max_retries - 1:
                if verbose:
                    print(f"     ⚠️ Error ({str(e)[:50]}), retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(retry_delay)
                continue
            else:
                if verbose:
                    print("     ⚠️ Failed after retries, returning empty result...")
    
    # Fallback empty result
    return {
        "observations": {},
        "transformation_hypotheses": [],
        "key_insight": "",
        "raw_response": "",
    }


def _extract_hypotheses_text(text: str) -> list[dict[str, Any]]:
    """Extract hypotheses from unstructured text."""
    hypotheses = []
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        # Look for numbered items
        if re.match(r'^[1-5][\.\):]', line):
            hypotheses.append({
                "rank": len(hypotheses) + 1,
                "confidence": "MEDIUM",
                "rule": line.lstrip('0123456789.)]: '),
                "evidence": "",
            })
    
    return hypotheses


def format_hypotheses_for_solver(hypotheses: list[dict[str, Any]]) -> str:
    """Format hypotheses for inclusion in solver prompt."""
    if not hypotheses:
        return ""
    
    lines = [
        "=" * 60,
        "🔮 TRANSFORMATION HYPOTHESES (from Perceiver)",
        "=" * 60,
    ]
    
    for h in hypotheses:
        rank = h.get("rank", "?")
        conf = h.get("confidence", "?")
        rule = h.get("rule", "No rule")
        evidence = h.get("evidence", "")
        
        lines.append(f"\n#{rank} [{conf}]: {rule}")
        if evidence:
            lines.append(f"   Evidence: {evidence}")
    
    return '\n'.join(lines)
