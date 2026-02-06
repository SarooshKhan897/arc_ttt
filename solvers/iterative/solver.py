"""
ARC-AGI Solver using Tool-Call based "Test Time Training"

Uses iterative hypothesis refinement through OpenAI function calling.
Each tool call acts as a gradient step, bringing the model closer to the solution.

This version includes a `run_code` tool that gives the model access to execute
arbitrary Python code for analyzing examples, with the training examples and 
test input pre-loaded in the execution environment.

Usage:
    from solver import ARCSolver
    
    solver = ARCSolver(api_key="your-openrouter-key", model="anthropic/claude-opus-4.6")
    result = solver.solve(task_data)
"""

import json
import re
import copy
import sys
import io
import math
import time
import threading
import queue
import collections
import itertools
import functools
import operator
import statistics
from dataclasses import dataclass, field
from typing import Any
from openai import OpenAI
import numpy as np
from scipy import ndimage

# Import provider config for routing
from config import PROVIDER_CONFIG


# =============================================================================
# COLOR NAMES (for display)
# =============================================================================

COLOR_NAMES = [
    'black(0)', 'blue(1)', 'red(2)', 'green(3)', 'yellow(4)',
    'gray(5)', 'magenta(6)', 'orange(7)', 'azure(8)', 'maroon(9)'
]

def color_name(c: int) -> str:
    """Get human-readable color name."""
    return COLOR_NAMES[c] if 0 <= c < len(COLOR_NAMES) else f"color({c})"


# =============================================================================
# GRID ANALYSIS (Code-based, fast, no LLM)
# =============================================================================

@dataclass
class GridAnalysis:
    """Comprehensive analysis of a single grid."""
    rows: int
    cols: int
    total_cells: int
    colors_used: int
    color_palette: list[str]
    background_color: str
    background_color_id: int
    fill_ratio: float
    non_background_cells: int
    shapes_by_color: dict[str, int]
    total_shapes: int
    unique_colors: list[int]
    color_counts: dict[str, int]
    has_symmetry: dict[str, bool]
    
    def to_display(self) -> str:
        """Format for display in prompts."""
        sym_strs = [k for k, v in self.has_symmetry.items() if v]
        sym_info = f"Symmetry: {', '.join(sym_strs)}" if sym_strs else "No symmetry"
        shapes_info = ", ".join(f"{c}: {n}" for c, n in self.shapes_by_color.items())
        return (
            f"  Size: {self.rows}×{self.cols} ({self.total_cells} cells)\n"
            f"  Colors: {', '.join(self.color_palette)}\n"
            f"  Background: {self.background_color}\n"
            f"  Fill: {self.fill_ratio:.1f}% non-background\n"
            f"  Shapes: {self.total_shapes} total ({shapes_info})\n"
            f"  {sym_info}"
        )


@dataclass
class TransformAnalysis:
    """Analysis of transformation between input and output."""
    size_change_cells: int
    size_change_percent: float
    same_size: bool
    new_colors: list[str]
    removed_colors: list[str]
    colors_preserved: bool
    density_change_percent: float
    input_shape_count: int
    output_shape_count: int
    shape_count_change: int
    hints: list[str]
    
    def to_display(self) -> str:
        """Format for display in prompts."""
        return "  " + " | ".join(self.hints)


@dataclass
class ExampleAnalysis:
    """Complete analysis of an input/output training example."""
    example_num: int
    input_analysis: GridAnalysis
    output_analysis: GridAnalysis
    transform_analysis: TransformAnalysis
    
    def to_display(self) -> str:
        return f"""
Example #{self.example_num}:
INPUT:
{self.input_analysis.to_display()}
OUTPUT:
{self.output_analysis.to_display()}
TRANSFORM: {self.transform_analysis.to_display()}
"""


@dataclass
class TaskAnalysis:
    """Complete analysis of an ARC task."""
    task_id: str
    train_examples: list[ExampleAnalysis]
    test_inputs: list[GridAnalysis]
    consistent_size_preservation: bool
    consistent_color_preservation: bool
    consistent_shape_count: bool
    common_hints: list[str]
    
    def to_display(self) -> str:
        lines = ["=" * 60, "TASK ANALYSIS (Code-based)", "=" * 60]
        for ex in self.train_examples:
            lines.append(ex.to_display())
        
        lines.append("\n" + "=" * 60)
        lines.append("TEST INPUT(S)")
        lines.append("=" * 60)
        for i, test in enumerate(self.test_inputs, 1):
            lines.append(f"\nTest #{i}:")
            lines.append(test.to_display())
        
        lines.append("\n" + "=" * 60)
        lines.append("CROSS-EXAMPLE PATTERNS")
        lines.append("=" * 60)
        lines.append(f"  Size always preserved: {self.consistent_size_preservation}")
        lines.append(f"  Colors always preserved: {self.consistent_color_preservation}")
        lines.append(f"  Shape count preserved: {self.consistent_shape_count}")
        if self.common_hints:
            lines.append(f"  Common patterns: {', '.join(self.common_hints)}")
        
        return "\n".join(lines)


def analyze_grid(grid: np.ndarray) -> GridAnalysis:
    """Perform comprehensive analysis of a single grid."""
    grid = np.array(grid)
    rows, cols = grid.shape
    total_cells = rows * cols
    
    unique_colors = list(np.unique(grid))
    color_counts_raw = {int(c): int(np.sum(grid == c)) for c in unique_colors}
    
    background_id = max(color_counts_raw, key=color_counts_raw.get)
    background_name = color_name(background_id)
    
    non_bg_cells = total_cells - color_counts_raw[background_id]
    fill_ratio = (non_bg_cells / total_cells) * 100
    
    color_palette = [color_name(c) for c in unique_colors]
    
    shapes_by_color = {}
    total_shapes = 0
    
    for c in unique_colors:
        if c == background_id:
            continue
        mask = (grid == c).astype(int)
        _, num_shapes = ndimage.label(mask)
        if num_shapes > 0:
            shapes_by_color[color_name(c)] = num_shapes
            total_shapes += num_shapes
    
    has_symmetry = {
        "horizontal": bool(np.array_equal(grid, np.flip(grid, axis=1))),
        "vertical": bool(np.array_equal(grid, np.flip(grid, axis=0))),
        "diagonal": bool(rows == cols and np.array_equal(grid, grid.T)),
        "rotational_180": bool(np.array_equal(grid, np.rot90(grid, 2))),
    }
    
    color_counts = {color_name(int(k)): int(v) for k, v in color_counts_raw.items()}
    
    return GridAnalysis(
        rows=rows, cols=cols, total_cells=total_cells,
        colors_used=len(unique_colors), color_palette=color_palette,
        background_color=background_name, background_color_id=background_id,
        fill_ratio=fill_ratio, non_background_cells=non_bg_cells,
        shapes_by_color=shapes_by_color, total_shapes=total_shapes,
        unique_colors=[int(c) for c in unique_colors], color_counts=color_counts,
        has_symmetry=has_symmetry,
    )


def analyze_transform(input_analysis: GridAnalysis, output_analysis: GridAnalysis) -> TransformAnalysis:
    """Analyze the transformation between input and output."""
    size_change_cells = output_analysis.total_cells - input_analysis.total_cells
    size_change_percent = (size_change_cells / input_analysis.total_cells) * 100 if input_analysis.total_cells > 0 else 0
    same_size = input_analysis.total_cells == output_analysis.total_cells
    
    input_colors = set(input_analysis.unique_colors)
    output_colors = set(output_analysis.unique_colors)
    new_colors = [color_name(c) for c in (output_colors - input_colors)]
    removed_colors = [color_name(c) for c in (input_colors - output_colors)]
    colors_preserved = input_colors == output_colors
    
    density_change = output_analysis.fill_ratio - input_analysis.fill_ratio
    shape_change = output_analysis.total_shapes - input_analysis.total_shapes
    
    hints = []
    if same_size:
        hints.append("Same size")
    elif size_change_cells > 0:
        hints.append(f"Size +{size_change_percent:.0f}%")
    else:
        hints.append(f"Size {size_change_percent:.0f}%")
    
    if colors_preserved:
        hints.append("Colors preserved")
    if new_colors:
        hints.append(f"New: {', '.join(new_colors)}")
    if removed_colors:
        hints.append(f"Removed: {', '.join(removed_colors)}")
    
    if input_analysis.total_shapes == output_analysis.total_shapes:
        hints.append("Shape count same")
    elif shape_change > 0:
        hints.append(f"Shapes +{shape_change}")
    else:
        hints.append(f"Shapes {shape_change}")
    
    for sym_type, has_sym in output_analysis.has_symmetry.items():
        if has_sym and not input_analysis.has_symmetry.get(sym_type, False):
            hints.append(f"{sym_type} symmetry created")
    
    return TransformAnalysis(
        size_change_cells=size_change_cells, size_change_percent=size_change_percent,
        same_size=same_size, new_colors=new_colors, removed_colors=removed_colors,
        colors_preserved=colors_preserved, density_change_percent=density_change,
        input_shape_count=input_analysis.total_shapes, output_shape_count=output_analysis.total_shapes,
        shape_count_change=shape_change, hints=hints,
    )


def analyze_example(input_grid: np.ndarray, output_grid: np.ndarray, example_num: int = 1) -> ExampleAnalysis:
    """Analyze a single training example."""
    input_analysis = analyze_grid(input_grid)
    output_analysis = analyze_grid(output_grid)
    transform_analysis = analyze_transform(input_analysis, output_analysis)
    return ExampleAnalysis(example_num, input_analysis, output_analysis, transform_analysis)


def analyze_task(task_data: dict, task_id: str = "unknown") -> TaskAnalysis:
    """Perform comprehensive analysis of an entire ARC task."""
    train_analyses = []
    for i, pair in enumerate(task_data['train'], 1):
        input_grid = np.array(pair['input'])
        output_grid = np.array(pair['output'])
        train_analyses.append(analyze_example(input_grid, output_grid, i))
    
    test_analyses = []
    for test in task_data['test']:
        test_input = np.array(test['input'])
        test_analyses.append(analyze_grid(test_input))
    
    if train_analyses:
        consistent_size = all(ex.transform_analysis.same_size for ex in train_analyses)
        consistent_color = all(ex.transform_analysis.colors_preserved for ex in train_analyses)
        consistent_shapes = all(ex.transform_analysis.shape_count_change == 0 for ex in train_analyses)
        hint_sets = [set(ex.transform_analysis.hints) for ex in train_analyses]
        common_hints = list(set.intersection(*hint_sets)) if hint_sets else []
    else:
        consistent_size = consistent_color = consistent_shapes = False
        common_hints = []
    
    return TaskAnalysis(
        task_id=task_id, train_examples=train_analyses, test_inputs=test_analyses,
        consistent_size_preservation=consistent_size, consistent_color_preservation=consistent_color,
        consistent_shape_count=consistent_shapes, common_hints=common_hints,
    )


# =============================================================================
# TASK PERCEIVER (LLM-based hypothesis generation)
# =============================================================================

TASK_PERCEIVER_SYSTEM = """You are the PERCEIVER specialist analyzing ARC-AGI puzzles.

Your job is to analyze ALL training examples together and identify:
1. What objects/patterns exist in the grids
2. What transformations occur between input and output  
3. Generate EXACTLY 3 POSSIBLE TRANSFORMATION HYPOTHESES (ranked by likelihood)

RULES FOR HYPOTHESES:
- Each hypothesis must be DIFFERENT - explore various interpretations
- Be SPECIFIC: "move objects" is bad, "move each colored shape 2 cells right" is good
- Rank 1 = your best guess, Rank 3 = least likely but still plausible
- Use evidence from multiple examples to support each hypothesis

Keep your response concise and actionable."""


def perceive_task_llm(task_data: dict, client: OpenAI, model: str, reasoning_effort: str = None, verbose: bool = False) -> str:
    """
    Use LLM to perceive a task and generate transformation hypotheses.
    Returns formatted string with observations and hypotheses.
    """
    train_examples = task_data['train']
    
    # Build prompt with all examples
    prompt_parts = ["Analyze these training examples to identify the transformation rule.\n"]
    
    for idx, pair in enumerate(train_examples):
        inp = np.array(pair['input'])
        out = np.array(pair['output'])
        
        prompt_parts.append(f"\n{'='*50}")
        prompt_parts.append(f"EXAMPLE {idx + 1}")
        prompt_parts.append(f"{'='*50}")
        
        # Format grids
        inp_str = "[\n" + ",\n".join("  [" + ",".join(str(v) for v in row) + "]" for row in inp) + "\n]"
        out_str = "[\n" + ",\n".join("  [" + ",".join(str(v) for v in row) + "]" for row in out) + "\n]"
        
        prompt_parts.append(f"\nINPUT ({inp.shape[0]}x{inp.shape[1]}):")
        prompt_parts.append(inp_str)
        prompt_parts.append(f"\nOUTPUT ({out.shape[0]}x{out.shape[1]}):")
        prompt_parts.append(out_str)
        
        inp_colors = set(np.unique(inp).tolist())
        out_colors = set(np.unique(out).tolist())
        prompt_parts.append(f"\nStats: Input colors={inp_colors}, Output colors={out_colors}")
        prompt_parts.append(f"Size change: {inp.shape} → {out.shape}")
    
    # Add test input for context
    test_input = np.array(task_data['test'][0]['input'])
    test_str = "[\n" + ",\n".join("  [" + ",".join(str(v) for v in row) + "]" for row in test_input) + "\n]"
    prompt_parts.append(f"\n{'='*50}")
    prompt_parts.append("TEST INPUT (apply your rule to this)")
    prompt_parts.append(f"{'='*50}")
    prompt_parts.append(f"\nTEST ({test_input.shape[0]}x{test_input.shape[1]}):")
    prompt_parts.append(test_str)
    
    prompt_parts.append("""

Provide your analysis in this format:

## Key Observations
- What stays the same across examples?
- What changes?
- What patterns do you see?

## Hypothesis #1 (Most Likely)
[Specific rule description]
Evidence: [How this explains all examples]

## Hypothesis #2 (Alternative)
[Different interpretation]
Evidence: [Supporting observations]

## Hypothesis #3 (Fallback)
[Another possibility]
Evidence: [Why this could work]

## Recommended Approach
[How to implement the most likely hypothesis]
""")
    
    user_prompt = '\n'.join(prompt_parts)
    
    # Call LLM
    api_kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": TASK_PERCEIVER_SYSTEM},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 120000,  # Allow full reasoning with extended thinking models
    }
    
    api_kwargs["extra_body"] = {"reasoning": {"enabled": True}, **PROVIDER_CONFIG}
    
    try:
        response = _call_with_retry(
            lambda: client.chat.completions.create(**api_kwargs)
        )
        result = response.choices[0].message.content or ""
        
        if verbose:
            print(f"     ✓ Perceiver generated hypotheses")
        
        return result
        
    except Exception as e:
        if verbose:
            print(f"     ⚠️ Perceiver failed: {str(e)[:50]}")
        return ""

# Verbose output limit
VERBOSE_LIMIT = 100

def truncate(text: str, limit: int = VERBOSE_LIMIT) -> str:
    """Truncate text to limit characters with ellipsis."""
    if text is None:
        return "None"
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


class EmptyResponseError(Exception):
    """Raised when API returns empty/invalid response."""
    pass


def _call_with_retry(func, max_retries=3, base_delay=2.0):
    """Call a function with exponential backoff retry on transient errors."""
    last_error = None
    for attempt in range(max_retries):
        try:
            response = func()
            # Check for empty response
            if response is None or not hasattr(response, 'choices') or not response.choices:
                raise EmptyResponseError("API returned empty response (no choices)")
            return response
        except (json.JSONDecodeError, EmptyResponseError) as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"⚠️ {type(e).__name__} (attempt {attempt + 1}/{max_retries}), retrying in {delay}s...")
                time.sleep(delay)
        except Exception as e:
            # Check if it's a wrapped JSON error or other transient error
            error_str = str(e).lower()
            if "json" in error_str or "expecting value" in error_str or "empty" in error_str:
                last_error = e
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"⚠️ API response error (attempt {attempt + 1}/{max_retries}), retrying in {delay}s...")
                    time.sleep(delay)
            else:
                raise  # Re-raise non-transient errors immediately
    raise last_error

# Try to import numpy
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

# Try to import scipy
try:
    import scipy
    import scipy.ndimage
    import scipy.signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    scipy = None

# Try to import scikit-image
try:
    import skimage
    import skimage.measure
    import skimage.morphology
    import skimage.segmentation
    import skimage.filters
    import skimage.transform
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    skimage = None

# Try to import scikit-learn
try:
    import sklearn
    import sklearn.cluster
    import sklearn.preprocessing
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    sklearn = None


class ARCSolver:
    """
    Solves ARC-AGI tasks using iterative tool-call based refinement.
    
    The solver gives the model tools to:
    1. Run arbitrary Python code for analysis (with examples pre-loaded)
    2. Test transformation hypotheses against training examples
    3. Inspect patterns in the grids
    4. Submit final answers
    
    Each tool call provides structured feedback that helps the model
    refine its hypothesis - mimicking test-time training.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "anthropic/claude-opus-4.6",
        base_url: str = "https://openrouter.ai/api/v1",
        max_iterations: int = 20,
        verbose: bool = True,
        reasoning_effort: str = None,  # "low", "medium", "high", "xhigh" for OpenAI models
        # Phase 1 (hypothesis) can use a different model
        hypothesis_model: str = None,  # If None, uses main model
        hypothesis_reasoning_effort: str = None,  # If None, uses main reasoning_effort
        # Early stopping support
        stop_event: "threading.Event" = None  # Set by external code to signal early stop
    ):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.reasoning_effort = reasoning_effort
        # Phase 1 settings (default to main model/effort if not specified)
        self.hypothesis_model = hypothesis_model or model
        self.hypothesis_reasoning_effort = hypothesis_reasoning_effort if hypothesis_reasoning_effort is not None else reasoning_effort
        self.stop_event = stop_event  # For early stopping
        self.current_task = None
        self.iteration_history = []
        self.pending_submission = None  # Stores answer awaiting confirmation
        self.working_code = None  # Code that passed training
        self.working_rule = None  # Rule description for working code
        
        # Cost tracking
        self.total_cost = 0.0
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
    
    def _track_usage(self, response):
        """Track token usage and cost from API response."""
        if hasattr(response, 'usage') and response.usage:
            usage = response.usage
            self.prompt_tokens += getattr(usage, 'prompt_tokens', 0) or 0
            self.completion_tokens += getattr(usage, 'completion_tokens', 0) or 0
            self.total_tokens += getattr(usage, 'total_tokens', 0) or 0
            # OpenRouter returns cost in the usage object
            self.total_cost += getattr(usage, 'cost', 0.0) or 0.0
    
    def get_usage(self) -> dict[str, Any]:
        """Return current usage stats for this solver instance."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
        }
        
    @property
    def tools(self) -> list[dict]:
        """Define the tools available to the model."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "observe_examples",
                    "description": (
                        "MANDATORY FIRST STEP: Document your observations about the training examples.\n\n"
                        "Before doing ANY analysis or coding, you MUST call this tool to record:\n"
                        "1. Key patterns you observe in the input/output pairs\n"
                        "2. Invariants that must hold in any solution\n\n"
                        "This creates a structured observation record that guides your solution.\n"
                        "You MUST call this BEFORE using run_code or any other tool."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "examples": {
                                "type": "array",
                                "description": "Per-example observations",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "example_num": {"type": "integer", "description": "Example number (0-indexed)"},
                                        "input_description": {"type": "string", "description": "What you see in the input grid"},
                                        "output_description": {"type": "string", "description": "What you see in the output grid"},
                                        "transformation": {"type": "string", "description": "How input becomes output"}
                                    },
                                    "required": ["example_num", "input_description", "output_description", "transformation"]
                                }
                            },
                            "key_patterns": {
                                "type": "array",
                                "description": "List of key patterns identified across all examples (use bullet points starting with •)",
                                "items": {"type": "string"}
                            },
                            "invariants": {
                                "type": "array",
                                "description": "List of invariants that must hold in solution (things that never change)",
                                "items": {"type": "string"}
                            },
                            "cross_example_patterns": {
                                "type": "string",
                                "description": "What patterns are consistent across ALL examples?"
                            }
                        },
                        "required": ["key_patterns", "invariants"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "run_code",
                    "description": (
                        "Execute Python code to analyze training examples, test inputs, or experiment with transformations.\n\n"
                        "## Pre-loaded Variables:\n"
                        "- `train_examples`: List of dicts, each with 'input' and 'output' (2D lists)\n"
                        "- `test_input`: First test input (2D list)\n"
                        "- `test_inputs`: ALL test inputs (list of 2D lists)\n"
                        "- `num_train`: Number of training examples\n"
                        "- `num_test`: Number of test inputs\n\n"
                        "## Available Libraries:\n"
                        "numpy, scipy, skimage, sklearn, collections, itertools, math, re, copy\n\n"
                        "## Example:\n"
                        "```python\n"
                        "for i, ex in enumerate(train_examples):\n"
                        "    inp, out = np.array(ex['input']), np.array(ex['output'])\n"
                        "    print(f'Ex {i}: {inp.shape} -> {out.shape}')\n"
                        "```\n\n"
                        "Use print() for output."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "Python code to execute."},
                            "purpose": {"type": "string", "description": "What you're analyzing."}
                        },
                        "required": ["code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "apply_transformation",
                    "description": (
                        "Test a transformation rule by applying it to a training example's input "
                        "and comparing against the expected output. Returns detailed feedback on "
                        "any mismatches to help refine your hypothesis. Use this iteratively to "
                        "develop and verify your transformation rule."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "rule_description": {
                                "type": "string",
                                "description": (
                                    "Natural language description of the transformation rule you're testing. "
                                    "Be specific about what changes and what stays the same."
                                )
                            },
                            "python_code": {
                                "type": "string",
                                "description": (
                                    "Python code defining a function called 'transform' that takes "
                                    "input_grid (a 2D list of integers 0-9) and returns output_grid "
                                    "(also a 2D list of integers). Example:\n"
                                    "def transform(input_grid):\n"
                                    "    # Your transformation logic\n"
                                    "    return output_grid"
                                )
                            },
                            "example_index": {
                                "type": "integer",
                                "description": "Which training example to test against (0-indexed)"
                            }
                        },
                        "required": ["rule_description", "python_code", "example_index"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "test_on_all_training",
                    "description": (
                        "Test your transformation rule against ALL training examples at once. "
                        "Use this once you're confident your rule works on individual examples "
                        "to verify it generalizes across the full training set.\n\n"
                        "Once your code passes ALL training, use `apply_to_test` to apply it to the test input."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "rule_description": {
                                "type": "string",
                                "description": "Natural language description of the transformation rule"
                            },
                            "python_code": {
                                "type": "string",
                                "description": "Python code with a 'transform' function"
                            }
                        },
                        "required": ["rule_description", "python_code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "inspect_grids",
                    "description": (
                        "Analyze specific aspects of the input/output grids across examples. "
                        "Useful for understanding patterns before forming hypotheses."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": (
                                    "What to analyze. Examples: 'grid dimensions', 'color counts', "
                                    "'unique colors', 'bounding boxes of non-zero regions', "
                                    "'positions of color N', 'symmetry check'"
                                )
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_differences",
                    "description": (
                        "Compare two training examples to identify what varies and what stays constant. "
                        "This helps identify which input features the transformation rule must be sensitive to, "
                        "and helps avoid overfitting to specific positions or layouts. "
                        "Returns: differences in inputs, differences in outputs, and whether they CORRELATE."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "example_a": {
                                "type": "integer",
                                "description": "First training example index (0-indexed)"
                            },
                            "example_b": {
                                "type": "integer",
                                "description": "Second training example index (0-indexed)"
                            }
                        },
                        "required": ["example_a", "example_b"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "apply_to_test",
                    "description": (
                        "Apply your transformation code to ALL test inputs and preview the outputs.\n\n"
                        "WORKFLOW:\n"
                        "1. First pass test_on_all_training to verify your code works\n"
                        "2. Call apply_to_test to see outputs for all test inputs\n"
                        "3. Review - if satisfied, call confirm_submission(verified=true)"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "python_code": {"type": "string", "description": "Python code with transform(input_grid) -> output_grid"},
                            "final_rule_description": {"type": "string", "description": "Description of the transformation rule"}
                        },
                        "required": ["python_code", "final_rule_description"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "submit_answer",
                    "description": (
                        "Submit your answer for review. The grid will be shown back to you for verification. "
                        "After reviewing, call confirm_submission to finalize, or use other tools to refine. "
                        "PREFER using apply_to_test instead - it automatically applies your code to test input."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "test_output": {
                                "type": "array",
                                "items": {
                                    "type": "array",
                                    "items": {"type": "integer"}
                                },
                                "description": "The predicted output grid for the test input"
                            },
                            "final_rule_description": {
                                "type": "string",
                                "description": "Final description of the transformation rule you discovered"
                            },
                            "confidence": {
                                "type": "number",
                                "description": "Confidence level 0-1 in your answer"
                            }
                        },
                        "required": ["test_output", "final_rule_description"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "confirm_submission",
                    "description": (
                        "Final gate before submission. You MUST review the apply_to_test preview and verify correctness.\n\n"
                        
                        "## Pre-Confirmation Checklist\n"
                        "Compare the preview against your expectations:\n"
                        "- [ ] Output dimensions match what the rule predicts\n"
                        "- [ ] All colors are correct (no unexpected 0s or wrong fills)\n"
                        "- [ ] Spatial relationships preserved (objects in right positions)\n"
                        "- [ ] Edge cases handled (boundaries, empty regions, overlaps)\n"
                        "- [ ] Pattern is consistent with ALL training examples, not just one\n\n"
                        
                        "## Red Flags (reject if ANY present)\n"
                        "- Grid is all zeros or unchanged from input\n"
                        "- Dimensions wildly different from training outputs\n"
                        "- Colors appear that weren't in training outputs\n"
                        "- Obvious artifacts: random scattered pixels, incomplete fills\n"
                        "- Shape/structure doesn't match the pattern from training\n\n"
                        
                        "## Decision\n"
                        "- verified=true: Preview matches expected transformation. Submit.\n"
                        "- verified=false: Something is off. Return to debugging.\n\n"
                        
                        "Rejection is CHEAP. Wrong submissions are EXPENSIVE. When in doubt, reject."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "verified": {
                                "type": "boolean",
                                "description": (
                                    "true = Preview is correct, submit as final answer. "
                                    "false = Preview has issues, abort and return to run_code for debugging."
                                )
                            },
                            "verification_notes": {
                                "type": "string",
                                "description": (
                                    "Brief explanation of what you verified (if true) or what looks wrong (if false). "
                                    "E.g., 'Output 5x5 matches training pattern, all regions filled correctly' or "
                                    "'Rejecting: bottom-right corner shows color 3 but should be 0'"
                                )
                            }
                        },
                        "required": ["verified", "verification_notes"]
                    }
                }
            }
        ]
    
    def _create_execution_environment(self) -> dict:
        """Create the execution environment with task data pre-loaded."""
        task = self.current_task
        
        # Deep copy to prevent modification
        train_examples = copy.deepcopy(task.get("train", []))
        
        # Load ALL test inputs
        test_inputs = [copy.deepcopy(tc.get("input", [])) for tc in task.get("test", [])]
        test_input = test_inputs[0] if test_inputs else []  # First test for convenience
        num_test = len(test_inputs)
        
        exec_globals = {
            "__builtins__": {
                "range": range,
                "len": len,
                "min": min,
                "max": max,
                "abs": abs,
                "sum": sum,
                "enumerate": enumerate,
                "zip": zip,
                "list": list,
                "dict": dict,
                "set": set,
                "tuple": tuple,
                "sorted": sorted,
                "reversed": reversed,
                "any": any,
                "all": all,
                "map": map,
                "filter": filter,
                "int": int,
                "float": float,
                "str": str,
                "bool": bool,
                "isinstance": isinstance,
                "print": print,
                "round": round,
                "pow": pow,
                "divmod": divmod,
                "hex": hex,
                "bin": bin,
                "oct": oct,
                "ord": ord,
                "chr": chr,
                "repr": repr,
                "type": type,
                "hasattr": hasattr,
                "getattr": getattr,
                "setattr": setattr,
                "callable": callable,
                "iter": iter,
                "next": next,
                "slice": slice,
                "frozenset": frozenset,
                "bytes": bytes,
                "bytearray": bytearray,
                "Exception": Exception,
                "ValueError": ValueError,
                "TypeError": TypeError,
                "IndexError": IndexError,
                "KeyError": KeyError,
                "AttributeError": AttributeError,
                "__import__": __import__,  # Allow import statements
            },
            "copy": copy,
            "deepcopy": copy.deepcopy,
            # Task data - THE KEY FEATURE
            "train_examples": train_examples,
            "test_input": test_input,       # First test input (for convenience)
            "test_inputs": test_inputs,     # ALL test inputs
            "num_train": len(train_examples),
            "num_test": num_test,
            # Standard library modules
            "math": math,
            "re": re,
            "collections": collections,
            "deque": collections.deque,
            "Counter": collections.Counter,
            "defaultdict": collections.defaultdict,
            "OrderedDict": collections.OrderedDict,
            "itertools": itertools,
            "functools": functools,
            "reduce": functools.reduce,
            "operator": operator,
            "statistics": statistics,
        }
        
        # Add numpy if available
        if HAS_NUMPY:
            exec_globals["np"] = np
            exec_globals["numpy"] = np
            exec_globals["__builtins__"]["array"] = np.array
        
        # Add scipy if available
        if HAS_SCIPY:
            exec_globals["scipy"] = scipy
            exec_globals["ndimage"] = scipy.ndimage
            exec_globals["signal"] = scipy.signal
        
        # Add scikit-image if available
        if HAS_SKIMAGE:
            exec_globals["skimage"] = skimage
            exec_globals["measure"] = skimage.measure
            exec_globals["morphology"] = skimage.morphology
            exec_globals["segmentation"] = skimage.segmentation
            exec_globals["filters"] = skimage.filters
            exec_globals["transform"] = skimage.transform
        
        # Add scikit-learn if available
        if HAS_SKLEARN:
            exec_globals["sklearn"] = sklearn
            exec_globals["cluster"] = sklearn.cluster
            exec_globals["numpy"] = np
            exec_globals["__builtins__"]["array"] = np.array
        
        return exec_globals
    
    def _handle_run_code(self, code: str, purpose: str = None, timeout: float = 60.0) -> dict:
        """
        Execute arbitrary Python code for analysis with timeout.
        
        The code runs in an environment with train_examples, test_input, and test_inputs pre-loaded.
        """
        exec_globals = self._create_execution_environment()
        result_queue = queue.Queue()
        
        def run_code():
            # Capture stdout
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()
            
            result = {
                "status": "success",
                "output": "",
                "error": None
            }
            
            if purpose:
                result["purpose"] = purpose
            
            try:
                exec(code, exec_globals)
                result["output"] = captured_output.getvalue()
                
                # Check if output is empty
                if not result["output"].strip():
                    result["output"] = "(No output - use print() to see results)"
                else:
                    # Add analysis prompt to encourage reflection
                    result["analysis_prompt"] = (
                        "Before your next tool call, answer: "
                        "1) What pattern does this output reveal? "
                        "2) How does this refine your hypothesis? "
                        "3) What specific question should your next code answer?"
                    )
                    
            except Exception as e:
                result["status"] = "error"
                error_msg = f"{type(e).__name__}: {str(e)}"
                # Add helpful hints for common errors
                if "'module' object is not callable" in str(e):
                    error_msg += "\n💡 Hint: You're trying to call a module as a function. Use module.function() instead (e.g., collections.Counter(), transform.resize(), etc.)"
                result["error"] = error_msg
                result["output"] = captured_output.getvalue()
                
            finally:
                sys.stdout = old_stdout
            
            result_queue.put(result)
        
        # Run in thread with timeout
        exec_thread = threading.Thread(target=run_code, daemon=True)
        exec_thread.start()
        exec_thread.join(timeout=timeout)
        
        if exec_thread.is_alive():
            return {
                "status": "error",
                "output": "",
                "error": f"TimeoutError: Code execution exceeded {timeout:.0f}s limit. Your code may have an infinite loop or be too expensive.",
                "suggestion": "Check for infinite loops (while True, recursion) or very expensive operations. Simplify your analysis."
            }
        
        try:
            return result_queue.get_nowait()
        except queue.Empty:
            return {
                "status": "error",
                "output": "",
                "error": "ExecutionError: Code execution failed unexpectedly"
            }
    
    def _grid_to_string(self, grid: list[list[int]], indent: int = 0) -> str:
        """Convert a grid to a readable string representation."""
        prefix = " " * indent
        lines = []
        for row in grid:
            lines.append(prefix + "[" + ", ".join(str(c) for c in row) + "]")
        return "[\n" + ",\n".join(lines) + "\n" + prefix + "]"
    
    def _find_mismatches(
        self, 
        predicted: list[list[int]], 
        expected: list[list[int]],
        input_grid: list[list[int]]
    ) -> list[dict]:
        """Find cell-level mismatches between predicted and expected grids."""
        mismatches = []
        
        pred_h, pred_w = len(predicted), len(predicted[0]) if predicted else 0
        exp_h, exp_w = len(expected), len(expected[0]) if expected else 0
        
        max_h = max(pred_h, exp_h)
        max_w = max(pred_w, exp_w)
        
        for i in range(max_h):
            for j in range(max_w):
                pred_val = predicted[i][j] if i < pred_h and j < pred_w else "OUT_OF_BOUNDS"
                exp_val = expected[i][j] if i < exp_h and j < exp_w else "OUT_OF_BOUNDS"
                
                if pred_val != exp_val:
                    input_val = None
                    if input_grid and i < len(input_grid) and j < len(input_grid[0]):
                        input_val = input_grid[i][j]
                    
                    mismatches.append({
                        "position": [i, j],
                        "predicted": pred_val,
                        "expected": exp_val,
                        "input_value": input_val
                    })
        
        return mismatches
    
    def _generate_hint(
        self,
        predicted: list[list[int]],
        expected: list[list[int]],
        input_grid: list[list[int]],
        mismatches: list[dict]
    ) -> str:
        """Generate a helpful hint based on the type of errors."""
        hints = []
        
        pred_h, pred_w = len(predicted), len(predicted[0]) if predicted else 0
        exp_h, exp_w = len(expected), len(expected[0]) if expected else 0
        
        if pred_h != exp_h or pred_w != exp_w:
            hints.append(f"Shape mismatch: you produced {pred_h}x{pred_w}, expected {exp_h}x{exp_w}. "
                        "Check if output should be cropped, padded, or resized.")
        
        if mismatches:
            wrong_colors = {}
            for m in mismatches:
                if m["predicted"] != "OUT_OF_BOUNDS" and m["expected"] != "OUT_OF_BOUNDS":
                    key = (m["predicted"], m["expected"])
                    wrong_colors[key] = wrong_colors.get(key, 0) + 1
            
            if wrong_colors:
                most_common = max(wrong_colors.items(), key=lambda x: x[1])
                hints.append(f"Common error: predicting {most_common[0][0]} where it should be "
                           f"{most_common[0][1]} ({most_common[1]} times)")
            
            rows = [m["position"][0] for m in mismatches]
            cols = [m["position"][1] for m in mismatches]
            
            if len(set(rows)) <= 2:
                hints.append(f"Errors concentrated in rows: {sorted(set(rows))}")
            if len(set(cols)) <= 2:
                hints.append(f"Errors concentrated in columns: {sorted(set(cols))}")
        
        return " | ".join(hints) if hints else "Check your logic carefully against the example."
    
    def _execute_transform(self, python_code: str, input_grid: list[list[int]], timeout: float = 60.0) -> tuple[Any, str | None]:
        """Safely execute the transformation code with timeout."""
        input_copy = copy.deepcopy(input_grid)
        result_queue = queue.Queue()
        
        exec_globals = {
            "__builtins__": {
                "range": range,
                "len": len,
                "min": min,
                "max": max,
                "abs": abs,
                "sum": sum,
                "enumerate": enumerate,
                "zip": zip,
                "list": list,
                "dict": dict,
                "set": set,
                "tuple": tuple,
                "sorted": sorted,
                "reversed": reversed,
                "any": any,
                "all": all,
                "map": map,
                "filter": filter,
                "int": int,
                "float": float,
                "str": str,
                "bool": bool,
                "isinstance": isinstance,
                "print": print,
                "__import__": __import__,  # Allow import statements
            },
            "copy": copy,
            "deepcopy": copy.deepcopy,
            # Pre-import common libraries
            "collections": collections,
            "deque": collections.deque,
            "Counter": collections.Counter,
            "defaultdict": collections.defaultdict,
            "itertools": itertools,
            "functools": functools,
            "math": math,
        }
        
        # Add numpy if available
        if HAS_NUMPY:
            exec_globals["np"] = np
            exec_globals["numpy"] = np
            exec_globals["__builtins__"]["array"] = np.array
        
        def run_transform():
            try:
                exec(python_code, exec_globals)
                
                if "transform" not in exec_globals:
                    result_queue.put((None, "No 'transform' function defined in your code"))
                    return
                
                transform_fn = exec_globals["transform"]
                result = transform_fn(input_copy)
                
                if not isinstance(result, list):
                    result_queue.put((None, f"transform() must return a list, got {type(result).__name__}"))
                    return
                if not result:
                    result_queue.put((None, "transform() returned an empty list"))
                    return
                if not all(isinstance(row, list) for row in result):
                    result_queue.put((None, "transform() must return a 2D list (list of lists)"))
                    return
                if not all(isinstance(cell, int) for row in result for cell in row):
                    result_queue.put((None, "All cells must be integers"))
                    return
                
                result_queue.put((result, None))
                
            except Exception as e:
                result_queue.put((None, f"{type(e).__name__}: {str(e)}"))
        
        # Run in thread with timeout
        exec_thread = threading.Thread(target=run_transform, daemon=True)
        exec_thread.start()
        exec_thread.join(timeout=timeout)
        
        if exec_thread.is_alive():
            return None, f"TimeoutError: Code execution exceeded {timeout:.0f}s. Possible infinite loop or expensive computation."
        
        try:
            return result_queue.get_nowait()
        except queue.Empty:
            return None, "ExecutionError: Code execution failed unexpectedly"
    
    def _handle_apply_transformation(
        self,
        rule_description: str,
        python_code: str,
        example_index: int
    ) -> dict:
        """Handle the apply_transformation tool call."""
        task = self.current_task
        
        if example_index < 0 or example_index >= len(task["train"]):
            return {
                "status": "error",
                "error": f"Invalid example_index {example_index}. Valid range: 0-{len(task['train'])-1}"
            }
        
        example = task["train"][example_index]
        input_grid = example["input"]
        expected_output = example["output"]
        
        predicted, error = self._execute_transform(python_code, input_grid)
        
        if error:
            return {
                "status": "execution_error",
                "error": error,
                "suggestion": "Fix the Python syntax or logic error and try again."
            }
        
        if predicted == expected_output:
            return {
                "status": "correct",
                "message": f"✓ Your transformation correctly produces the expected output for training example {example_index}!",
                "next_step": "Test on other training examples or use test_on_all_training to verify generalization."
            }
        
        mismatches = self._find_mismatches(predicted, expected_output, input_grid)
        hint = self._generate_hint(predicted, expected_output, input_grid, mismatches)
        
        return {
            "status": "incorrect",
            "predicted_shape": f"{len(predicted)}x{len(predicted[0])}",
            "expected_shape": f"{len(expected_output)}x{len(expected_output[0])}",
            "total_cells_wrong": len(mismatches),
            "total_cells": len(expected_output) * len(expected_output[0]),
            "accuracy": f"{100 * (1 - len(mismatches) / (len(expected_output) * len(expected_output[0]))):.1f}%",
            "all_mismatches": mismatches,
            "hint": hint,
            "suggestion": "Analyze the mismatches and refine your transformation rule."
        }
    
    def _handle_test_on_all_training(
        self,
        rule_description: str,
        python_code: str
    ) -> dict:
        """Handle testing on all training examples."""
        task = self.current_task
        results = []
        all_correct = True
        
        for i, example in enumerate(task["train"]):
            predicted, error = self._execute_transform(python_code, example["input"])
            
            if error:
                results.append({
                    "example": i,
                    "status": "execution_error",
                    "error": error
                })
                all_correct = False
            elif predicted == example["output"]:
                results.append({
                    "example": i,
                    "status": "correct"
                })
            else:
                mismatches = self._find_mismatches(predicted, example["output"], example["input"])
                results.append({
                    "example": i,
                    "status": "incorrect",
                    "cells_wrong": len(mismatches),
                    "all_mismatches": mismatches
                })
                all_correct = False
        
        if all_correct:
            # Store the working code and rule
            self.working_code = python_code
            self.working_rule = rule_description
            # Signal to show first test input
            self.training_verified = True
            self.ready_for_test = True
            
            return {
                "status": "all_correct",
                "message": "✓ Your transformation works on ALL training examples!",
                "results": results,
                "next_step": "Training complete! The test input will now be shown. Use apply_to_test to apply your code."
            }
        
        return {
            "status": "some_failures",
            "message": "Your transformation doesn't work on all training examples yet.",
            "results": results,
            "suggestion": "Analyze which examples fail and refine your rule to handle all cases."
        }
    
    def _handle_analyze_differences(self, example_a: int, example_b: int) -> dict:
        """Compare two training examples to identify what varies and what stays constant."""
        task = self.current_task
        n_examples = len(task["train"])
        
        if example_a < 0 or example_a >= n_examples:
            return {"error": f"Invalid example_a: {example_a}. Valid range: 0-{n_examples-1}"}
        if example_b < 0 or example_b >= n_examples:
            return {"error": f"Invalid example_b: {example_b}. Valid range: 0-{n_examples-1}"}
        if example_a == example_b:
            return {"error": "Please compare two DIFFERENT examples"}
        
        ex_a = task["train"][example_a]
        ex_b = task["train"][example_b]
        
        inp_a, out_a = ex_a["input"], ex_a["output"]
        inp_b, out_b = ex_b["input"], ex_b["output"]
        
        result = {
            "comparing": f"Example {example_a} vs Example {example_b}",
            "input_differences": {},
            "output_differences": {},
            "correlations": [],
            "insights": []
        }
        
        inp_shape_a = (len(inp_a), len(inp_a[0]) if inp_a else 0)
        inp_shape_b = (len(inp_b), len(inp_b[0]) if inp_b else 0)
        result["input_differences"]["shapes"] = {
            "example_a": f"{inp_shape_a[0]}x{inp_shape_a[1]}",
            "example_b": f"{inp_shape_b[0]}x{inp_shape_b[1]}",
            "same": inp_shape_a == inp_shape_b
        }
        
        inp_colors_a = set(c for row in inp_a for c in row)
        inp_colors_b = set(c for row in inp_b for c in row)
        result["input_differences"]["colors"] = {
            "example_a": sorted(inp_colors_a),
            "example_b": sorted(inp_colors_b),
            "same": inp_colors_a == inp_colors_b,
            "only_in_a": sorted(inp_colors_a - inp_colors_b),
            "only_in_b": sorted(inp_colors_b - inp_colors_a)
        }
        
        inp_nonzero_a = sum(1 for row in inp_a for c in row if c != 0)
        inp_nonzero_b = sum(1 for row in inp_b for c in row if c != 0)
        result["input_differences"]["non_zero_cells"] = {
            "example_a": inp_nonzero_a,
            "example_b": inp_nonzero_b
        }
        
        out_shape_a = (len(out_a), len(out_a[0]) if out_a else 0)
        out_shape_b = (len(out_b), len(out_b[0]) if out_b else 0)
        result["output_differences"]["shapes"] = {
            "example_a": f"{out_shape_a[0]}x{out_shape_a[1]}",
            "example_b": f"{out_shape_b[0]}x{out_shape_b[1]}",
            "same": out_shape_a == out_shape_b
        }
        
        out_colors_a = set(c for row in out_a for c in row)
        out_colors_b = set(c for row in out_b for c in row)
        result["output_differences"]["colors"] = {
            "example_a": sorted(out_colors_a),
            "example_b": sorted(out_colors_b),
            "same": out_colors_a == out_colors_b,
            "only_in_a": sorted(out_colors_a - out_colors_b),
            "only_in_b": sorted(out_colors_b - out_colors_a)
        }
        
        out_nonzero_a = sum(1 for row in out_a for c in row if c != 0)
        out_nonzero_b = sum(1 for row in out_b for c in row if c != 0)
        result["output_differences"]["non_zero_cells"] = {
            "example_a": out_nonzero_a,
            "example_b": out_nonzero_b
        }
        
        if inp_shape_a != inp_shape_b:
            if out_shape_a != out_shape_b:
                result["correlations"].append({
                    "type": "shape_correlation",
                    "observation": "Input shapes differ AND output shapes differ",
                    "implication": "Output size likely depends on input size"
                })
            else:
                result["correlations"].append({
                    "type": "shape_normalization",
                    "observation": "Input shapes differ but output shapes are SAME",
                    "implication": "Output may be a fixed-size summary/extraction of input"
                })
        
        if inp_colors_a != inp_colors_b:
            if out_colors_a != out_colors_b:
                result["correlations"].append({
                    "type": "color_correlation",
                    "observation": "Input colors differ AND output colors differ",
                    "implication": "Output colors likely depend on input colors"
                })
            else:
                result["correlations"].append({
                    "type": "color_abstraction",
                    "observation": "Input colors differ but output colors are SAME",
                    "implication": "Colors may be symbolic/abstracted in transformation"
                })
        
        inp_ratio = inp_nonzero_a / inp_nonzero_b if inp_nonzero_b > 0 else 0
        out_ratio = out_nonzero_a / out_nonzero_b if out_nonzero_b > 0 else 0
        if inp_ratio > 0 and out_ratio > 0:
            if abs(inp_ratio - out_ratio) < 0.1:
                result["correlations"].append({
                    "type": "density_correlation",
                    "observation": f"Non-zero cell ratio roughly preserved (in: {inp_ratio:.2f}, out: {out_ratio:.2f})",
                    "implication": "Transformation likely preserves relative density/count"
                })
        
        insights = []
        if inp_shape_a == inp_shape_b and out_shape_a == out_shape_b:
            insights.append("✓ Both examples have same shapes - rule likely works on fixed-size or relative positioning")
        if inp_colors_a == inp_colors_b:
            insights.append("✓ Same input colors - rule doesn't need to handle different color sets")
        else:
            insights.append("⚠ Different input colors - rule must be COLOR-AGNOSTIC or handle varying colors")
        if result["correlations"]:
            insights.append(f"Found {len(result['correlations'])} correlations between input/output differences")
        else:
            insights.append("No clear correlations found - differences may be independent")
        
        result["insights"] = insights
        result["recommendation"] = (
            "Look for how DIFFERENCES in inputs lead to DIFFERENCES in outputs. "
            "Your rule should reference RELATIVE features (positions relative to objects, color relationships) "
            "not ABSOLUTE features (row 3, column 5)."
        )
        
        return result
    
    def _handle_inspect_grids(self, query: str) -> dict:
        """Handle grid inspection queries."""
        task = self.current_task
        query_lower = query.lower()
        
        result = {"query": query, "training_examples": [], "test_input": {}}
        
        for i, example in enumerate(task["train"]):
            inp = example["input"]
            out = example["output"]
            
            example_info = {"example": i}
            
            if "dimension" in query_lower or "size" in query_lower or "shape" in query_lower:
                example_info["input_shape"] = f"{len(inp)}x{len(inp[0])}"
                example_info["output_shape"] = f"{len(out)}x{len(out[0])}"
            
            if "color" in query_lower or "unique" in query_lower:
                inp_colors = set(c for row in inp for c in row)
                out_colors = set(c for row in out for c in row)
                example_info["input_colors"] = sorted(inp_colors)
                example_info["output_colors"] = sorted(out_colors)
                example_info["colors_added"] = sorted(out_colors - inp_colors)
                example_info["colors_removed"] = sorted(inp_colors - out_colors)
            
            if "count" in query_lower:
                inp_counts = {}
                out_counts = {}
                for row in inp:
                    for c in row:
                        inp_counts[c] = inp_counts.get(c, 0) + 1
                for row in out:
                    for c in row:
                        out_counts[c] = out_counts.get(c, 0) + 1
                example_info["input_color_counts"] = inp_counts
                example_info["output_color_counts"] = out_counts
            
            if "position" in query_lower:
                color_match = re.search(r'color\s*(\d+)', query_lower)
                if color_match:
                    color = int(color_match.group(1))
                    inp_pos = [(i, j) for i, row in enumerate(inp) for j, c in enumerate(row) if c == color]
                    out_pos = [(i, j) for i, row in enumerate(out) for j, c in enumerate(row) if c == color]
                    example_info[f"color_{color}_input_positions"] = inp_pos
                    example_info[f"color_{color}_output_positions"] = out_pos
            
            if "bounding" in query_lower or "bbox" in query_lower:
                def get_bbox(grid):
                    non_zero = [(i, j) for i, row in enumerate(grid) for j, c in enumerate(row) if c != 0]
                    if not non_zero:
                        return None
                    rows, cols = zip(*non_zero)
                    return {"min_row": min(rows), "max_row": max(rows), 
                           "min_col": min(cols), "max_col": max(cols)}
                
                example_info["input_bbox"] = get_bbox(inp)
                example_info["output_bbox"] = get_bbox(out)
            
            if "symmetr" in query_lower:
                def check_symmetry(grid):
                    h, w = len(grid), len(grid[0])
                    h_sym = all(grid[i][j] == grid[h-1-i][j] for i in range(h) for j in range(w))
                    v_sym = all(grid[i][j] == grid[i][w-1-j] for i in range(h) for j in range(w))
                    return {"horizontal": h_sym, "vertical": v_sym}
                
                example_info["input_symmetry"] = check_symmetry(inp)
                example_info["output_symmetry"] = check_symmetry(out)
            
            result["training_examples"].append(example_info)
        
        test_inp = task["test"][0]["input"]
        test_info = {}
        
        if "dimension" in query_lower or "size" in query_lower or "shape" in query_lower:
            test_info["shape"] = f"{len(test_inp)}x{len(test_inp[0])}"
        if "color" in query_lower or "unique" in query_lower:
            test_info["colors"] = sorted(set(c for row in test_inp for c in row))
        
        result["test_input"] = test_info
        
        return result
    
    def _handle_submit_answer(
        self,
        test_output: list[list[int]],
        final_rule_description: str,
        confidence: float = None
    ) -> dict:
        """Handle answer submission - returns grid for verification."""
        # Store pending submission
        self.pending_submission = {
            "answer": test_output,
            "rule": final_rule_description,
            "confidence": confidence,
            "source": "submit_answer"
        }
        
        # Format grid for display
        grid_display = self._grid_to_string(test_output, indent=2)
        
        return {
            "status": "pending_verification",
            "message": "Review your answer below. Call confirm_submission(verified=true) to finalize, or use other tools to refine.",
            "grid_preview": grid_display,
            "shape": f"{len(test_output)}x{len(test_output[0]) if test_output else 0}",
            "rule": final_rule_description
        }
    
    def _handle_confirm_submission(self, verified: bool, verification_notes: str = "") -> dict:
        """Confirm and finalize the pending submission."""
        if not verified:
            self.pending_submission = None
            return {
                "status": "rejected",
                "reason": verification_notes,
                "message": "Preview rejected - good catch! Use run_code to debug what went wrong, or analyze_differences to compare with expected output. Then try apply_to_test again with fixed code."
            }
        
        if not self.pending_submission:
            return {
                "status": "error",
                "message": "No pending submission to confirm. Call apply_to_test first."
            }
        
        # Return the confirmed submission
        return {
            "status": "confirmed",
            "answer": self.pending_submission["answer"],
            "rule": self.pending_submission["rule"],
            "source": self.pending_submission.get("source", "apply_to_test"),
            "verification_notes": verification_notes
        }
    
    def _handle_apply_to_test(
        self,
        python_code: str,
        final_rule_description: str,
        adaptations: str = None
    ) -> dict:
        """
        Apply code to ALL test inputs and return outputs for review.
        """
        # Gate: warn if training wasn't explicitly verified
        if not self.training_verified:
            return {
                "status": "warning",
                "message": (
                    "⚠️ You haven't called test_on_all_training yet. "
                    "Verify your code passes ALL training examples before applying to test. "
                    "This helps catch bugs before submission."
                ),
                "suggestion": "Call test_on_all_training(python_code=..., rule_description=...) first."
            }
        
        task = self.current_task
        n_tests = len(task["test"])
        
        test_outputs = []
        test_errors = []
        
        for i, test_case in enumerate(task["test"]):
            test_input = test_case["input"]
            predicted, error = self._execute_transform(python_code, test_input)
            
            if error:
                test_errors.append({"test_index": i, "error": error})
                test_outputs.append(None)
            else:
                test_outputs.append(predicted)
        
        if test_errors:
            return {
                "status": "test_execution_error",
                "message": "Code failed on some test input(s).",
                "errors": test_errors,
                "suggestion": "Test inputs follow the SAME transformation but may have variations. Adapt your code."
            }
        
        # Format outputs for review
        output_preview = []
        for i, grid in enumerate(test_outputs):
            if grid:
                h, w = len(grid), len(grid[0]) if grid else 0
                output_preview.append({"test_index": i, "shape": f"{h}x{w}", "grid": grid})
        
        # Final answer format
        final_answer = test_outputs[0] if n_tests == 1 else test_outputs
        
        # Store pending submission
        self.pending_submission = {
            "answer": final_answer,
            "rule": final_rule_description,
            "adaptations": adaptations,
            "source": "apply_to_test",
            "code": python_code
        }
        
        return {
            "status": "preview",
            "message": f"✓ Code ran successfully on {n_tests} test input(s). REVIEW THE OUTPUT CAREFULLY before confirming!",
            "outputs": output_preview,
            "rule": final_rule_description,
            "next_step": "IMPORTANT: Check the grid dimensions, colors, and patterns. If correct, call confirm_submission(verified=true). If ANYTHING looks wrong, call confirm_submission(verified=false) and debug with run_code."
        }
    
    def _handle_observe_examples(self, arguments: dict) -> dict:
        """Handle the observe_examples tool call - records observations."""
        examples = arguments.get("examples", [])
        key_patterns = arguments.get("key_patterns", [])
        invariants = arguments.get("invariants", [])
        cross_example = arguments.get("cross_example_patterns", "")
        
        # Format the output nicely
        output_lines = [
            "✓ PHASE 1 COMPLETE: Observations recorded",
            "",
            "Key patterns identified:"
        ]
        
        for pattern in key_patterns:
            # Ensure bullet point format
            if not pattern.strip().startswith("•"):
                pattern = f"• {pattern}"
            output_lines.append(f"  {pattern}")
        
        output_lines.append("")
        output_lines.append("Invariants (must hold in solution):")
        for inv in invariants:
            if not inv.strip().startswith("✓"):
                inv = f"✓ {inv}"
            output_lines.append(f"  {inv}")
        
        if cross_example:
            output_lines.append("")
            output_lines.append(f"Cross-example patterns: {cross_example}")
        
        if examples:
            output_lines.append("")
            output_lines.append("Per-example observations:")
            for ex in examples:
                ex_num = ex.get("example_num", 0)
                output_lines.append(f"  Example {ex_num}:")
                output_lines.append(f"    Input: {ex.get('input_description', 'N/A')[:100]}")
                output_lines.append(f"    Output: {ex.get('output_description', 'N/A')[:100]}")
                output_lines.append(f"    Transform: {ex.get('transformation', 'N/A')[:100]}")
        
        output_lines.append("")
        output_lines.append("→ Now proceed to Phase 2: HYPOTHESIS FORMATION")
        output_lines.append("  Use run_code to analyze patterns and form a testable hypothesis.")
        
        return {
            "status": "success",
            "output": "\n".join(output_lines),
            "observations_recorded": True,
            "next_step": "Proceed to hypothesis formation. Use run_code to analyze patterns and test your hypothesis."
        }
    
    def _process_tool_call(self, tool_name: str, arguments: dict) -> dict:
        """Route tool calls to appropriate handlers."""
        if tool_name == "observe_examples":
            return self._handle_observe_examples(arguments)
        elif tool_name == "run_code":
            return self._handle_run_code(
                arguments.get("code", ""),
                arguments.get("purpose")
            )
        elif tool_name == "apply_transformation":
            return self._handle_apply_transformation(
                arguments.get("rule_description", ""),
                arguments.get("python_code", ""),
                arguments.get("example_index", 0)
            )
        elif tool_name == "test_on_all_training":
            return self._handle_test_on_all_training(
                arguments.get("rule_description", ""),
                arguments.get("python_code", "")
            )
        elif tool_name == "inspect_grids":
            return self._handle_inspect_grids(arguments.get("query", ""))
        elif tool_name == "analyze_differences":
            return self._handle_analyze_differences(
                arguments.get("example_a", 0),
                arguments.get("example_b", 1)
            )
        elif tool_name == "apply_to_test":
            return self._handle_apply_to_test(
                arguments.get("python_code", ""),
                arguments.get("final_rule_description", ""),
                arguments.get("adaptations")
            )
        elif tool_name == "submit_answer":
            return self._handle_submit_answer(
                arguments.get("test_output", []),
                arguments.get("final_rule_description", ""),
                arguments.get("confidence")
            )
        elif tool_name == "confirm_submission":
            return self._handle_confirm_submission(
                arguments.get("verified", False),
                arguments.get("verification_notes", "")
            )
        else:
            return {"error": f"Unknown tool: {tool_name}"}
    
    def _format_task_for_prompt(self, task: dict) -> str:
        """Format the task data for the system prompt - TRAINING + ALL TEST INPUTS."""
        lines = ["## Training Examples\n"]
        
        for i, example in enumerate(task["train"]):
            lines.append(f"### Training Example {i}")
            lines.append(f"Input ({len(example['input'])}x{len(example['input'][0])}):")
            lines.append(self._grid_to_string(example["input"], indent=2))
            lines.append(f"Output ({len(example['output'])}x{len(example['output'][0])}):")
            lines.append(self._grid_to_string(example["output"], indent=2))
            lines.append("")
        
        # Show ALL test inputs
        n_tests = len(task["test"])
        if n_tests == 1:
            lines.append("## Test Input\n")
            test_input = task["test"][0]["input"]
            lines.append(f"Test Input ({len(test_input)}x{len(test_input[0])}):")
            lines.append(self._grid_to_string(test_input, indent=2))
        else:
            lines.append(f"## Test Inputs ({n_tests} tests)\n")
            for i, test_case in enumerate(task["test"]):
                test_input = test_case["input"]
                lines.append(f"### Test Input {i}")
                lines.append(f"({len(test_input)}x{len(test_input[0])}):")
                lines.append(self._grid_to_string(test_input, indent=2))
                lines.append("")
        
        lines.append("\nYour task: Discover the transformation rule, write a `transform(input_grid)` function that passes ALL training examples, then apply it to the test input(s).")
        
        return "\n".join(lines)
    
    @property
    def system_prompt(self) -> str:
        """The system prompt that guides human-like problem solving."""
        return """You are an expert at solving ARC-AGI puzzles, specifically designed as an iterative solver for ARC AGI 2 tasks. Your goal is to discover the transformation rule that converts inputs to outputs, applying it methodically to the test input while continuously refining your approach.

## ⚠️ MANDATORY: ALWAYS USE TOOL CALLS

**NEVER write code directly in your response text.** You MUST use the provided tools:
- **FIRST STEP (required)** → call `observe_examples(key_patterns=..., invariants=...)`
- To run/test code → call `run_code(code=..., purpose=...)`
- To verify solution → call `test_on_all_training(python_code=..., rule_description=...)`
- To apply to test → call `apply_to_test(python_code=..., final_rule_description=...)`
- To submit → call `confirm_submission(verified=..., verification_notes=...)`

If you write code in your message without using a tool, it will NOT be executed. The only way to execute code is through the `run_code` tool.

## ⚠️ STRUCTURED OBSERVATION (AUTO-ENFORCED)

Your first tool call will be automatically set to `observe_examples`. Use this to record:
- **Key patterns**: What you notice in the input/output pairs (e.g., "A divider column splits the grid")
- **Invariants**: Things that NEVER change (e.g., "Grid size unchanged", "Divider position unchanged")

These observations will guide your solution. Be thorough and specific!

## CRITICAL:
You must avoid getting stuck in local maxima by explicitly iterating through hypotheses, building on existing ideas, and, crucially, not abandoning promising directions at the first sign of failure. For every hypothesis iteration, analyze failures, extract what worked, and use it as the base for your next step. Do not revert to random search; always refine or extend from your last progress point.
When encountering a near-miss (close but not perfect result), investigate the small gap between success and failure. Ask: What minimal change would make the hypothesis succeed, and try simple modifications before exploring complex alternatives.
For seemingly simple transformations, always test the most basic plausible explanations before considering more elaborate ones. Avoid adding unnecessary complexity—start simple, and only build complexity if absolutely required by the data.
For harder tasks, Think hard and reason through each phase before using tools. 

## KEY TOOL: run_code
## Phased Problem-Solving with Tool Budgets
You have a LIMITED number of `run_code` calls.
It has available libraries: numpy, scipy, opencv, skimage, sklearn, collections, defaultdict, OrderedDict, math, re, itertools, functools, reduce, operator, statistics, copy, deepcopy.
Use them strategically across these phases:

### Phase 0: STRUCTURED OBSERVATION (AUTO-ENFORCED)
**Goal**: Document your initial observations BEFORE running any code.

This phase is **automatically enforced** - your first tool call will be `observe_examples`.

**Provide:**
- `key_patterns`: List of patterns you observe (use bullet points starting with •)
- `invariants`: List of things that MUST hold in solution

**Good example:**
```
key_patterns: [
  "• A single full-height gray (5) vertical divider column splits the grid",
  "• Black (0) pixels exist only on the left of the divider",
  "• Each connected black component is translated horizontally to touch the divider",
  "• Red (2) appears only on the right side as full-width horizontal bars"
]
invariants: [
  "Grid size unchanged",
  "Divider column (color 5) unchanged and full-height",
  "Black pixels remain black; no recoloring"
]
```

---

### Phase 1: DETAILED ANALYSIS (1 tool call)
**Goal**: Understand what you're looking at before theorizing. Do ALL analysis in ONE call.

**Single comprehensive run_code call:**
```python
from scipy.ndimage import label as ndimage_label

def extract_object_tree(grid, bg_color=0):
    # Extract objects and their relationships
    grid = np.array(grid)
    objects = []
    for color in sorted(set(grid.flat)):
        if color == bg_color:
            continue
        mask = (grid == color).astype(int)
        labeled, n_objects = ndimage_label(mask)
        for obj_id in range(1, n_objects + 1):
            coords = np.argwhere(labeled == obj_id)
            bbox = (coords[:,0].min(), coords[:,1].min(), 
                    coords[:,0].max(), coords[:,1].max())
            objects.append({
                'color': color, 'size': len(coords), 'bbox': bbox,
                'center': (coords[:,0].mean(), coords[:,1].mean()),
                'shape': (bbox[2]-bbox[0]+1, bbox[3]-bbox[1]+1)
            })
    return objects

# ═══════════════════════════════════════════════════════════
# PART 1: VISUALIZE ALL GRIDS
# ═══════════════════════════════════════════════════════════
for i, ex in enumerate(task['train']):
    print(f"{'='*50}")
    print(f"EXAMPLE {i}")
    print(f"{'='*50}")
    inp, out = np.array(ex['input']), np.array(ex['output'])
    
    # Grid visualization (list of lists format)
    print(f"INPUT {inp.shape}:")
    for row in inp:
        print(f"  {list(row)}")
    print(f"\nOUTPUT {out.shape}:")
    for row in out:
        print(f"  {list(row)}")

# ═══════════════════════════════════════════════════════════
# PART 2: BASIC STATISTICS
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*50}")
print("STATISTICS SUMMARY")
print(f"{'='*50}")
for i, ex in enumerate(task['train']):
    inp, out = np.array(ex['input']), np.array(ex['output'])
    print(f"Ex {i}: {inp.shape} → {out.shape}")
    print(f"  Colors: {sorted(set(inp.flat))} → {sorted(set(out.flat))}")
    if inp.shape == out.shape:
        print(f"  Changed cells: {np.sum(inp != out)}")

# ═══════════════════════════════════════════════════════════
# PART 3: OBJECT TREE EXTRACTION
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*50}")
print("OBJECT ANALYSIS")
print(f"{'='*50}")
for i, ex in enumerate(task['train']):
    print(f"\nExample {i} Objects:")
    print("  INPUT:")
    for obj in extract_object_tree(ex['input']):
        print(f"    Color {obj['color']}: {obj['size']} cells, bbox={obj['bbox']}, shape={obj['shape']}")
    print("  OUTPUT:")
    for obj in extract_object_tree(ex['output']):
        print(f"    Color {obj['color']}: {obj['size']} cells, bbox={obj['bbox']}, shape={obj['shape']}")
```

**After running, identify:**
- **Parent-Child**: Which objects contain others? (bbox enclosure)
- **Spatial**: Adjacent, overlapping, aligned, symmetric?
- **Color Roles**: Background, boundary, fill, marker, noise?
- **Changes**: Which objects appear/disappear/move/transform?

**End of Phase 1**: Write a plain English description including:
- Grid-level: "Output is always smaller than input"
- Object-level: "Blue rectangles contain red pixels"
- Relationships: "Small objects move toward large objects"

DO NOT proceed to Phase 2 until you can describe BOTH the transformation AND the object structure.

---

### Phase 2: HYPOTHESIS FORMATION (1-2 tool calls)
**Goal**: Form a SPECIFIC, TESTABLE hypothesis based on Phase 1 observations.

**FIRST: Review the transformation taxonomy**
Before forming your hypothesis, consider which transformation TYPE applies:

| Category | Examples | Key Signal |
|----------|----------|------------|
| **Extraction** | Crop, filter, select objects | Output smaller than input |
| **Construction** | Build pattern, fill template | Output larger or new structure |
| **Movement** | Translate, gravity, collision | Same objects, different positions |
| **Transformation** | Rotate, reflect, scale | Same structure, different orientation/size |
| **Mapping** | Color swap, pattern replacement | Same positions, different values |
| **Composition** | Overlay, merge, tile | Multiple inputs combined |
| **Conditional** | If-then rules, case handling | Different examples follow different sub-rules |

**SECOND: Plan your testing strategy**
Based on the transformation type, decide:
1. What specific measurements will CONFIRM your hypothesis?
2. What pattern in the data would DISPROVE it?
3. What edge cases should you check?

**Allowed activities**:
- Compute specific measurements to refine your hypothesis
- Analyze relationships between input features and output features
- Compare examples to identify what varies vs. stays constant

**Required output**: A hypothesis in this format:
```
TRANSFORMATION TYPE: [From taxonomy above]
HYPOTHESIS: [Specific transformation rule]
PREDICTS: [What this would produce for each training example]
WOULD FAIL IF: [What would disprove this]
TESTING PLAN: [2-3 specific tests to run]
```

**Example good hypothesis:**
```
TRANSFORMATION TYPE: Extraction + Mapping
HYPOTHESIS: Output is the bounding box of all foreground pixels, 
            downsampled by taking every 2nd row and column.
PREDICTS: Ex0 input (20,25) with fg bbox (18,20) → output (9,10)
WOULD FAIL IF: Output dimensions don't equal ceil(bbox/2)
TESTING PLAN: 
  1. Verify bbox dimensions match output * 2
  2. Check if specific pixels map correctly
  3. Test edge case where bbox is odd-sized
```

**Example bad hypothesis:**
```
HYPOTHESIS: Maybe it's some kind of pooling or sampling.
```
(Too vague — no transformation type, no predictions, no testing plan)

---

### Phase 3: TESTING (3-6 tool calls)
**Goal**: Verify or refine your hypothesis with targeted tests.

**Rules**:
1. Test your SPECIFIC hypothesis, not random alternatives
2. After each test, analyze the result before the next test
3. If hypothesis fails, identify WHAT failed and refine (don't abandon)

**Structure each test**:
```python
# Test: [What specific prediction am I checking?]
# Expected: [What should happen if hypothesis is correct?]

<your test code here>

# Actual: [What happened?]
# Conclusion: [Does this support, refine, or refute the hypothesis?]
```

**If test fails**:
- Compute the DIFFERENCE between prediction and actual
- Ask: What do the errors have in common?
- Refine hypothesis to account for the discrepancy

**Good iteration example**:
```
Test 1: Check if output = downsampled bbox by factor 2
Result: Ex0 predicted (9,10), actual (11,12) — CLOSE but not exact
Analysis: Off by 2 in each dimension. Maybe there's a border?

Test 2: Check if output = downsampled bbox + 1-cell border
Result: MATCH for Ex0, but Ex1 fails
Analysis: What's different about Ex1? → investigate further
```

---

### Phase 4: VERIFICATION (2-3 tool calls)
**Goal**: Confirm your rule works on ALL training examples.

**Required**:
1. Implement the full transformation as a function
2. Run on ALL training examples
3. Compare output to expected pixel-by-pixel

```python
def transform(input_grid):
    # Your complete transformation logic
    grid = np.array(input_grid)
    # ... implementation ...
    return output_grid.tolist()

# Verify on all training
for i, ex in enumerate(task['train']):
    predicted = transform(ex['input'])
    expected = ex['output']
    match = np.array_equal(np.array(predicted), np.array(expected))
    print(f"Example {i}: {'✓ MATCH' if match else '✗ FAIL'}")
    if not match:
        pred, exp = np.array(predicted), np.array(expected)
        diff_count = np.sum(pred != exp)
        print(f"  Differences: {diff_count} cells")
```

If ANY example fails → return to Phase 3 with specific error analysis. 
**DO NOT abandon working approaches.** Build on your best attempt. 

---

### Phase 5: APPLICATION & SUBMISSION (2-3 tool calls)  
**Goal**: Apply verified transformation to test input and carefully confirm before submitting.

**Step 1: Analyze Test Input (run_code)**
Before applying, verify the test input matches your expectations:
```python
test_inp = np.array(task['test'][0]['input'])
print(f"Test input shape: {test_inp.shape}")
print(f"Test input colors: {sorted(set(test_inp.flat))}")
print("Test input:")
for row in test_inp:
    print(f"  {list(row)}")

# Check: Does this match training patterns?
# - Same color palette?
# - Similar object structure?
# - Expected size range?
```

**Step 2: Apply Transformation (apply_to_test)**
Call `apply_to_test` with your verified code and rule description:
- `python_code`: Your complete `transform(input_grid)` function
- `final_rule_description`: Clear English description of what the code does

**Step 3: CRITICAL - Review Output Before Confirming**
You will see a preview of your generated output. CAREFULLY CHECK:

✅ **Pre-Confirmation Checklist:**
| Check | What to Verify |
|-------|----------------|
| **Dimensions** | Does output shape match expected pattern from training? |
| **Colors** | Are all output colors valid (appeared in training outputs)? |
| **Structure** | Does the output "look like" training outputs? |
| **Edge Cases** | If test is larger/smaller/different, did code handle it? |
| **No Artifacts** | No unexpected patterns, noise, or boundary issues? |


**Step 4: Confirm or Reject (confirm_submission)**
- If output looks correct: `confirm_submission(verified=true, verification_notes="...")`
- If output looks wrong: `confirm_submission(verified=false, verification_notes="...")`

**If you REJECT**, explain what's wrong and debug:
- What specific issue did you notice?
- What part of your code might have caused it?
- Go back to run_code to investigate and fix

---

## Tool Call Budget Summary

| Phase | Calls | Purpose |
|-------|-------|---------|
| 0. Structured Obs | **1** | `observe_examples` - MANDATORY FIRST |
| 1. Detailed Analysis | **1** | Visualize + stats + objects (all-in-one) |
| 2. Hypothesis | 1-2 | Classify type, form testable theory |
| 3. Testing | 3-6 | Verify and refine |
| 4. Verification | 2-3 | Confirm on all examples |
| 5. Application | 2-3 | Apply, review, confirm/reject |
| **Total** | **10-16** | |

If you reach 15 tool calls without convergence, PAUSE and reassess:
- Have you actually looked at the grids?
- Can you describe the transformation in plain English?
- Are you testing specific hypotheses or randomly searching?
- What's your best hypothesis so far? Build on it instead of starting over.

---

## Anti-Patterns to Avoid

❌ **Skipping Phase 1**: Jumping straight to testing pooling/sampling without looking at the grids

❌ **Brute-force search**: Testing every geometric primitive hoping one matches

❌ **Ignoring near-misses**: When a test is CLOSE, don't abandon it

❌ **Not synthesizing**: Running 50 tests without ever writing "here's what I've learned"

❌ **Vague hypotheses**: "Maybe it's related to components" is not testable

---

## Phase Transition Checklist

**Before leaving Phase 1**, you must have:
- [ ] Visualized all training input/output pairs
- [ ] Written a plain English description of what changes
- [ ] Identified key features (shapes, colors, objects)

**Before leaving Phase 2**, you must have:
- [ ] A specific, testable hypothesis
- [ ] Predictions for what each training output should be
- [ ] Criteria for what would disprove the hypothesis

**Before leaving Phase 3**, you must have:
- [ ] A hypothesis that explains ALL training examples
- [ ] Understanding of WHY the transformation works
- [ ] Ability to implement it as code

**Before leaving Phase 4**, you must have:
- [ ] Code that produces EXACT matches on ALL training examples
- [ ] Zero pixel differences

**Before leaving Phase 5**, you must have:
- [ ] Verified test input matches expected structure
- [ ] Sanity-checked the test output


### CRITICAL: Analyze Your Output Before Moving On

After every `run_code` result, STOP and ask:

1. **What pattern do I see in this output?**
   - Don't just print data — interpret it
   - Look for relationships between columns/rows
   - Spot regularities or near-regularities

2. **Does this suggest a refinement to my hypothesis?**
   - If flood fill overpredicts, what do the EXTRA cells have in common?
   - If it underpredicts, what do the MISSED cells have in common?

3. **Should my next iteration BUILD ON this, or is this a dead end?**
   - If the data shows a clear pattern, FOLLOW IT
   - Only abandon a direction if the data clearly contradicts it

4. Try to look for the most general rules and simplest explanation that would work for all examples. 



### Workflow Integration

1. **Start with `run_code`**: Analyze structure, find patterns, quantify changes
2. **Form hypothesis**: Based on computational analysis, not just visual inspection  
3. **Test with `run_code`**: Verify hypothesis holds across ALL training examples
4. **Build transform in `run_code`**: Develop incrementally with print statements
5. **Validate on test with `run_code`**: Check assumptions hold for test input
6. **Submit via `apply_to_test`**: Use your verified transformation

**Remember**: The Python environment is your laboratory. Every claim about the transformation should be backed by code that demonstrates it.

## CRITICAL: Avoid Overfitting

Before finalizing your rule:

1. **Does your rule reference ABSOLUTE positions (row 3, column 5)?**
   → Almost always WRONG. Use RELATIVE positions (adjacent to blue, edge of shape)

2. **Can you explain WHY each output cell has its value in terms of the INPUT?**
   → If not, you're memorizing, not generalizing

3. **Would your rule work if the input was shifted/rotated/scaled?**
   → If not, you're overfitting

4. **For each cell: "What in the INPUT causes this in the OUTPUT?"**

## CRITICAL: Handle Variation in Code, Not by Manipulating Inputs

Your transformation code must work directly on the input grid AS-IS. 

**NEVER preprocess/manipulate the input grid to fit your transformation logic.**

This includes:
- Rotating or flipping to a "canonical" orientation
- Shifting/translating objects to expected positions  
- Scaling/resizing to match a template size
- Recoloring to match expected color assignments
- Cropping regions before processing, then stitching back

These are SHORTCUTS that work on training but break on test inputs with natural variation.

**INSTEAD: Parameterize your transformation to handle variation directly.**

| Variation Type | ❌ Wrong Approach | ✅ Right Approach |
|----------------|-------------------|-------------------|
| Direction differs | Rotate grid → apply "down" logic → rotate back | Detect direction → use direction vectors (dr, dc) |
| Position differs | Shift object to origin → transform → shift back | Find object position → apply transform relative to it |
| Scale differs | Resize to fixed size → transform → resize back | Detect scale/bounds → apply transform with dynamic ranges |
| Colors differ | Map colors to expected → transform → map back | Detect color roles (background, boundary, fill) → use role-based logic |

**The Test**: If you removed all pre/post-processing, would your core logic still express the transformation correctly for ANY valid input? If not, the variation handling belongs IN the logic, not around it.

**Why This Matters**: Test inputs follow the same RULE as training, but may vary in orientation, position, scale, or color assignment. Preprocessing assumes the variation is noise to be normalized away — but it's actually part of what your code must handle.

---

## 🎯 SUMMARY: Expected Behavior

**You are an iterative, hypothesis-driven solver. Here's what success looks like:**

1. **OBSERVE FIRST** (1 call): Visualize grids, extract objects, understand structure before theorizing
2. **FORM SPECIFIC HYPOTHESIS**: Classify the transformation type, make testable predictions, plan your tests
3. **TEST & REFINE**: Run targeted tests. When close, debug the gap—don't abandon. Build on near-misses.
4. **VERIFY ON ALL TRAINING**: Code must pass 100% of training examples with zero pixel differences
5. **APPLY & CAREFULLY CONFIRM**: Review test output against checklist. Reject if anything looks wrong.

**The winning pattern:**
- Start simple, add complexity only when data demands it
- Each tool call should make measurable progress toward the solution
- When stuck, synthesize what you've learned before trying new directions
- Never confirm an output you haven't carefully reviewed

**You have up to 25 tool calls. Use them wisely—quality over quantity. If iterations run out, you'll be asked for a direct grid output.**"""    

    def _build_early_exit_result(self, task: dict, iterations: int, return_history: bool = False,
                                  training_pass_solution: dict = None, apply_to_test_solution: dict = None,
                                  messages: list = None) -> dict:
        """Build result dict when early stopping due to another run finding solution."""
        # Try to return best available solution
        attempt_1 = None
        attempt_2 = None
        
        if apply_to_test_solution:
            attempt_1 = apply_to_test_solution
        
        # If we have working_code, auto-apply it
        if self.working_code and not attempt_1:
            try:
                backup_outputs = []
                for test_case in task["test"]:
                    pred, err = self._execute_transform(self.working_code, test_case["input"])
                    if err:
                        backup_outputs = None
                        break
                    backup_outputs.append(pred)
                
                if backup_outputs:
                    backup_answer = backup_outputs[0] if len(backup_outputs) == 1 else backup_outputs
                    attempt_1 = {
                        "answer": backup_answer,
                        "rule": self.working_rule or "Early exit - training code auto-applied",
                        "source": "early_exit_training_code"
                    }
            except:
                pass
        
        result = {
            "answer": attempt_1["answer"] if attempt_1 else None,
            "rule": attempt_1["rule"] if attempt_1 else "Early exit - no solution",
            "adaptations": None,
            "iterations": iterations,
            "success": attempt_1 is not None,
            "tool_calls_made": len(self.iteration_history),
            "attempt_1": attempt_1,
            "attempt_2": attempt_2,
            "cost": self.total_cost,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "early_exit": True  # Flag to indicate early exit
        }
        
        if return_history:
            result["history"] = self.iteration_history
            result["messages"] = messages or []
        
        return result

    def solve(self, task: dict, return_history: bool = False) -> dict:
        """
        Solve an ARC-AGI task.
        
        Args:
            task: Dict with 'train' and 'test' keys
            return_history: If True, include full conversation history
            
        Returns:
            Dict with 'answer', 'rule', 'iterations', 'success', 'cost'
        """
        self.current_task = task
        self.iteration_history = []
        self.pending_submission = None
        self.training_verified = False
        self.working_code = None
        
        # Reset cost tracking for this solve call
        self.total_cost = 0.0
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.working_rule = None
        
        # Check for early stop signal at start
        if self.stop_event and self.stop_event.is_set():
            if self.verbose:
                print("🛑 Early stop signal received at start - another run found solution")
            return self._build_early_exit_result(task, 0, return_history)
        
        task_description = self._format_task_for_prompt(task)
        
        # =====================================================================
        # PHASE 1: Perception & Analysis (Code-based + LLM hypotheses)
        # =====================================================================
        if self.verbose:
            print(f"\n{'='*60}")
            print("PHASE 1: Perception & Hypothesis Formation")
            print('='*60)
        
        # Step 1A: Code-based task analysis (fast, no LLM cost)
        if self.verbose:
            print("  📊 Running code-based task analysis...")
        
        task_analysis = analyze_task(task, task_id="current")
        code_analysis_str = task_analysis.to_display()
        
        if self.verbose:
            print(f"     ✓ Analyzed {len(task_analysis.train_examples)} training examples")
            print(f"     ✓ Common patterns: {', '.join(task_analysis.common_hints) if task_analysis.common_hints else 'None detected'}")
        
        # Step 1B: LLM-based hypothesis generation
        if self.verbose:
            print(f"\n  🔮 Generating hypotheses with LLM...")
            print(f"     Using: {self.hypothesis_model} (effort={self.hypothesis_reasoning_effort or 'default'})")
        
        llm_hypotheses = perceive_task_llm(
            task, 
            self.client, 
            self.hypothesis_model, 
            self.hypothesis_reasoning_effort,
            verbose=self.verbose
        )
        
        # Combine code analysis and LLM hypotheses into initial analysis
        initial_analysis = f"""## Code-Based Analysis (Automated)

{code_analysis_str}

## LLM-Generated Hypotheses

{llm_hypotheses if llm_hypotheses else "No hypotheses generated - will discover through testing."}
"""
        
        if self.verbose:
            print(f"\n📝 Phase 1 Complete - Analysis ready for Phase 2")
        
        # =====================================================================
        # PHASE 2: Tool-based Testing & Refinement
        # =====================================================================
        if self.verbose:
            print(f"\n{'='*60}")
            print("PHASE 2: Testing & Refinement with Tools")
            print('='*60)
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Please solve this ARC-AGI puzzle:\n\n{task_description}"},
            {"role": "assistant", "content": f"Let me start by analyzing the examples:\n\n{initial_analysis}"},
            {"role": "user", "content": "Good analysis! Now you MUST call `observe_examples` to document your structured observations (key patterns, invariants) before proceeding."}
        ]
        
        # =====================================================================
        # PHASE 0: FORCED OBSERVATION STEP (MANDATORY)
        # =====================================================================
        if self.verbose:
            print(f"\n{'='*60}")
            print("PHASE 0: Structured Observation (MANDATORY)")
            print('='*60)
            print("  📝 Forcing observe_examples call...")
        
        # Force the observe_examples tool call
        observe_api_kwargs = {
            "model": self.model,
            "messages": messages,
            "tools": self.tools,
            "tool_choice": {"type": "function", "function": {"name": "observe_examples"}},
            "max_tokens": 120000
        }
        
        observe_api_kwargs["extra_body"] = {"reasoning": {"enabled": True}, **PROVIDER_CONFIG}
        
        try:
            observe_response = _call_with_retry(
                lambda: self.client.chat.completions.create(**observe_api_kwargs)
            )
            self._track_usage(observe_response)
            
            observe_message = observe_response.choices[0].message
            # Sanitize message to ensure content is not None (Anthropic requirement)
            msg = observe_message.model_dump()
            if msg.get("content") is None:
                msg["content"] = ""
            messages.append(msg)
            
            # Process the forced observe_examples call
            if observe_message.tool_calls:
                tool_call = observe_message.tool_calls[0]
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}
                
                result = self._handle_observe_examples(arguments)
                
                if self.verbose:
                    print(f"\n{result.get('output', 'Observations recorded')}")
                
                # Add the tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })
                
                # Add prompt to continue
                messages.append({
                    "role": "user",
                    "content": "Excellent observations! Now use the tools to test your hypotheses. Start with `run_code` to verify your observations programmatically, then use `apply_transformation` to test your rule."
                })
            else:
                if self.verbose:
                    print("  ⚠️ No observation tool call returned, continuing anyway...")
        
        except Exception as e:
            if self.verbose:
                print(f"  ⚠️ Observation step failed: {str(e)[:100]}, continuing anyway...")
        
        if self.verbose:
            print(f"\n📝 Phase 0 Complete - Observations recorded")
        
        final_answer = None
        final_rule = None
        final_adaptations = None
        iterations = 0
        solution_found = False
        total_tool_calls = 0  # Track total tool calls made
        
        # Track multiple solution candidates
        training_pass_solution = None  # Solution from test_on_all_training
        apply_to_test_solution = None  # Solution from apply_to_test
        
        while iterations < self.max_iterations and not solution_found:
            iterations += 1
            
            # Check for early stop signal from other run
            if self.stop_event and self.stop_event.is_set():
                if self.verbose:
                    print(f"\n🛑 Early stop signal received at iteration {iterations} - another run found solution")
                # Return best available result
                return self._build_early_exit_result(task, iterations, return_history, 
                                                     training_pass_solution, apply_to_test_solution, messages)
            
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Iteration {iterations}/{self.max_iterations}")
                print('='*60)
            
            api_kwargs = {
                "model": self.model,
                "messages": messages,
                "tools": self.tools,
                "tool_choice": "auto",
                "max_tokens": 120000
            }
            
            api_kwargs["extra_body"] = {"reasoning": {"enabled": True}, **PROVIDER_CONFIG}
            
            # Call with retry for transient JSON decode errors
            response = _call_with_retry(
                lambda: self.client.chat.completions.create(**api_kwargs)
            )
            self._track_usage(response)
            
            assistant_message = response.choices[0].message
            # Sanitize message to ensure content is not None (Anthropic requirement)
            msg = assistant_message.model_dump()
            if msg.get("content") is None:
                msg["content"] = ""
            messages.append(msg)
            
            if self.verbose and assistant_message.content:
                print(f"\nModel reasoning:\n{truncate(assistant_message.content)}")
            
            if not assistant_message.tool_calls:
                # Check if the last message was a checkpoint - text response is expected
                last_user_msg = next((m for m in reversed(messages[:-1]) if m.get("role") == "user"), None)
                is_checkpoint_response = last_user_msg and "CHECKPOINT" in last_user_msg.get("content", "")
                
                if is_checkpoint_response:
                    # Checkpoint response - acknowledge and prompt for next action
                    if self.verbose:
                        print("\n📝 Checkpoint summary received. Continuing...")
                    messages.append({
                        "role": "user",
                        "content": "Good summary. Now continue with your next action - use tools to make progress."
                    })
                else:
                    # Unexpected text-only response - give context-aware prompt
                    content = assistant_message.content or ""
                    
                    # Detect if model output code but didn't use run_code tool
                    has_code_block = "```python" in content or "def transform" in content or "def solve" in content
                    
                    if self.verbose:
                        print("\n⚠️ Model responded with text instead of tool calls - redirecting...")
                    
                    if has_code_block:
                        # Model wrote code directly - tell it to use run_code
                        recovery_prompt = (
                            "⚠️ You wrote code in your response, but you need to use the `run_code` tool to execute it!\n\n"
                            "DO NOT write code directly in your response. Instead:\n"
                            "1. Call `run_code(code='your_code_here', purpose='what you want to test')`\n"
                            "2. The code will be executed and you'll see the output\n\n"
                            "Please make a tool call now."
                        )
                    elif self.training_verified:
                        # Training passed - should apply to test
                        recovery_prompt = (
                            "Your training code has already passed all examples! Now you need to:\n"
                            "1. Call `apply_to_test(python_code=..., final_rule_description=...)` to apply your solution\n"
                            "2. Then call `confirm_submission(verified=true/false, ...)` to finalize\n\n"
                            "Please make a tool call now."
                        )
                    elif total_tool_calls == 0:
                        # Haven't started - should observe first
                        recovery_prompt = (
                            "You haven't made any tool calls yet. Start by exploring the data:\n"
                            "1. Call `run_code(code='...', purpose='...')` to analyze the grids programmatically\n"
                            "2. Look for patterns: colors, shapes, positions, transformations\n\n"
                            "Please make a tool call now."
                        )
                    else:
                        # In the middle - generic but directive
                        recovery_prompt = (
                            f"You've used {total_tool_calls} tool calls so far. Please continue with one of:\n"
                            "- `run_code(...)` to test hypotheses or analyze patterns\n"
                            "- `test_on_all_training(...)` to verify your transformation works on ALL training examples\n"
                            "- `apply_to_test(...)` after training passes, to generate the final answer\n\n"
                            "Please make a tool call now - do NOT write code directly in your response."
                        )
                    
                    messages.append({
                        "role": "user", 
                        "content": recovery_prompt
                    })
                continue
            
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}
                
                if self.verbose:
                    print(f"\n→ Tool call: {tool_name}")
                    if tool_name == "run_code":
                        purpose = arguments.get('purpose', 'analysis')
                        print(f"  Purpose: {truncate(purpose)}")
                        print(f"  Code: {truncate(arguments.get('code', ''))}")
                    elif tool_name == "apply_transformation":
                        print(f"  Rule: {truncate(arguments.get('rule_description', 'N/A'))}")
                        print(f"  Example: {arguments.get('example_index', 'N/A')}")
                    elif tool_name == "analyze_differences":
                        print(f"  Comparing: Example {arguments.get('example_a', 'N/A')} vs Example {arguments.get('example_b', 'N/A')}")
                    elif tool_name == "apply_to_test":
                        print(f"  Rule: {truncate(arguments.get('final_rule_description', 'N/A'))}")
                    elif tool_name == "submit_answer":
                        print(f"  Rule: {truncate(arguments.get('final_rule_description', 'N/A'))}")
                
                result = self._process_tool_call(tool_name, arguments)
                status = result.get("status", "unknown")
                
                # Only count successful tool calls (errors don't count toward limit)
                if status not in ["error", "execution_error"]:
                    total_tool_calls += 1
                    
                    # Add synthesis checkpoint every 5 tool calls
                    if total_tool_calls > 0 and total_tool_calls % 5 == 0:
                        messages.append({
                            "role": "user",
                            "content": (
                                "CHECKPOINT: Summarize what you've learned so far. "
                                "What hypotheses have been ruled out? "
                                "What's your current best theory?"
                            )
                        })
                        if self.verbose:
                            print(f"\n  📍 CHECKPOINT ({total_tool_calls} tool calls) - requesting synthesis")
                
                if self.verbose:
                    print(f"← Result: {status}")
                    if status in ["error", "execution_error"]:
                        # Print error details
                        error_msg = result.get("error", "Unknown error")
                        print(f"  ❌ Error: {truncate(error_msg)}")
                        if result.get("suggestion"):
                            print(f"  💡 Suggestion: {truncate(result.get('suggestion'))}")
                        print(f"  (This error does NOT count toward your tool call limit)")
                    elif tool_name == "run_code" and result.get("output"):
                        print(f"  Output: {truncate(result['output'])}")
                    elif status == "preview":
                        # Show preview from apply_to_test
                        outputs = result.get("outputs", [])
                        print(f"  📋 Preview: {len(outputs)} test(s) generated")
                    elif status == "incorrect":
                        print(f"  Accuracy: {result.get('accuracy', 'N/A')}")
                        print(f"  Hint: {truncate(result.get('hint', 'N/A'))}")
                
                # When training passes - first test is already visible in prompt
                if tool_name == "test_on_all_training" and result.get("status") == "all_correct":
                    self.training_verified = True
                    # Store the working code
                    self.working_code = arguments.get("python_code", "")
                    self.working_rule = arguments.get("rule_description", "")
                    training_pass_solution = {
                        "code": self.working_code,
                        "rule": self.working_rule,
                        "source": "test_on_all_training"
                    }
                    if self.verbose:
                        print(f"  💾 Training PASSED - model can now apply to test input (already visible in prompt)")
                
                self.iteration_history.append({
                    "iteration": iterations,
                    "tool": tool_name,
                    "arguments": arguments,
                    "result": result
                })
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, indent=2)
                })
                
                # Capture solution preview when apply_to_test runs successfully
                if tool_name == "apply_to_test" and result.get("status") == "preview":
                    # Preview generated - pending_submission is already stored in handler
                    if self.verbose:
                        outputs = result.get("outputs", [])
                        print(f"  👀 Preview generated - awaiting confirm_submission")
                        for out in outputs:
                            print(f"     Test {out['test_index']}: {out['shape']}")
                
                # Handle pending verification - show grid to model for review
                if tool_name == "submit_answer" and result.get("status") == "pending_verification":
                    if self.verbose:
                        print(f"  📋 Answer pending verification:")
                        print(f"     Shape: {result.get('shape')}")
                        print(f"     Rule: {result.get('rule')}")
                        # Show a preview of the grid
                        grid_preview = result.get("grid_preview", "")
                        if grid_preview:
                            lines = grid_preview.strip().split('\n')[:5]
                            for line in lines:
                                print(f"     {line.strip()}")
                            if len(grid_preview.strip().split('\n')) > 5:
                                print(f"     ... (more rows)")
                        print(f"  → Model should call confirm_submission(verified=true) to finalize")
                
                # Handle confirmed submission
                if tool_name == "confirm_submission" and result.get("status") == "confirmed":
                    final_answer = result.get("answer")
                    final_rule = result.get("rule")
                    
                    apply_to_test_solution = {
                        "answer": final_answer,
                        "rule": final_rule,
                        "source": "apply_to_test"
                    }
                    
                    if self.verbose:
                        print(f"\n{'='*60}")
                        print("✅ SOLUTION CONFIRMED!")
                        print(f"Rule: {final_rule}")
                        print('='*60)
                    
                    solution_found = True
                    break
            
            # Add iteration status message (if not solved yet)
            if not solution_found:
                remaining = self.max_iterations - iterations
                if remaining <= 3:
                    urgency = "⚠️ RUNNING LOW ON ITERATIONS! "
                else:
                    urgency = ""
                
                status_msg = (
                    f"{urgency}[Status: Iteration {iterations}/{self.max_iterations}, "
                    f"Tool calls used: {total_tool_calls}] "
                )
                
                if remaining <= 3:
                    status_msg += "You must submit soon. If you have a working solution, use apply_to_test now!"
                else:
                    status_msg += "Tip: Group multiple analyses into single run_code calls to use iterations wisely."
                
                messages.append({
                    "role": "user",
                    "content": status_msg
                })
        
        # =====================================================================
        # FINAL DIRECT GRID CALL (when iterations exhausted without solution)
        # =====================================================================
        direct_grid_solution = None
        
        if not solution_found:
            if self.verbose:
                print(f"\n{'='*60}")
                print("⚠️ ITERATIONS EXHAUSTED - REQUESTING DIRECT GRID OUTPUT")
                print('='*60)
            
            # Handle ALL test inputs
            n_tests = len(task["test"])
            
            # Format ALL test inputs for the final call
            test_inputs_str = ""
            for i, test_case in enumerate(task["test"]):
                test_input = test_case["input"]
                if n_tests > 1:
                    test_inputs_str += f"\nTEST INPUT {i+1}/{n_tests} ({len(test_input)}x{len(test_input[0])}):\n"
                else:
                    test_inputs_str += f"\nTEST INPUT ({len(test_input)}x{len(test_input[0])}):\n"
                test_inputs_str += "\n".join(f"  {list(row)}" for row in test_input)
                test_inputs_str += "\n"
            
            # Different prompt for single vs multi-test
            if n_tests == 1:
                format_instruction = """FORMAT YOUR ANSWER EXACTLY LIKE THIS:
```
VERIFICATION:
- Expected dimensions: [your reasoning]
- Colors used: [list them]
- Pattern check: [verify it matches]

FINAL_GRID:
[[row1], [row2], ...]
```"""
            else:
                format_instruction = f"""FORMAT YOUR ANSWER EXACTLY LIKE THIS (you have {n_tests} test inputs):
```
VERIFICATION:
- Expected dimensions: [your reasoning]
- Colors used: [list them]
- Pattern check: [verify it matches]

FINAL_GRID_1:
[[row1], [row2], ...]

FINAL_GRID_2:
[[row1], [row2], ...]
{f"... (continue for all {n_tests} grids)" if n_tests > 2 else ""}
```

IMPORTANT: You MUST provide {n_tests} separate output grids, one for each test input."""
            
            direct_grid_prompt = f"""You have exhausted your tool calls. Now you MUST provide your final answer directly.

TASK RECAP:
- You analyzed {len(task['train'])} training examples
- Number of test inputs: {n_tests}
{test_inputs_str}

INSTRUCTIONS:
1. Think through what you learned from the training examples
2. Apply the transformation rule to {"each test input" if n_tests > 1 else "the test input"}
3. Write out the COMPLETE output grid(s) as Python list of lists
4. VERIFY your answer:
   - Does it have the correct dimensions?
   - Does it use only colors seen in training outputs?
   - Does it follow the pattern you observed?

{format_instruction}

Be precise. Double-check before submitting. Your answer must be valid Python list(s) of lists."""

            messages.append({
                "role": "user",
                "content": direct_grid_prompt
            })
            
            # Make the final call WITHOUT tools
            final_api_kwargs = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 120000
            }
            
            final_api_kwargs["extra_body"] = {"reasoning": {"enabled": True}, **PROVIDER_CONFIG}
            
            try:
                final_response = _call_with_retry(
                    lambda: self.client.chat.completions.create(**final_api_kwargs)
                )
                self._track_usage(final_response)
                
                final_content = final_response.choices[0].message.content or ""
                
                if self.verbose:
                    print(f"\nModel response:\n{truncate(final_content, 1000)}")
                
                # Try to extract the grid(s) from the response
                import re
                
                parsed_grids = []
                
                if n_tests == 1:
                    # Single test - look for FINAL_GRID: followed by a list
                    grid_match = re.search(r'FINAL_GRID:\s*(\[\[.*?\]\])', final_content, re.DOTALL)
                    
                    if not grid_match:
                        # Try to find any list of lists pattern
                        grid_match = re.search(r'(\[\s*\[[\d,\s\[\]]+\]\s*\])', final_content, re.DOTALL)
                    
                    if grid_match:
                        try:
                            grid_str = grid_match.group(1)
                            grid_str = grid_str.replace('\n', '').replace(' ', '')
                            parsed_grid = eval(grid_str)
                            if isinstance(parsed_grid, list) and all(isinstance(row, list) for row in parsed_grid):
                                parsed_grids.append(parsed_grid)
                        except Exception as e:
                            if self.verbose:
                                print(f"\n✗ Failed to parse grid: {e}")
                else:
                    # Multiple tests - look for FINAL_GRID_N: patterns
                    for i in range(1, n_tests + 1):
                        grid_match = re.search(rf'FINAL_GRID_{i}:\s*(\[\[.*?\]\])', final_content, re.DOTALL)
                        
                        if grid_match:
                            try:
                                grid_str = grid_match.group(1)
                                grid_str = grid_str.replace('\n', '').replace(' ', '')
                                parsed_grid = eval(grid_str)
                                if isinstance(parsed_grid, list) and all(isinstance(row, list) for row in parsed_grid):
                                    parsed_grids.append(parsed_grid)
                            except Exception as e:
                                if self.verbose:
                                    print(f"\n✗ Failed to parse grid {i}: {e}")
                    
                    # Fallback: if we didn't find numbered grids, try to find all list-of-lists
                    if len(parsed_grids) < n_tests:
                        all_grid_matches = re.findall(r'(\[\s*\[[\d,\s]+\](?:\s*,\s*\[[\d,\s]+\])*\s*\])', final_content)
                        for grid_str in all_grid_matches:
                            if len(parsed_grids) >= n_tests:
                                break
                            try:
                                grid_str = grid_str.replace('\n', '').replace(' ', '')
                                parsed_grid = eval(grid_str)
                                if isinstance(parsed_grid, list) and all(isinstance(row, list) for row in parsed_grid):
                                    # Avoid duplicates
                                    if parsed_grid not in parsed_grids:
                                        parsed_grids.append(parsed_grid)
                            except:
                                pass
                
                # Build the answer
                if parsed_grids:
                    if n_tests == 1 and len(parsed_grids) == 1:
                        final_answer = parsed_grids[0]
                    else:
                        final_answer = parsed_grids
                    
                    direct_grid_solution = {
                        "answer": final_answer,
                        "rule": "Direct grid output after iteration exhaustion",
                        "source": "direct_grid_final_call"
                    }
                    if self.verbose:
                        if n_tests == 1:
                            print(f"\n✓ Parsed direct grid: {len(parsed_grids[0])}x{len(parsed_grids[0][0]) if parsed_grids[0] else 0}")
                        else:
                            print(f"\n✓ Parsed {len(parsed_grids)}/{n_tests} direct grids")
                else:
                    if self.verbose:
                        print("\n✗ No grid found in response")
                        
            except Exception as e:
                if self.verbose:
                    print(f"\n✗ Final call failed: {e}")
        
        # Build attempt_1 and attempt_2
        # attempt_1 = what the model confirmed via apply_to_test
        # attempt_2 = backup from training_pass code, OR direct grid fallback
        attempt_1 = None
        attempt_2 = None
        
        # attempt_1: Use confirmed solution
        if apply_to_test_solution:
            attempt_1 = apply_to_test_solution
        
        # attempt_2: Priority order for backup:
        # 1. Training-verified code auto-applied
        # 2. Direct grid from final call
        backup_solution = None
        
        # Try training code first
        if self.working_code and training_pass_solution:
            backup_outputs = []
            for test_case in task["test"]:
                pred, err = self._execute_transform(self.working_code, test_case["input"])
                if err:
                    backup_outputs = None
                    break
                backup_outputs.append(pred)
            
            if backup_outputs:
                backup_answer = backup_outputs[0] if len(backup_outputs) == 1 else backup_outputs
                backup_solution = {
                    "answer": backup_answer,
                    "rule": training_pass_solution.get("rule", ""),
                    "source": "training_code_auto_applied"
                }
        
        # If no training code backup, use direct grid
        if not backup_solution and direct_grid_solution:
            backup_solution = direct_grid_solution
        
        # Assign attempt_2 if different from attempt_1
        if backup_solution:
            if attempt_1 is None or backup_solution.get("answer") != attempt_1.get("answer"):
                attempt_2 = backup_solution
        
        # If we only have attempt_2 but no attempt_1, swap them
        if not attempt_1 and attempt_2:
            attempt_1 = attempt_2
            attempt_2 = None
        
        # If still no attempt_1 but we have direct_grid, use it
        if not attempt_1 and direct_grid_solution:
            attempt_1 = direct_grid_solution
        
        result = {
            "answer": attempt_1["answer"] if attempt_1 else None,
            "rule": attempt_1["rule"] if attempt_1 else final_rule,
            "adaptations": attempt_1.get("adaptations") if attempt_1 else final_adaptations,
            "iterations": iterations,
            "success": attempt_1 is not None,
            "tool_calls_made": len(self.iteration_history),
            # Both attempts for submission
            "attempt_1": attempt_1,
            "attempt_2": attempt_2,
            # Cost tracking
            "cost": self.total_cost,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }
        
        if return_history:
            result["history"] = self.iteration_history
            result["messages"] = messages
        
        return result
    
    def evaluate(self, task: dict, expected_output: list[list[int]] = None) -> dict:
        """Solve a task and evaluate against expected output if provided."""
        result = self.solve(task, return_history=True)
        
        if expected_output and result["answer"]:
            result["correct"] = result["answer"] == expected_output
            if not result["correct"]:
                mismatches = self._find_mismatches(
                    result["answer"], 
                    expected_output, 
                    task["test"][0]["input"]
                )
                result["mismatches"] = len(mismatches)
        
        return result


def solve_task(
    task: dict,
    api_key: str,
    model: str = "anthropic/claude-opus-4.6",
    verbose: bool = True
) -> dict:
    """Quick function to solve an ARC task."""
    solver = ARCSolver(api_key=api_key, model=model, verbose=verbose)
    return solver.solve(task)
