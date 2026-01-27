# Prompt Module Documentation

The `prompt.py` module generates rich, decision-focused prompts for ARC puzzle solving. It integrates outputs from multiple analysis components to help the LLM make informed transformation decisions.

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     PROMPT GENERATION FLOW                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐        │
│  │   Analyzer   │   │  Perceiver   │   │   Objects    │        │
│  │              │   │              │   │              │        │
│  │ GridAnalysis │   │  Hypotheses  │   │ EnhancedObj  │        │
│  │ TransformA.  │   │ Observations │   │ Relationships│        │
│  │ TaskAnalysis │   │  KeyInsight  │   │ Patterns     │        │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘        │
│         │                  │                  │                 │
│         └──────────────────┼──────────────────┘                 │
│                            ▼                                    │
│                   ┌─────────────────┐                          │
│                   │  generate_prompt │                          │
│                   │                 │                          │
│                   │ • Quick Summary │                          │
│                   │ • Hypotheses    │                          │
│                   │ • Examples      │                          │
│                   │ • Test Inputs   │                          │
│                   └────────┬────────┘                          │
│                            ▼                                    │
│                   ┌─────────────────┐                          │
│                   │   User Prompt    │                          │
│                   │   (+ SOLVER_     │                          │
│                   │    SYSTEM)       │                          │
│                   └─────────────────┘                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. System Prompt (`SOLVER_SYSTEM`)

The system prompt (~24K chars) provides the LLM with:

- **Hard Constraints**: Grid specs (0-9 colors, 1-30 dimensions), output format
- **Cognitive Workflow**: 4-phase approach (Observe → Hypothesize → Verify → Implement)
- **Key Heuristics**: Size patterns, color mapping, object roles
- **Pattern Taxonomy**: 17 categories (A-Q) of transformation patterns
- **NumPy Reference**: Common operations and idioms
- **Type Safety**: Error prevention guidelines

### 2. User Prompt (`generate_prompt`)

The user prompt integrates multiple data sources into a structured, decision-focused format.

## Usage

### Basic Usage (No Perception Data)

```python
from src.solve.prompt import generate_prompt, SOLVER_SYSTEM

# Load task
task_data = {"train": [...], "test": [...]}

# Generate prompt
prompt = generate_prompt(
    task_data=task_data,
    include_analysis=True
)

# Use with LLM
response = await call_llm(
    system_prompt=SOLVER_SYSTEM,
    user_prompt=prompt
)
```

### Full Usage (With Perception Pipeline)

```python
from src.solve.prompt import generate_prompt

prompt = generate_prompt(
    task_data=task_data,
    perceptions=perceptions,        # Per-example object detection
    deltas=deltas,                  # Per-example transform deltas
    test_perception=test_perc,      # Test input perception
    hypotheses=hypotheses,          # Ranked transformation hypotheses
    observations=observations,      # Task-level observations
    key_insight=key_insight,        # The key insight string
    feedback=None,                  # For retry attempts
    include_analysis=True           # Include grid stats
)
```

## Input Data Formats

### Perceptions

```python
perceptions = [
    {
        "input": {
            "objects": [
                {"id": 1, "color": "red(2)", "shape": "rectangle", "size": 4, "position": "top-left", "special": ""},
            ],
            "relationships": [
                {"type": "adjacent", "obj1": 1, "obj2": 2, "direction": "right"},
                {"type": "contained", "outer": 1, "inner": 2},
                {"type": "aligned", "objects": [1, 2, 3], "axis": "horizontal"},
            ],
            "global_patterns": ["grid has horizontal symmetry"],
            "notable_features": ["single red pixel at center"],
        },
        "output": {
            "objects": [...],
        }
    },
    # ... one per training example
]
```

### Hypotheses

```python
hypotheses = [
    {
        "rank": 1,
        "confidence": "HIGH",  # HIGH, MEDIUM, or LOW
        "rule": "Clear description of the transformation rule",
        "evidence": "How this explains all training examples"
    },
    # ... up to 5 hypotheses, ranked
]
```

### Observations

```python
observations = {
    "common_input_features": ["sparse pixels", "black background"],
    "common_output_features": ["patterns added", "colors preserved"],
    "size_pattern": "same size",  # or "grows", "shrinks", "varies"
    "color_changes": "new colors added"  # or "none", "recoloring"
}
```

### Deltas

```python
deltas = [
    {
        "summary": "Size preserved, colors added",
        "object_changes": ["Object 1 moved right", "New object created"],
        "color_changes": [{"type": "added", "colors": ["yellow(4)"]}],
        "structural_changes": ["symmetry created"],
        "constants": ["grid_size_preserved", "color_palette_preserved"]
    },
    # ... one per training example
]
```

## Output Format

The generated prompt follows this structure:

```
╔══════════════════════════════════════════════════════════╗
║          ARC PUZZLE - TRANSFORMATION TASK                ║
╚══════════════════════════════════════════════════════════╝

┌─ QUICK SUMMARY ─────────────────────────────────────────┐
│  Training Examples: N
│  Test Cases: M
│  Likely Patterns: ...
│  Common Transforms: ...
│  Size Changes: Ex1: (H,W)→(H',W') | ...
└─────────────────────────────────────────────────────────┘

╔══════════════════════════════════════════════════════════╗
║  🧠 TRANSFORMATION HYPOTHESES (Use These for Decision)      ║
╚══════════════════════════════════════════════════════════╝

💡 KEY INSIGHT: ...

  🟢 #1 [HIGH]
     Rule: ...
     Evidence: ...

  🟡 #2 [MEDIUM]
     Rule: ...
     Evidence: ...

============================================================
TRAINING EXAMPLES
============================================================

────────────────────────────────────────────────────────────
EXAMPLE 1 of N
────────────────────────────────────────────────────────────

┌─ INPUT (H×W) ────────────────────────────────┐
[grid visualization]

  📊 Stats: X colors | Y shapes | Z% filled
  🎨 Palette: ...
  🔲 Background: ...

  🔍 DETECTED OBJECTS (N):
  #1: color shape (size=S) @ position

  🔗 RELATIONSHIPS:
  • Object 1 ↔ Object 2 (type, direction)

┌─ OUTPUT (H×W) ───────────────────────────────┐
[grid visualization]

  ⚡ TRANSFORMATION:
     Size: SAME / changed
     Colors Added/Removed: ...
     Shapes: X → Y

============================================================
🔎 CROSS-EXAMPLE PATTERNS (Decision Factors)
============================================================

  INVARIANTS (must hold in your solution):
    ✓ Size always preserved
    ✓ Colors always preserved

  COMMON HINTS: ...
  INFERRED PATTERN CATEGORIES: ...

============================================================
🎯 TEST INPUT - APPLY YOUR RULE HERE
============================================================

[test grid with analysis]

════════════════════════════════════════════════════════════════
YOUR TASK
════════════════════════════════════════════════════════════════

1. **STUDY** the training examples above
2. **IDENTIFY** the SINGLE rule that transforms ALL inputs to outputs
3. **VERIFY** your rule mentally on each example before coding
4. **IMPLEMENT** a `transform(grid)` function

...
```

## Additional Functions

### `generate_feedback_prompt`

For retry attempts after failures:

```python
from src.solve.prompt import generate_feedback_prompt

feedback_prompt = generate_feedback_prompt(
    original_prompt=prompt,
    code=failed_code,
    feedback_messages=["Shape mismatch: expected (3,3), got (9,9)"],
    attempt_num=2,
    expected_outputs=[expected_arr],
    actual_outputs=[actual_arr]
)
```

### `generate_minimal_prompt`

Lightweight prompt without perception data:

```python
from src.solve.prompt import generate_minimal_prompt

minimal = generate_minimal_prompt(task_data)
```

### `generate_hypothesis_verification_prompt`

For iterative hypothesis refinement:

```python
from src.solve.prompt import generate_hypothesis_verification_prompt

verify_prompt = generate_hypothesis_verification_prompt(
    task_data=task_data,
    hypothesis="Objects move right by 2 cells",
    code=code,
    results=[{"correct": True}, {"correct": False, "diff_count": 4}]
)
```

## Design Principles

1. **Decision-First**: Hypotheses and key insights appear at the top before examples
2. **Invariants Highlighted**: Cross-example patterns that must hold are explicitly listed
3. **Compact Stats**: Grid analysis condensed to single-line summaries
4. **Visual Hierarchy**: Unicode boxes and emoji icons for scanability
5. **Verification Reminders**: Prompts to check hypotheses against all examples

## Integration Example

```python
import asyncio
from src.solve.prompt import generate_prompt, SOLVER_SYSTEM
from src.perception.perceiver import perceive_task
from src.llms.client import call_llm

async def solve_with_perception(task_data):
    # Get perceptions and hypotheses
    task_perception = await perceive_task(task_data)
    
    # Generate prompt
    prompt = generate_prompt(
        task_data=task_data,
        hypotheses=task_perception.get("transformation_hypotheses"),
        observations=task_perception.get("observations"),
        key_insight=task_perception.get("key_insight"),
    )
    
    # Call solver
    response, _ = await call_llm(
        model="gpt-4",
        system_prompt=SOLVER_SYSTEM,
        user_prompt=prompt
    )
    
    return response

# Run
result = asyncio.run(solve_with_perception(task_data))
```

## Prompt Size Estimates

| Component | Approximate Size |
|-----------|------------------|
| System Prompt (`SOLVER_SYSTEM`) | ~24,000 chars |
| Minimal User Prompt | ~1,500 chars |
| User Prompt (no perception) | ~3,000-6,000 chars |
| User Prompt (with perception) | ~5,000-10,000 chars |
| Feedback Section | ~500-1,000 chars |

## Pattern Taxonomy Quick Reference

| Category | Description |
|----------|-------------|
| A. Geometric | Rotations, flips, scaling |
| B. Tiling | Repeat, mirror, stack |
| C. Cropping | Extract, crop, sample |
| D. Extension | Extend lines, grow, pad |
| E. Fill | Flood fill, enclosed, gradient |
| F. Color | Swap, replace, map, cycle |
| G. Object | Copy, move, align, sort |
| H. Containment | Bounding box, frame, outline |
| I. Masking | Apply mask, overlay, composite |
| J. Symmetry | Complete, restore, reflect |
| K. Pattern | Repair, inpaint, denoise |
| L. Counting | Count objects/colors/pixels |
| M. Sorting | Sort rows/cols, gravity |
| N. Conditional | If-then based on properties |
| O. Reference | Template, palette, stencil |
| P. Connectivity | Connect, path, components |
| Q. Projection | Horizontal/vertical collapse |

