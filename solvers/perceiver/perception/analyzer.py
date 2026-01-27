"""Enhanced grid analyzer - captures detailed insights for ARC tasks."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage

from solvers.perceiver.models import COLOR_NAMES, color_name


# =============================================================================
# Data Classes for Analysis Results
# =============================================================================

@dataclass
class GridAnalysis:
    """Comprehensive analysis of a single grid."""
    # Dimensions
    rows: int
    cols: int
    total_cells: int
    
    # Colors
    colors_used: int
    color_palette: list[str]
    background_color: str
    background_color_id: int
    
    # Density
    fill_ratio: float  # percentage of non-background cells
    non_background_cells: int
    
    # Shapes by color
    shapes_by_color: dict[str, int]
    total_shapes: int
    
    # Additional insights
    unique_colors: list[int]
    color_counts: dict[str, int]
    has_symmetry: dict[str, bool]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "grid_size": f"{self.rows} × {self.cols} ({self.total_cells} cells)",
            "colors_used": f"{self.colors_used} colors",
            "color_palette": self.color_palette,
            "background": self.background_color,
            "fill_ratio": f"{self.fill_ratio:.1f}%",
            "shapes_by_color": self.shapes_by_color,
            "total_shapes": self.total_shapes,
            "symmetry": self.has_symmetry,
        }
    
    def to_display(self) -> str:
        """Format for display in prompts."""
        lines = [
            f"Grid Size: {self.rows} × {self.cols} ({self.total_cells} cells)",
            f"Colors Used: {self.colors_used} colors",
            f"Color Palette: {', '.join(self.color_palette)}",
            f"Background: {self.background_color}",
            f"Fill Ratio: {self.fill_ratio:.1f}%",
            f"Shapes by Color:",
        ]
        for color, count in self.shapes_by_color.items():
            lines.append(f"  {color}: {count}")
        return "\n".join(lines)


@dataclass  
class TransformAnalysis:
    """Analysis of transformation between input and output."""
    # Size changes
    size_change_cells: int  # absolute change
    size_change_percent: float
    same_size: bool
    
    # Color changes
    new_colors_introduced: bool
    new_colors: list[str]
    removed_colors: list[str]
    colors_preserved: bool
    
    # Density changes
    density_change_percent: float
    same_density: bool
    
    # Shape changes
    input_shape_count: int
    output_shape_count: int
    shape_count_change: int
    same_shape_count: bool
    
    # Transform hints (human-readable)
    hints: list[str]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "size_change": f"{self.size_change_cells} cells ({self.size_change_percent:+.1f}%)",
            "new_colors_introduced": "YES" if self.new_colors_introduced else "NO",
            "new_colors": self.new_colors,
            "removed_colors": self.removed_colors,
            "density_change": f"{self.density_change_percent:+.1f}%",
            "shape_count": f"{self.input_shape_count} → {self.output_shape_count}",
            "transform_hints": self.hints,
        }
    
    def to_display(self) -> str:
        """Format for display in prompts."""
        lines = [
            "Key Transformations:",
            f"  Size Change: {self.size_change_cells} cells ({self.size_change_percent:+.1f}%)",
            f"  New Colors Introduced: {'YES' if self.new_colors_introduced else 'NO'}",
            f"  Density Change: {self.density_change_percent:+.1f}%",
            f"  Shape Count: {self.input_shape_count} shapes → {self.output_shape_count} shapes",
            "Transform Hints:",
        ]
        for hint in self.hints:
            lines.append(f"  • {hint}")
        return "\n".join(lines)


@dataclass
class ExampleAnalysis:
    """Complete analysis of an input/output training example."""
    example_num: int
    input_analysis: GridAnalysis
    output_analysis: GridAnalysis
    transform_analysis: TransformAnalysis
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "example": self.example_num,
            "input": self.input_analysis.to_dict(),
            "output": self.output_analysis.to_dict(),
            "transformation": self.transform_analysis.to_dict(),
        }
    
    def to_display(self) -> str:
        """Format for display in prompts."""
        return f"""
{'='*60}
Training Example #{self.example_num}
{'='*60}

--- INPUT ---
{self.input_analysis.to_display()}

--- OUTPUT ---
{self.output_analysis.to_display()}

--- TRANSFORMATION ---
{self.transform_analysis.to_display()}
"""


@dataclass
class TaskAnalysis:
    """Complete analysis of an ARC task."""
    task_id: str
    train_examples: list[ExampleAnalysis]
    test_inputs: list[GridAnalysis]
    
    # Cross-example patterns
    consistent_size_preservation: bool
    consistent_color_preservation: bool
    consistent_shape_count: bool
    common_hints: list[str]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "train_examples": [ex.to_dict() for ex in self.train_examples],
            "test_inputs": [t.to_dict() for t in self.test_inputs],
            "patterns": {
                "consistent_size_preservation": self.consistent_size_preservation,
                "consistent_color_preservation": self.consistent_color_preservation,
                "consistent_shape_count": self.consistent_shape_count,
                "common_hints": self.common_hints,
            }
        }
    
    def to_display(self) -> str:
        """Format complete task analysis for display."""
        lines = [
            f"{'='*60}",
            f"Task Details",
            f"Task ID: {self.task_id}",
            f"{'='*60}",
        ]
        
        for ex in self.train_examples:
            lines.append(ex.to_display())
        
        lines.append(f"\n{'='*60}")
        lines.append("Test Input(s)")
        lines.append(f"{'='*60}")
        
        for i, test in enumerate(self.test_inputs, 1):
            lines.append(f"\n--- Test #{i} ---")
            lines.append(test.to_display())
        
        lines.append(f"\n{'='*60}")
        lines.append("Cross-Example Patterns")
        lines.append(f"{'='*60}")
        lines.append(f"  Size always preserved: {self.consistent_size_preservation}")
        lines.append(f"  Colors always preserved: {self.consistent_color_preservation}")
        lines.append(f"  Shape count preserved: {self.consistent_shape_count}")
        lines.append(f"  Common hints: {', '.join(self.common_hints)}")
        
        return "\n".join(lines)


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_grid(grid: np.ndarray) -> GridAnalysis:
    """Perform comprehensive analysis of a single grid."""
    grid = np.array(grid)
    rows, cols = grid.shape
    total_cells = rows * cols
    
    # Color analysis
    unique_colors = list(np.unique(grid))
    color_counts_raw = {int(c): int(np.sum(grid == c)) for c in unique_colors}
    
    # Background is most common color
    background_id = max(color_counts_raw, key=color_counts_raw.get)
    background_name = color_name(background_id)
    
    # Non-background stats
    non_bg_cells = total_cells - color_counts_raw[background_id]
    fill_ratio = (non_bg_cells / total_cells) * 100
    
    # Color palette (excluding background, sorted by count)
    color_palette = [color_name(c) for c in unique_colors]
    
    # Count shapes (connected components) by color
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
    
    # Symmetry detection
    has_symmetry = {
        "horizontal": bool(np.array_equal(grid, np.flip(grid, axis=1))),
        "vertical": bool(np.array_equal(grid, np.flip(grid, axis=0))),
        "diagonal": bool(rows == cols and np.array_equal(grid, grid.T)),
        "rotational_180": bool(np.array_equal(grid, np.rot90(grid, 2))),
    }
    
    # Color counts as strings
    color_counts = {color_name(int(k)): int(v) for k, v in color_counts_raw.items()}
    
    return GridAnalysis(
        rows=rows,
        cols=cols,
        total_cells=total_cells,
        colors_used=len(unique_colors),
        color_palette=color_palette,
        background_color=background_name,
        background_color_id=background_id,
        fill_ratio=fill_ratio,
        non_background_cells=non_bg_cells,
        shapes_by_color=shapes_by_color,
        total_shapes=total_shapes,
        unique_colors=[int(c) for c in unique_colors],
        color_counts=color_counts,
        has_symmetry=has_symmetry,
    )


def analyze_transform(
    input_analysis: GridAnalysis,
    output_analysis: GridAnalysis,
) -> TransformAnalysis:
    """Analyze the transformation between input and output."""
    
    # Size changes
    size_change_cells = output_analysis.total_cells - input_analysis.total_cells
    size_change_percent = (size_change_cells / input_analysis.total_cells) * 100 if input_analysis.total_cells > 0 else 0
    same_size = input_analysis.total_cells == output_analysis.total_cells
    
    # Color changes
    input_colors = set(input_analysis.unique_colors)
    output_colors = set(output_analysis.unique_colors)
    new_colors = [color_name(c) for c in (output_colors - input_colors)]
    removed_colors = [color_name(c) for c in (input_colors - output_colors)]
    new_colors_introduced = len(new_colors) > 0
    colors_preserved = input_colors == output_colors
    
    # Density changes
    density_change = output_analysis.fill_ratio - input_analysis.fill_ratio
    same_density = abs(density_change) < 0.5  # within 0.5%
    
    # Shape count changes
    shape_change = output_analysis.total_shapes - input_analysis.total_shapes
    same_shape_count = input_analysis.total_shapes == output_analysis.total_shapes
    
    # Generate hints
    hints = []
    if same_size:
        hints.append("same Size")
    elif size_change_cells > 0:
        hints.append(f"size Increase ({size_change_percent:+.1f}%)")
    else:
        hints.append(f"size Decrease ({size_change_percent:+.1f}%)")
    
    if colors_preserved:
        hints.append("colors Preserved")
    if new_colors_introduced:
        hints.append(f"new Colors: {', '.join(new_colors)}")
    if removed_colors:
        hints.append(f"colors Removed: {', '.join(removed_colors)}")
    
    if same_density:
        hints.append("density Preserved")
    
    if same_shape_count:
        hints.append("shape Count Preserved")
    elif shape_change > 0:
        hints.append(f"shapes Added (+{shape_change})")
    else:
        hints.append(f"shapes Removed ({shape_change})")
    
    # Symmetry hints
    for sym_type, has_sym in output_analysis.has_symmetry.items():
        if has_sym and not input_analysis.has_symmetry.get(sym_type, False):
            hints.append(f"{sym_type} Symmetry Created")
    
    return TransformAnalysis(
        size_change_cells=size_change_cells,
        size_change_percent=size_change_percent,
        same_size=same_size,
        new_colors_introduced=new_colors_introduced,
        new_colors=new_colors,
        removed_colors=removed_colors,
        colors_preserved=colors_preserved,
        density_change_percent=density_change,
        same_density=same_density,
        input_shape_count=input_analysis.total_shapes,
        output_shape_count=output_analysis.total_shapes,
        shape_count_change=shape_change,
        same_shape_count=same_shape_count,
        hints=hints,
    )


def analyze_example(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    example_num: int = 1,
) -> ExampleAnalysis:
    """Analyze a single training example."""
    input_analysis = analyze_grid(input_grid)
    output_analysis = analyze_grid(output_grid)
    transform_analysis = analyze_transform(input_analysis, output_analysis)
    
    return ExampleAnalysis(
        example_num=example_num,
        input_analysis=input_analysis,
        output_analysis=output_analysis,
        transform_analysis=transform_analysis,
    )


def analyze_task(
    task_data: dict[str, Any],
    task_id: str = "unknown",
) -> TaskAnalysis:
    """Perform comprehensive analysis of an entire ARC task."""
    
    # Analyze training examples
    train_analyses = []
    for i, pair in enumerate(task_data['train'], 1):
        input_grid = np.array(pair['input'])
        output_grid = np.array(pair['output'])
        train_analyses.append(analyze_example(input_grid, output_grid, i))
    
    # Analyze test inputs
    test_analyses = []
    for test in task_data['test']:
        test_input = np.array(test['input'])
        test_analyses.append(analyze_grid(test_input))
    
    # Find cross-example patterns
    if train_analyses:
        consistent_size = all(ex.transform_analysis.same_size for ex in train_analyses)
        consistent_color = all(ex.transform_analysis.colors_preserved for ex in train_analyses)
        consistent_shapes = all(ex.transform_analysis.same_shape_count for ex in train_analyses)
        
        # Find common hints across all examples
        hint_sets = [set(ex.transform_analysis.hints) for ex in train_analyses]
        common_hints = list(set.intersection(*hint_sets)) if hint_sets else []
    else:
        consistent_size = False
        consistent_color = False
        consistent_shapes = False
        common_hints = []
    
    return TaskAnalysis(
        task_id=task_id,
        train_examples=train_analyses,
        test_inputs=test_analyses,
        consistent_size_preservation=consistent_size,
        consistent_color_preservation=consistent_color,
        consistent_shape_count=consistent_shapes,
        common_hints=common_hints,
    )


# =============================================================================
# Quick Analysis Functions
# =============================================================================

def quick_grid_stats(grid: np.ndarray) -> str:
    """Get a quick one-line summary of grid stats."""
    analysis = analyze_grid(grid)
    return (
        f"{analysis.rows}×{analysis.cols} | "
        f"{analysis.colors_used} colors | "
        f"{analysis.total_shapes} shapes | "
        f"{analysis.fill_ratio:.1f}% fill"
    )


def quick_transform_stats(input_grid: np.ndarray, output_grid: np.ndarray) -> str:
    """Get a quick summary of transformation."""
    example = analyze_example(input_grid, output_grid)
    t = example.transform_analysis
    return " | ".join(t.hints)

