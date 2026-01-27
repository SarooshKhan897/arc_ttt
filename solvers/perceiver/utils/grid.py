"""Grid utility functions."""

import numpy as np


def grid_to_text(grid: np.ndarray) -> str:
    """Convert a grid to a readable text representation."""
    rows = []
    for row in grid:
        rows.append("[" + ",".join(str(int(v)) for v in row) + "]")
    return "[\n  " + ",\n  ".join(rows) + "\n]"


def format_grid_for_display(grid: np.ndarray, label: str = "") -> str:
    """Format a grid for display with optional label."""
    shape_str = f"({grid.shape[0]}x{grid.shape[1]})"
    header = f"--- {label} {shape_str} ---" if label else f"--- Grid {shape_str} ---"
    return f"{header}\n{grid_to_text(grid)}"


def grid_diff_matrix(expected: np.ndarray, actual: np.ndarray) -> str:
    """
    Generate a cell-by-cell diff matrix showing which cells match/differ.

    Returns a string with ✓ for matches and X for differences.
    """
    if expected.shape != actual.shape:
        return f"Shape mismatch: expected {expected.shape}, got {actual.shape}"

    lines = []
    for i in range(expected.shape[0]):
        row = []
        for j in range(expected.shape[1]):
            if expected[i, j] == actual[i, j]:
                row.append("✓")
            else:
                row.append(f"{actual[i,j]}→{expected[i,j]}")
        lines.append(" | ".join(row))

    return "\n".join(lines)


def compute_diff_feedback(expected: np.ndarray, actual: np.ndarray) -> str:
    """
    Generate detailed feedback about differences between expected and actual grids.
    """
    if expected.shape != actual.shape:
        return f"Shape mismatch: expected {expected.shape}, got {actual.shape}"

    diff_count = np.sum(expected != actual)
    total_cells = expected.size
    accuracy = (total_cells - diff_count) / total_cells * 100

    lines = [
        f"Accuracy: {accuracy:.1f}% ({total_cells - diff_count}/{total_cells} cells correct)",
        f"Errors: {diff_count} cells differ",
        "",
        "Diff matrix (actual→expected for wrong cells):",
        grid_diff_matrix(expected, actual),
    ]

    return "\n".join(lines)

