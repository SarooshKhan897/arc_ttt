"""Object extraction and grid analysis utilities."""

from typing import Any

import numpy as np
from scipy import ndimage

from solvers.perceiver.models import EnhancedObject, GridPerception, TransformDelta, color_name


class ObjectPreprocessor:
    """Extract and analyze objects from grids."""

    @staticmethod
    def extract_objects(grid: np.ndarray, background: int | None = None) -> list[EnhancedObject]:
        """Extract connected components (objects) from a grid."""
        if background is None:
            unique, counts = np.unique(grid, return_counts=True)
            background = unique[np.argmax(counts)]

        objects = []
        for color in np.unique(grid):
            if color == background:
                continue

            mask = (grid == color).astype(int)
            labeled, num_features = ndimage.label(mask)

            for label_id in range(1, num_features + 1):
                pixels = set(zip(*np.where(labeled == label_id)))
                if not pixels:
                    continue

                rows = [p[0] for p in pixels]
                cols = [p[1] for p in pixels]
                min_row, max_row = min(rows), max(rows)
                min_col, max_col = min(cols), max(cols)

                bbox_area = (max_row - min_row + 1) * (max_col - min_col + 1)
                is_rectangle = len(pixels) == bbox_area

                centroid = (sum(rows) / len(rows), sum(cols) / len(cols))

                objects.append(EnhancedObject(
                    pixels=pixels,
                    color=int(color),
                    bounding_box=(min_row, min_col, max_row, max_col),
                    size=len(pixels),
                    is_rectangle=is_rectangle,
                    centroid=centroid,
                ))

        return objects

    @staticmethod
    def detect_symmetry(grid: np.ndarray) -> dict[str, bool]:
        """Detect various types of symmetry in a grid."""
        h_sym = np.array_equal(grid, np.flip(grid, axis=1))
        v_sym = np.array_equal(grid, np.flip(grid, axis=0))
        d_sym = grid.shape[0] == grid.shape[1] and np.array_equal(grid, grid.T)
        rot_180 = np.array_equal(grid, np.rot90(grid, 2))

        return {
            "horizontal": h_sym,
            "vertical": v_sym,
            "diagonal": d_sym,
            "rotational_180": rot_180,
        }

    @staticmethod
    def detect_tiling(grid: np.ndarray) -> dict[str, Any]:
        """Detect if the grid is composed of repeated tiles."""
        h, w = grid.shape
        for tile_h in range(1, h // 2 + 1):
            if h % tile_h != 0:
                continue
            for tile_w in range(1, w // 2 + 1):
                if w % tile_w != 0:
                    continue

                tile = grid[:tile_h, :tile_w]
                is_tiled = True

                for i in range(0, h, tile_h):
                    for j in range(0, w, tile_w):
                        if not np.array_equal(grid[i:i + tile_h, j:j + tile_w], tile):
                            is_tiled = False
                            break
                    if not is_tiled:
                        break

                if is_tiled and (tile_h < h or tile_w < w):
                    return {
                        "is_tiled": True,
                        "tile_shape": (tile_h, tile_w),
                        "repetitions": (h // tile_h, w // tile_w),
                    }

        return {"is_tiled": False}

    @staticmethod
    def detect_frame(grid: np.ndarray) -> dict[str, Any]:
        """Detect if the grid has a uniform border/frame."""
        if grid.shape[0] < 3 or grid.shape[1] < 3:
            return {"has_frame": False}

        top = grid[0, :]
        bottom = grid[-1, :]
        left = grid[:, 0]
        right = grid[:, -1]

        border_colors = set(top) | set(bottom) | set(left) | set(right)

        if len(border_colors) == 1:
            border_color = border_colors.pop()
            return {
                "has_frame": True,
                "uniform_border": True,
                "border_color": int(border_color),
            }

        return {"has_frame": False, "uniform_border": False}


def perceive_grid_fast(grid: np.ndarray) -> GridPerception:
    """Fast code-based perception of a grid."""
    unique, counts = np.unique(grid, return_counts=True)
    background = unique[np.argmax(counts)]

    objects = ObjectPreprocessor.extract_objects(grid, background)
    symmetry = ObjectPreprocessor.detect_symmetry(grid)
    tiling = ObjectPreprocessor.detect_tiling(grid)

    patterns = []
    if symmetry["horizontal"]:
        patterns.append("horizontal_symmetry")
    if symmetry["vertical"]:
        patterns.append("vertical_symmetry")
    if symmetry["diagonal"]:
        patterns.append("diagonal_symmetry")
    if symmetry["rotational_180"]:
        patterns.append("rotational_180_symmetry")
    if tiling.get("is_tiled"):
        patterns.append(f"tiled_{tiling['tile_shape']}")

    return GridPerception(
        grid_shape=grid.shape,
        background_color=int(background),
        objects=objects,
        symmetry=symmetry,
        patterns=patterns,
        color_counts={int(c): int(cnt) for c, cnt in zip(unique, counts)},
    )


def compare_grids_fast(input_grid: np.ndarray, output_grid: np.ndarray) -> TransformDelta:
    """Create a structured delta between input and output (code-based)."""
    inp = np.array(input_grid)
    out = np.array(output_grid)

    size_change = (out.shape[0] - inp.shape[0], out.shape[1] - inp.shape[1])

    inp_colors = set(np.unique(inp))
    out_colors = set(np.unique(out))
    new_colors = out_colors - inp_colors
    removed_colors = inp_colors - out_colors

    color_changes = []
    if new_colors:
        color_changes.append({"type": "added", "colors": [color_name(c) for c in new_colors]})
    if removed_colors:
        color_changes.append({"type": "removed", "colors": [color_name(c) for c in removed_colors]})

    constants = []
    if size_change == (0, 0):
        constants.append("grid_size_preserved")
    if inp_colors == out_colors:
        constants.append("color_palette_preserved")

    return TransformDelta(
        object_changes=[],
        color_changes=color_changes,
        size_change=size_change,
        structural_changes=[],
        constants=constants,
        summary=f"Size: {inp.shape} → {out.shape}",
    )

