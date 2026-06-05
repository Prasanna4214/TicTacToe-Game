import numpy as np


class GridOps:
    @staticmethod
    def rotate90(grid: np.ndarray) -> np.ndarray:
        return np.rot90(grid)

    @staticmethod
    def rotate180(grid: np.ndarray) -> np.ndarray:
        return np.rot90(grid, 2)

    @staticmethod
    def rotate270(grid: np.ndarray) -> np.ndarray:
        return np.rot90(grid, 3)

    @staticmethod
    def flip_horizontal(grid: np.ndarray) -> np.ndarray:
        return np.fliplr(grid)

    @staticmethod
    def flip_vertical(grid: np.ndarray) -> np.ndarray:
        return np.flipud(grid)

    @staticmethod
    def transpose(grid: np.ndarray) -> np.ndarray:
        return np.transpose(grid)

    @staticmethod
    def same_shape(grid1: np.ndarray, grid2: np.ndarray) -> bool:
        return grid1.shape == grid2.shape

    @staticmethod
    def same_content(grid1: np.ndarray, grid2: np.ndarray) -> bool:
        return np.array_equal(grid1, grid2)

    @staticmethod
    def nonzero_positions(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.where(grid != 0)

    @staticmethod
    def bounding_box(grid: np.ndarray) -> tuple[int, int, int, int] | None:
        rows, cols = GridOps.nonzero_positions(grid)
        if len(rows) == 0:
            return None
        return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())

    @staticmethod
    def crop_bounding_box(grid: np.ndarray) -> np.ndarray:
        bbox = GridOps.bounding_box(grid)
        if bbox is None:
            return grid.copy()
        r1, r2, c1, c2 = bbox
        return grid[r1:r2 + 1, c1:c2 + 1]

    @staticmethod
    def create_empty(rows: int, cols: int, fill: int = 0) -> np.ndarray:
        return np.full((rows, cols), fill)


class ColorOps:
    @staticmethod
    def unique_colors(grid: np.ndarray) -> list:
        return list(np.unique(grid))

    @staticmethod
    def nonzero_colors(grid: np.ndarray) -> list:
        return [color for color in np.unique(grid) if color != 0]

    @staticmethod
    def replace_color(grid: np.ndarray, old_color: int, new_color: int) -> np.ndarray:
        result = grid.copy()
        result[result == old_color] = new_color
        return result

    @staticmethod
    def count_color(grid: np.ndarray, color: int) -> int:
        return int(np.sum(grid == color))

    @staticmethod
    def count_nonzero(grid: np.ndarray) -> int:
        return int(np.sum(grid != 0))


class Components:
    NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))

    @staticmethod
    def cells_bounding_box(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        return min(rows), max(rows), min(cols), max(cols)

    @staticmethod
    def connected_components(grid: np.ndarray) -> list[dict]:
        rows, cols = grid.shape
        visited = np.zeros((rows, cols), dtype=bool)
        components = []

        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == 0 or visited[r, c]:
                    continue

                stack = [(r, c)]
                visited[r, c] = True
                cells = []

                while stack:
                    cr, cc = stack.pop()
                    cells.append((cr, cc))

                    for dr, dc in Components.NEIGHBORS:
                        nr, nc = cr + dr, cc + dc
                        if not (0 <= nr < rows and 0 <= nc < cols):
                            continue
                        if visited[nr, nc] or grid[nr, nc] == 0:
                            continue
                        visited[nr, nc] = True
                        stack.append((nr, nc))

                components.append({
                    "id": len(components) + 1,
                    "cells": cells,
                    "size": len(cells),
                    "bbox": Components.cells_bounding_box(cells),
                })

        return components

    @staticmethod
    def detect_objects(grid: np.ndarray) -> list[dict]:
        objects = []
        for component in Components.connected_components(grid):
            first_r, first_c = component["cells"][0]
            objects.append({
                **component,
                "color": int(grid[first_r, first_c]),
            })
        return objects

    @staticmethod
    def extract_object(grid: np.ndarray, obj: dict) -> np.ndarray:
        r1, r2, c1, c2 = obj["bbox"]
        return grid[r1:r2 + 1, c1:c2 + 1]

    @staticmethod
    def connected_blobs(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
        return [component["bbox"] for component in Components.connected_components(grid)]


class SectionRules:
    @staticmethod
    def find_separator(grid: np.ndarray) -> tuple[int | None, int | None]:
        rows, cols = grid.shape

        for r in range(rows):
            row_vals = set(grid[r, :].tolist())
            if len(row_vals) == 1:
                sep_val = next(iter(row_vals))
                if sep_val != 0 and not np.any(np.delete(grid, r, axis=0) == sep_val):
                    return 0, r

        for c in range(cols):
            col_vals = set(grid[:, c].tolist())
            if len(col_vals) == 1:
                sep_val = next(iter(col_vals))
                if sep_val != 0 and not np.any(np.delete(grid, c, axis=1) == sep_val):
                    return 1, c

        return None, None

    @staticmethod
    def split_grid(grid: np.ndarray, axis: int, idx: int) -> tuple[np.ndarray, np.ndarray]:
        if axis == 0:
            return grid[:idx], grid[idx + 1:]
        return grid[:, :idx], grid[:, idx + 1:]

    @staticmethod
    def detect(train_pairs: list) -> tuple[str | None, int | None]:
        candidates = ["union", "intersection", "neither", "xor"]
        learned_marker = None

        for inp, out in train_pairs:
            axis, idx = SectionRules.find_separator(inp)
            if axis is None:
                return None, None

            a, b = SectionRules.split_grid(inp, axis, idx)
            if a.shape != b.shape:
                return None, None

            out_vals = [int(v) for v in np.unique(out) if v != 0]
            if len(out_vals) != 1:
                return None, None

            marker = out_vals[0]
            if learned_marker is None:
                learned_marker = marker
            elif learned_marker != marker:
                return None, None

            ops = SectionRules._ops(a, b, marker)
            candidates = [name for name in candidates if np.array_equal(ops[name], out)]
            if not candidates:
                return None, None

        return candidates[0], learned_marker

    @staticmethod
    def apply(grid: np.ndarray, rule: str, marker: int) -> np.ndarray | None:
        axis, idx = SectionRules.find_separator(grid)
        if axis is None or rule is None or marker is None:
            return None

        a, b = SectionRules.split_grid(grid, axis, idx)
        if a.shape != b.shape:
            return None

        return SectionRules._ops(a, b, marker).get(rule)

    @staticmethod
    def _ops(a: np.ndarray, b: np.ndarray, marker: int) -> dict[str, np.ndarray]:
        return {
            "union": np.where((a != 0) | (b != 0), marker, 0),
            "intersection": np.where((a != 0) & (b != 0), marker, 0),
            "neither": np.where((a == 0) & (b == 0), marker, 0),
            "xor": np.where((a != 0) ^ (b != 0), marker, 0),
        }


class ColorRules:
    @staticmethod
    def learn_mapping(train_pairs: list) -> dict | None:
        mapping = {}
        for inp, out in train_pairs:
            if inp.shape != out.shape:
                return None
            for val_in, val_out in zip(inp.flatten(), out.flatten()):
                val_in, val_out = int(val_in), int(val_out)
                if val_in in mapping and mapping[val_in] != val_out:
                    return None
                mapping[val_in] = val_out
        return mapping

    @staticmethod
    def apply_mapping(grid: np.ndarray, mapping: dict) -> np.ndarray:
        return np.vectorize(lambda x: mapping.get(int(x), int(x)))(grid)

    @staticmethod
    def learn_masked_color(train_pairs: list) -> dict | None:
        source_color = None

        for inp, out in train_pairs:
            input_colors = [int(v) for v in np.unique(inp) if v != 0]
            output_colors = [int(v) for v in np.unique(out) if v != 0]
            if len(input_colors) != 2 or len(output_colors) != 1:
                return None

            paint_color = output_colors[0]
            candidates = [
                color for color in input_colors
                if np.array_equal(np.where(inp == color, paint_color, 0), out)
            ]
            if len(candidates) != 1:
                return None

            if source_color is None:
                source_color = candidates[0]
            elif source_color != candidates[0]:
                return None

        return {"source_color": source_color}

    @staticmethod
    def apply_masked_color(grid: np.ndarray, rule: dict) -> np.ndarray | None:
        source_color = rule.get("source_color") if rule else None
        if source_color is None:
            return None

        paint_colors = [
            int(v) for v in np.unique(grid)
            if int(v) != 0 and int(v) != int(source_color)
        ]
        if len(paint_colors) != 1:
            return None

        return np.where(grid == source_color, paint_colors[0], 0)


class ShapeRules:
    @staticmethod
    def make_hollow(grid: np.ndarray) -> np.ndarray:
        result = grid.copy()
        for component in Components.connected_components(grid):
            top, bottom, left, right = component["bbox"]
            result[top + 1:bottom, left + 1:right] = 0
        return result

    @staticmethod
    def construct_spiral_rectangle(grid: np.ndarray, color: int = 3) -> np.ndarray:
        rows, cols = grid.shape
        result = np.zeros((rows, cols), dtype=int)

        k = 0
        while True:
            r1, c1 = 2 * k, 2 * k
            r2, c2 = rows - 1 - 2 * k, cols - 1 - 2 * k

            if k > 0 and rows >= 4 * k:
                result[r1, c1 - 1] = color

            if r1 >= r2 or c1 >= c2:
                if r1 == r2 and c1 == c2:
                    result[r1, c1] = color
                break

            result[r1, c1:c2] = color
            result[r1:r2 + 1, c2] = color
            result[r2, c1:c2 + 1] = color
            result[r1 + 2:r2, c1] = color
            k += 1

        return result

    @staticmethod
    def draw_border(grid: np.ndarray, color: int) -> np.ndarray:
        result = grid.copy()
        result[0, :] = color
        result[-1, :] = color
        result[:, 0] = color
        result[:, -1] = color
        return result

    @staticmethod
    def invert_nested_colors(grid: np.ndarray) -> np.ndarray:
        blobs = Components.connected_blobs(grid)
        if not blobs:
            return grid.copy()

        result = grid.copy()
        for r1, r2, c1, c2 in blobs:
            region = grid[r1:r2 + 1, c1:c2 + 1]
            colors = [v for v in np.unique(region) if v != 0]
            if len(colors) != 2:
                continue
            a, b = colors
            target = result[r1:r2 + 1, c1:c2 + 1]
            target[region == a] = b
            target[region == b] = a

        all_r1 = min(blob[0] for blob in blobs)
        all_r2 = max(blob[1] for blob in blobs)
        all_c1 = min(blob[2] for blob in blobs)
        all_c2 = max(blob[3] for blob in blobs)
        return result[all_r1:all_r2 + 1, all_c1:all_c2 + 1]

    @staticmethod
    def expand_dots_to_blocks(grid: np.ndarray, out_val: int) -> np.ndarray | None:
        dot_positions = list(zip(*np.where(grid != 0)))
        if not dot_positions:
            return None

        block_size = ShapeRules._infer_block_size(grid, dot_positions)
        if block_size is None:
            return None

        result = np.zeros_like(grid)
        for dr, dc in dot_positions:
            tile_r = int(dr) // block_size
            tile_c = int(dc) // block_size
            r1 = tile_r * block_size
            c1 = tile_c * block_size
            result[r1:r1 + block_size, c1:c1 + block_size] = out_val
        return result

    @staticmethod
    def mirror_bottom_half(grid: np.ndarray) -> np.ndarray | None:
        first_nonzero = next(
            (r for r in range(grid.shape[0]) if np.any(grid[r] != 0)),
            None,
        )
        if first_nonzero is None:
            return None

        bottom = grid[first_nonzero:]
        return np.vstack([bottom[::-1], bottom])

    @staticmethod
    def draw_x_diagonals(grid: np.ndarray) -> np.ndarray | None:
        positions = list(zip(*np.where(grid != 0)))
        if not positions:
            return None

        rows, cols = grid.shape
        start_r, start_c = positions[0]
        color = grid[start_r, start_c]
        output = grid.copy()

        for dr, dc in ((-1, -1), (1, 1), (-1, 1), (1, -1)):
            r, c = int(start_r), int(start_c)
            while 0 <= r < rows and 0 <= c < cols:
                output[r, c] = color
                r += dr
                c += dc

        return output

    @staticmethod
    def pull_cells_to_matching_border(grid: np.ndarray) -> np.ndarray:
        rows, cols = grid.shape
        output = np.zeros_like(grid)
        output[0, :] = grid[0, :]
        output[-1, :] = grid[-1, :]
        output[:, 0] = grid[:, 0]
        output[:, -1] = grid[:, -1]

        top_color = ShapeRules._first_nonzero(grid[0, :])
        bottom_color = ShapeRules._first_nonzero(grid[-1, :])
        left_color = ShapeRules._first_nonzero(grid[:, 0])
        right_color = ShapeRules._first_nonzero(grid[:, -1])

        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                val = grid[r, c]
                if val == 0:
                    continue
                if top_color is not None and val == top_color:
                    output[1, c] = val
                elif bottom_color is not None and val == bottom_color:
                    output[rows - 2, c] = val
                elif left_color is not None and val == left_color:
                    output[r, 1] = val
                elif right_color is not None and val == right_color:
                    output[r, cols - 2] = val

        return output

    @staticmethod
    def _infer_block_size(grid: np.ndarray, dot_positions: list) -> int | None:
        rows, cols = grid.shape
        for block_size in range(2, min(rows, cols) + 1):
            if rows % block_size != 0 or cols % block_size != 0:
                continue
            valid = all(
                (int(r) // block_size * block_size + block_size // 2 == int(r)) and
                (int(c) // block_size * block_size + block_size // 2 == int(c))
                for r, c in dot_positions
            )
            if valid:
                return block_size
        return None

    @staticmethod
    def _first_nonzero(values) -> int | None:
        vals = [int(v) for v in values if v != 0]
        return vals[0] if vals else None


class SymmetryRules:
    @staticmethod
    def is_horizontal_symmetric(grid: np.ndarray) -> bool:
        return np.array_equal(grid, np.flipud(grid))

    @staticmethod
    def is_vertical_symmetric(grid: np.ndarray) -> bool:
        return np.array_equal(grid, np.fliplr(grid))


class LearnedRules:
    @staticmethod
    def fill_matching_edge_rows(grid: np.ndarray) -> np.ndarray:
        result = grid.copy()
        for r in range(grid.shape[0]):
            left, right = int(grid[r, 0]), int(grid[r, -1])
            if left != 0 and left == right:
                result[r, :] = left
        return result

    @staticmethod
    def extend_marker_ray(grid: np.ndarray) -> np.ndarray | None:
        colors = [int(v) for v in np.unique(grid) if v != 0]
        single_colors = [color for color in colors if np.sum(grid == color) == 1]
        object_colors = [color for color in colors if color not in single_colors]
        if len(single_colors) != 1 or len(object_colors) != 1:
            return None

        marker = single_colors[0]
        marker_pos = tuple(int(v) for v in np.argwhere(grid == marker)[0])
        object_cells = np.argwhere(grid == object_colors[0])
        r1, c1 = object_cells.min(axis=0)
        r2, c2 = object_cells.max(axis=0)
        mr, mc = marker_pos
        output = grid.copy()

        coords = set(map(tuple, np.argwhere(grid != 0)))
        horizontal = all((2 * mr - r, c) in coords for r, c in coords)
        vertical = all((r, 2 * mc - c) in coords for r, c in coords)

        if horizontal:
            left = sum(1 for _, c in coords if c < mc)
            right = sum(1 for _, c in coords if c > mc)
            if right >= left:
                output[mr, int(c2) + 1:] = marker
            else:
                output[mr, :int(c1)] = marker
            return output

        if vertical:
            up = sum(1 for r, _ in coords if r < mr)
            down = sum(1 for r, _ in coords if r > mr)
            if down >= up:
                output[int(r2) + 1:, mc] = marker
            else:
                output[:int(r1), mc] = marker
            return output

        return None

    @staticmethod
    def crop_between_corner_markers(grid: np.ndarray) -> np.ndarray | None:
        for color in [int(v) for v in np.unique(grid) if v != 0]:
            positions = np.argwhere(grid == color)
            if len(positions) != 4:
                continue

            rows = sorted(set(int(r) for r, _ in positions))
            cols = sorted(set(int(c) for _, c in positions))
            corners = {(r, c) for r in rows for c in cols}
            actual = {tuple(map(int, pos)) for pos in positions}

            if len(rows) == 2 and len(cols) == 2 and actual == corners:
                r1, r2 = rows
                c1, c2 = cols
                interior = grid[r1 + 1:r2, c1 + 1:c2]
                return np.where(interior != 0, color, 0)

        return None

    @staticmethod
    def diagonal_block_rays(grid: np.ndarray, color_dirs: dict[int, tuple[int, int]]) -> np.ndarray | None:
        output = grid.copy()
        for color, direction in color_dirs.items():
            cells = np.argwhere(grid == color)
            if len(cells) != 4:
                return None

            rows = sorted(set(int(r) for r, _ in cells))
            cols = sorted(set(int(c) for _, c in cells))
            if len(rows) != 2 or len(cols) != 2:
                return None

            dr, dc = direction
            start_r = rows[0] if dr < 0 else rows[-1]
            start_c = cols[0] if dc < 0 else cols[-1]
            r, c = start_r + dr, start_c + dc
            while 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                output[r, c] = color
                r += dr
                c += dc

        return output

    @staticmethod
    def learn_diagonal_block_rays(train_pairs: list) -> dict[int, tuple[int, int]] | None:
        learned = {}
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        for inp, out in train_pairs:
            if inp.shape != out.shape:
                return None

            colors = [int(v) for v in np.unique(inp) if v != 0]
            for color in colors:
                added = np.argwhere((out == color) & (inp != color))
                if len(added) == 0:
                    continue

                matches = []
                for direction in directions:
                    pred = LearnedRules.diagonal_block_rays(inp, {color: direction})
                    if pred is not None and np.array_equal(pred == color, out == color):
                        matches.append(direction)

                if len(matches) != 1:
                    return None
                if color in learned and learned[color] != matches[0]:
                    return None
                learned[color] = matches[0]

        return learned if learned else None

    @staticmethod
    def mirror_expand_2x2(grid: np.ndarray) -> np.ndarray:
        top = np.hstack([grid, np.fliplr(grid)])
        bottom = np.flipud(top)
        return np.vstack([top, bottom])

    @staticmethod
    def overlay_split_sections(grid: np.ndarray) -> np.ndarray | None:
        axis, idx = SectionRules.find_separator(grid)
        if axis is None:
            return None

        a, b = SectionRules.split_grid(grid, axis, idx)
        if a.shape != b.shape:
            return None

        result = a.copy()
        result[b != 0] = b[b != 0]
        return result

    @staticmethod
    def vertical_histogram(grid: np.ndarray) -> np.ndarray | None:
        colors = [int(v) for v in np.unique(grid) if v != 0]
        if len(colors) < 2:
            return None

        counts = [(color, int(np.sum(grid == color))) for color in colors]
        counts.sort(key=lambda item: (-item[1], item[0]))
        height = counts[0][1]
        output = np.zeros((height, len(counts)), dtype=int)

        for col, (color, count) in enumerate(counts):
            output[:count, col] = color

        return output

    @staticmethod
    def growing_prefix_staircase(grid: np.ndarray) -> np.ndarray | None:
        if grid.shape[0] != 1:
            return None

        row = grid[0]
        colors = [int(v) for v in np.unique(row) if v != 0]
        if len(colors) != 1:
            return None

        color = colors[0]
        filled = np.where(row == color)[0]
        if len(filled) == 0 or not np.array_equal(filled, np.arange(filled[-1] + 1)):
            return None

        height = grid.shape[1] // 2
        output = np.zeros((height, grid.shape[1]), dtype=int)
        for r in range(height):
            output[r, :min(grid.shape[1], len(filled) + r)] = color
        return output

    @staticmethod
    def overlay_panels(grid: np.ndarray) -> np.ndarray | None:
        full_cols_by_color = {}
        for c in range(grid.shape[1]):
            values = set(grid[:, c].tolist())
            if len(values) == 1 and next(iter(values)) != 0:
                full_cols_by_color.setdefault(int(next(iter(values))), []).append(c)

        for sep_cols in full_cols_by_color.values():
            cuts = [-1] + sep_cols + [grid.shape[1]]
            panels = [grid[:, cuts[i] + 1:cuts[i + 1]] for i in range(len(cuts) - 1)]
            if len({panel.shape for panel in panels}) != 1:
                continue

            result = np.zeros_like(panels[0])
            for panel in reversed(panels):
                result[panel != 0] = panel[panel != 0]
            return result

        return None

    @staticmethod
    def learn_dot_move(train_pairs: list) -> dict | None:
        moving_color = None

        for inp, out in train_pairs:
            colors = [int(v) for v in np.unique(inp) if v != 0]
            if len(colors) != 2 or inp.shape != out.shape:
                return None

            moved = []
            for color in colors:
                in_pos = np.argwhere(inp == color)
                out_pos = np.argwhere(out == color)
                if len(in_pos) != 1 or len(out_pos) != 1:
                    return None
                if not np.array_equal(in_pos[0], out_pos[0]):
                    moved.append(color)

            if len(moved) != 1:
                return None
            if moving_color is None:
                moving_color = moved[0]
            elif moving_color != moved[0]:
                return None

        return {"moving_color": moving_color}

    @staticmethod
    def move_dot_toward_anchor(grid: np.ndarray, rule: dict) -> np.ndarray | None:
        moving_color = rule.get("moving_color") if rule else None
        colors = [int(v) for v in np.unique(grid) if v != 0]
        if moving_color is None or moving_color not in colors or len(colors) != 2:
            return None

        anchor_color = next(color for color in colors if color != moving_color)
        moving_pos = np.argwhere(grid == moving_color)
        anchor_pos = np.argwhere(grid == anchor_color)
        if len(moving_pos) != 1 or len(anchor_pos) != 1:
            return None

        mr, mc = moving_pos[0]
        ar, ac = anchor_pos[0]
        nr = int(mr + np.sign(ar - mr))
        nc = int(mc + np.sign(ac - mc))

        output = grid.copy()
        output[mr, mc] = 0
        output[nr, nc] = moving_color
        return output

    @staticmethod
    def learn_region_fill(train_pairs: list) -> dict | None:
        outside_color = None
        inside_color = None

        for inp, out in train_pairs:
            if inp.shape != out.shape:
                return None

            outside = LearnedRules._outside_zero_mask(inp)
            inside = (inp == 0) & ~outside
            if not np.any(inside):
                return None

            out_outside = set(int(v) for v in out[outside])
            out_inside = set(int(v) for v in out[inside])
            if len(out_outside) != 1 or len(out_inside) != 1:
                return None

            current_outside = next(iter(out_outside))
            current_inside = next(iter(out_inside))
            if outside_color is None:
                outside_color = current_outside
                inside_color = current_inside
            elif outside_color != current_outside or inside_color != current_inside:
                return None

        return {"outside": outside_color, "inside": inside_color}

    @staticmethod
    def fill_regions(grid: np.ndarray, rule: dict) -> np.ndarray | None:
        if not rule:
            return None

        output = grid.copy()
        outside = LearnedRules._outside_zero_mask(grid)
        inside = (grid == 0) & ~outside
        output[outside] = rule["outside"]
        output[inside] = rule["inside"]
        return output

    @staticmethod
    def learn_complex_component_recolor(train_pairs: list) -> dict | None:
        source_color = None
        target_color = None

        for inp, out in train_pairs:
            if inp.shape != out.shape:
                return None

            diff = inp != out
            if not np.any(diff):
                return None

            sources = set(int(v) for v in inp[diff])
            targets = set(int(v) for v in out[diff])
            if len(sources) != 1 or len(targets) != 1:
                return None

            current_source = next(iter(sources))
            current_target = next(iter(targets))
            if source_color is None:
                source_color = current_source
                target_color = current_target
            elif source_color != current_source or target_color != current_target:
                return None

            if not LearnedRules._complex_recolor_matches(inp, out, source_color, target_color):
                return None

        return {"source": source_color, "target": target_color}

    @staticmethod
    def recolor_complex_components(grid: np.ndarray, rule: dict) -> np.ndarray | None:
        if not rule:
            return None

        source = rule["source"]
        target = rule["target"]
        output = grid.copy()
        mask = np.where(grid == source, source, 0)

        for component in Components.connected_components(mask):
            if LearnedRules._is_complex_component(component, grid.shape):
                for r, c in component["cells"]:
                    output[r, c] = target

        return output

    @staticmethod
    def move_marks_outside_container(grid: np.ndarray) -> np.ndarray | None:
        colors = [int(v) for v in np.unique(grid) if v != 0]
        if len(colors) != 2:
            return None

        counts = {color: int(np.sum(grid == color)) for color in colors}
        container = max(colors, key=lambda color: counts[color])
        mark = min(colors, key=lambda color: counts[color])
        container_cells = np.argwhere(grid == container)
        if len(container_cells) == 0:
            return None

        r1, c1 = container_cells.min(axis=0)
        r2, c2 = container_cells.max(axis=0)
        output = grid.copy()
        output[grid == mark] = 0

        mark_grid = np.where(grid == mark, mark, 0)
        for component in Components.connected_components(mark_grid):
            distances = LearnedRules._container_ray_distances(
                grid,
                component["cells"],
                container,
            )
            side = min(distances, key=distances.get)
            if distances[side] == float("inf"):
                return None

            for r, c in component["cells"]:
                if side == "top":
                    nr, nc = 2 * int(r1) - r, c
                elif side == "bottom":
                    nr, nc = 2 * int(r2) - r, c
                elif side == "left":
                    nr, nc = r, 2 * int(c1) - c
                else:
                    nr, nc = r, 2 * int(c2) - c

                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    output[nr, nc] = mark

        return output

    @staticmethod
    def learn_corner_dot_expansion(train_pairs: list) -> dict | None:
        connector = None

        for inp, out in train_pairs:
            input_colors = [int(v) for v in np.unique(inp) if v != 0]
            output_extra = [
                int(v) for v in np.unique(out)
                if v != 0 and int(v) not in input_colors
            ]
            if len(input_colors) != 2 or len(output_extra) != 1:
                return None

            current_connector = output_extra[0]
            if connector is None:
                connector = current_connector
            elif connector != current_connector:
                return None

            expected = LearnedRules.expand_corner_dots(inp, {"connector": connector})
            if expected is None or not np.array_equal(expected, out):
                return None

        return {"connector": connector}

    @staticmethod
    def expand_corner_dots(grid: np.ndarray, rule: dict) -> np.ndarray | None:
        connector = rule.get("connector") if rule else None
        colors = [int(v) for v in np.unique(grid) if v != 0]
        positions = np.argwhere(grid != 0)
        if connector is None or len(colors) != 2 or len(positions) != 4:
            return None

        rows = sorted(set(int(r) for r, _ in positions))
        cols = sorted(set(int(c) for _, c in positions))
        if len(rows) != 2 or len(cols) != 2:
            return None

        corners = {(r, c) for r in rows for c in cols}
        actual = {tuple(map(int, pos)) for pos in positions}
        if actual != corners:
            return None

        output = np.zeros_like(grid)
        other_color = {colors[0]: colors[1], colors[1]: colors[0]}

        for r, c in actual:
            color = int(grid[r, c])
            fill = other_color[color]
            output[
                max(0, r - 1):min(grid.shape[0], r + 2),
                max(0, c - 1):min(grid.shape[1], c + 2),
            ] = fill
            output[r, c] = color

        for r in rows:
            start = cols[0] + 2
            end = cols[1] - 2
            for c in LearnedRules._connector_positions(start, end):
                output[r, c] = connector

        for c in cols:
            start = rows[0] + 2
            end = rows[1] - 2
            for r in LearnedRules._connector_positions(start, end):
                output[r, c] = connector

        return output

    @staticmethod
    def learn_equal_half_rule(train_pairs: list) -> dict | None:
        candidates = ["union", "intersection", "xor"]
        marker = None

        for inp, out in train_pairs:
            if inp.shape[0] != out.shape[0] * 2 or inp.shape[1] != out.shape[1]:
                return None

            top = inp[:out.shape[0]]
            bottom = inp[out.shape[0]:]
            out_vals = [int(v) for v in np.unique(out) if v != 0]
            if len(out_vals) != 1:
                return None

            current_marker = out_vals[0]
            if marker is None:
                marker = current_marker
            elif marker != current_marker:
                return None

            ops = SectionRules._ops(top, bottom, marker)
            candidates = [name for name in candidates if np.array_equal(ops[name], out)]
            if not candidates:
                return None

        return {"rule": candidates[0], "marker": marker}

    @staticmethod
    def apply_equal_half_rule(grid: np.ndarray, rule: dict) -> np.ndarray | None:
        if not rule or grid.shape[0] % 2 != 0:
            return None

        mid = grid.shape[0] // 2
        top = grid[:mid]
        bottom = grid[mid:]
        return SectionRules._ops(top, bottom, rule["marker"]).get(rule["rule"])

    @staticmethod
    def learn_straight_path(train_pairs: list) -> dict | None:
        endpoint_color = None
        crossing_color = None
        path_color = None

        for inp, out in train_pairs:
            if inp.shape != out.shape:
                return None

            colors = [int(v) for v in np.unique(inp) if v != 0]
            endpoint_candidates = [
                color for color in colors
                if len(np.argwhere(inp == color)) == 2
            ]
            if len(endpoint_candidates) != 1:
                return None

            current_endpoint = endpoint_candidates[0]
            extra_colors = [
                int(v) for v in np.unique(out)
                if v != 0 and int(v) not in colors
            ]
            current_crossing = extra_colors[0] if len(extra_colors) == 1 else None

            changed_to_endpoint = (inp != current_endpoint) & (out == current_endpoint)
            changed_to_cross = (
                np.zeros_like(inp, dtype=bool)
                if current_crossing is None
                else ((inp != current_crossing) & (out == current_crossing))
            )
            if not np.any(changed_to_endpoint) and not np.any(changed_to_cross):
                return None

            current_path = current_endpoint
            if endpoint_color is None:
                endpoint_color = current_endpoint
                crossing_color = current_crossing
                path_color = current_path
            elif endpoint_color != current_endpoint or crossing_color != current_crossing:
                return None

            pred = LearnedRules.draw_straight_path(
                inp,
                {
                    "endpoint": endpoint_color,
                    "path": path_color,
                    "crossing": crossing_color,
                },
            )
            if pred is None or not np.array_equal(pred, out):
                return None

        return {"endpoint": endpoint_color, "path": path_color, "crossing": crossing_color}

    @staticmethod
    def draw_straight_path(grid: np.ndarray, rule: dict) -> np.ndarray | None:
        if not rule:
            return None

        endpoint = rule["endpoint"]
        path = rule["path"]
        crossing = rule["crossing"]
        points = np.argwhere(grid == endpoint)
        if len(points) != 2:
            return None

        r1, c1 = [int(v) for v in points[0]]
        r2, c2 = [int(v) for v in points[1]]
        dr = int(np.sign(r2 - r1))
        dc = int(np.sign(c2 - c1))

        if not (r1 == r2 or c1 == c2 or abs(r2 - r1) == abs(c2 - c1)):
            return None

        output = grid.copy()
        r, c = r1 + dr, c1 + dc
        while (r, c) != (r2, c2):
            if grid[r, c] != 0 and grid[r, c] != endpoint and crossing is not None:
                output[r, c] = crossing
            else:
                output[r, c] = path
            r += dr
            c += dc

        return output

    @staticmethod
    def learn_point_connector(train_pairs: list) -> dict | None:
        point_colors = None
        connector = None

        for inp, out in train_pairs:
            if inp.shape != out.shape:
                return None

            colors = [int(v) for v in np.unique(inp) if v != 0]
            singles = [color for color in colors if len(np.argwhere(inp == color)) == 1]
            if len(singles) != 2:
                return None

            extra = [int(v) for v in np.unique(out) if v != 0 and int(v) not in colors]
            if len(extra) != 1:
                return None

            if point_colors is None:
                point_colors = tuple(sorted(singles))
                connector = extra[0]
            elif point_colors != tuple(sorted(singles)) or connector != extra[0]:
                return None

            pred = LearnedRules.connect_points(inp, {"colors": point_colors, "connector": connector})
            if pred is None or not np.array_equal(pred, out):
                return None

        return {"colors": point_colors, "connector": connector}

    @staticmethod
    def connect_points(grid: np.ndarray, rule: dict) -> np.ndarray | None:
        if not rule:
            return None

        points = []
        for color in rule["colors"]:
            pos = np.argwhere(grid == color)
            if len(pos) != 1:
                return None
            points.append((color, int(pos[0][0]), int(pos[0][1])))

        start = next((point for point in points if point[0] == 1), points[0])
        end = points[1] if points[0] == start else points[0]
        _, r, c = start
        _, er, ec = end
        output = grid.copy()

        while abs(er - r) > 1 and abs(ec - c) > 1:
            r += int(np.sign(er - r))
            c += int(np.sign(ec - c))
            output[r, c] = rule["connector"]

        while abs(er - r) > 1:
            r += int(np.sign(er - r))
            output[r, c] = rule["connector"]

        while abs(ec - c) > 1:
            c += int(np.sign(ec - c))
            output[r, c] = rule["connector"]

        return output

    @staticmethod
    def learn_repeated_mirror_tile(train_pairs: list) -> dict | None:
        transforms = {
            "identity": lambda grid: grid,
            "rot180": lambda grid: np.rot90(grid, 2),
            "flipud": np.flipud,
            "fliplr": np.fliplr,
        }
        candidates = list(transforms)

        for inp, out in train_pairs:
            if out.shape != (inp.shape[0] * 3, inp.shape[1] * 3):
                return None

            valid = []
            for name in candidates:
                base = transforms[name](inp)
                band = np.hstack([base, np.fliplr(base), base])
                pred = np.vstack([band, np.flipud(band), band])
                if np.array_equal(pred, out):
                    valid.append(name)
            candidates = valid
            if not candidates:
                return None

        return {"transform": candidates[0]}

    @staticmethod
    def apply_repeated_mirror_tile(grid: np.ndarray, rule: dict) -> np.ndarray | None:
        transforms = {
            "identity": lambda value: value,
            "rot180": lambda value: np.rot90(value, 2),
            "flipud": np.flipud,
            "fliplr": np.fliplr,
        }
        if not rule or rule["transform"] not in transforms:
            return None

        base = transforms[rule["transform"]](grid)
        band = np.hstack([base, np.fliplr(base), base])
        return np.vstack([band, np.flipud(band), band])

    @staticmethod
    def block_frequency_histogram(grid: np.ndarray) -> np.ndarray | None:
        sep_color = LearnedRules._separator_color(grid)
        if sep_color is None:
            return None

        row_cuts = LearnedRules._full_indices(grid, axis=0, color=sep_color)
        col_cuts = LearnedRules._full_indices(grid, axis=1, color=sep_color)
        row_bounds = LearnedRules._bounds_from_cuts(grid.shape[0], row_cuts)
        col_bounds = LearnedRules._bounds_from_cuts(grid.shape[1], col_cuts)
        counts = {}

        for r1, r2 in row_bounds:
            for c1, c2 in col_bounds:
                region = grid[r1:r2, c1:c2]
                vals = [int(v) for v in np.unique(region) if v != 0 and int(v) != sep_color]
                if len(vals) == 1:
                    counts[vals[0]] = counts.get(vals[0], 0) + 1

        if not counts:
            return None

        items = sorted(counts.items(), key=lambda item: (item[1], item[0]))
        width = max(counts.values())
        output = np.zeros((len(items), width), dtype=int)
        for r, (color, count) in enumerate(items):
            output[r, :count] = color
        return output

    @staticmethod
    def pack_enclosed_markers(grid: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray | None:
        boundary_candidates = [
            int(v) for v in np.unique(grid)
            if v != 0 and np.sum(grid == v) > 1
        ]
        if not boundary_candidates:
            return None

        boundary = max(boundary_candidates, key=lambda color: np.sum(grid == color))
        outside = LearnedRules._outside_mask_with_walls(grid, {boundary})
        markers = [
            int(grid[r, c])
            for r in range(grid.shape[0])
            for c in range(grid.shape[1])
            if grid[r, c] != 0 and grid[r, c] != boundary and not outside[r, c]
        ]
        if not markers:
            return None

        color = markers[0]
        if any(marker != color for marker in markers):
            return None

        output = np.zeros(output_shape, dtype=int)
        for idx in range(min(len(markers), output.size)):
            output[idx // output.shape[1], idx % output.shape[1]] = color
        return output

    @staticmethod
    def drop_objects_to_floor(grid: np.ndarray) -> np.ndarray | None:
        bottom_vals = [int(v) for v in np.unique(grid[-1]) if v != 0]
        if len(bottom_vals) != 1 or not np.all(grid[-1] == bottom_vals[0]):
            return None

        floor = bottom_vals[0]
        output = np.where(grid == floor, floor, 0)
        movable = np.where((grid != 0) & (grid != floor), grid, 0)
        components = Components.detect_objects(movable)
        components.sort(key=lambda comp: comp["bbox"][1], reverse=True)

        for component in components:
            cells = [(int(r), int(c)) for r, c in component["cells"]]
            offset = 0
            while True:
                candidate = [(r + offset + 1, c) for r, c in cells]
                if any(r >= grid.shape[0] or output[r, c] != 0 for r, c in candidate):
                    break
                offset += 1

            for r, c in cells:
                output[r + offset, c] = grid[r, c]

        return output

    @staticmethod
    def learn_aligned_object_connector(train_pairs: list) -> dict | None:
        connector = None

        for inp, out in train_pairs:
            if inp.shape != out.shape:
                return None

            input_colors = [int(v) for v in np.unique(inp) if v != 0]
            extra = [int(v) for v in np.unique(out) if v != 0 and int(v) not in input_colors]
            if len(extra) != 1:
                return None

            if connector is None:
                connector = extra[0]
            elif connector != extra[0]:
                return None

            pred = LearnedRules.connect_aligned_objects(inp, {"connector": connector})
            if pred is None or not np.array_equal(pred, out):
                return None

        return {"connector": connector}

    @staticmethod
    def connect_aligned_objects(grid: np.ndarray, rule: dict) -> np.ndarray | None:
        if not rule:
            return None

        components = LearnedRules._eight_connected_objects(grid)
        output = grid.copy()
        made_connection = False

        for i, a in enumerate(components):
            for b in components[i + 1:]:
                if a["color"] != b["color"]:
                    continue

                ar1, ar2, ac1, ac2 = a["bbox"]
                br1, br2, bc1, bc2 = b["bbox"]
                acenter_r = int(round(sum(r for r, _ in a["cells"]) / len(a["cells"])))
                acenter_c = int(round(sum(c for _, c in a["cells"]) / len(a["cells"])))
                bcenter_r = int(round(sum(r for r, _ in b["cells"]) / len(b["cells"])))
                bcenter_c = int(round(sum(c for _, c in b["cells"]) / len(b["cells"])))

                if acenter_r == bcenter_r:
                    start = min(ac2, bc2) + 1
                    end = max(ac1, bc1)
                    if start < end and np.all(grid[acenter_r, start:end] == 0):
                        output[acenter_r, start:end] = rule["connector"]
                        made_connection = True

                if acenter_c == bcenter_c:
                    start = min(ar2, br2) + 1
                    end = max(ar1, br1)
                    if start < end and np.all(grid[start:end, acenter_c] == 0):
                        output[start:end, acenter_c] = rule["connector"]
                        made_connection = True

        return output if made_connection else None

    @staticmethod
    def external_mapping_crop(grid: np.ndarray) -> np.ndarray | None:
        objects = Components.detect_objects(grid)
        if len(objects) < 2:
            return None

        main = max(objects, key=lambda obj: obj["size"])
        main_cells = set(main["cells"])
        mapping = {}

        for obj in objects:
            if obj is main or obj["size"] != 2:
                continue
            cells = sorted(obj["cells"])
            values = [int(grid[r, c]) for r, c in cells]
            if len(set(values)) != 2:
                continue

            if cells[0][0] == cells[1][0] or cells[0][1] == cells[1][1]:
                mapping[values[1]] = values[0]

        if not mapping:
            return None

        r1, r2, c1, c2 = main["bbox"]
        cropped = grid[r1:r2 + 1, c1:c2 + 1].copy()
        for src, dst in mapping.items():
            cropped[cropped == src] = dst
        return cropped

    @staticmethod
    def fill_seeded_outlines(grid: np.ndarray) -> np.ndarray | None:
        output = grid.copy()
        changed = False
        for color in [int(v) for v in np.unique(grid) if v != 0]:
            color_grid = np.where(grid == color, color, 0)
            for obj in Components.detect_objects(color_grid):
                if obj["size"] == 1:
                    r, c = obj["cells"][0]
                    output[r, c] = 0

        for boundary in [int(v) for v in np.unique(grid) if v != 0]:
            mask = np.where(grid == boundary, boundary, 0)
            for component in Components.connected_components(mask):
                enclosed = LearnedRules._enclosed_cells_for_component(grid, component)
                if not enclosed:
                    continue

                seed_colors = [
                    int(grid[r, c])
                    for r, c in enclosed
                    if grid[r, c] != 0 and grid[r, c] != boundary
                ]
                if not seed_colors:
                    continue

                fill = max(set(seed_colors), key=seed_colors.count)
                for r, c in enclosed:
                    output[r, c] = fill
                    changed = True

        return output if changed else None

    @staticmethod
    def decorate_closed_shapes(grid: np.ndarray) -> np.ndarray | None:
        colors = [int(v) for v in np.unique(grid) if v != 0]
        if len(colors) != 1:
            return None

        source = colors[0]
        output = grid.copy()
        changed = False

        for component in Components.connected_components(grid):
            enclosed = LearnedRules._enclosed_cells_for_component(grid, component)
            if not enclosed:
                continue

            enclosed_set = set(enclosed)
            for r, c in enclosed:
                enclosed_neighbors = sum(
                    (r + dr, c + dc) in enclosed_set
                    for dr in (-1, 0, 1)
                    for dc in (-1, 0, 1)
                    if not (dr == 0 and dc == 0)
                )
                if enclosed_neighbors < 8:
                    output[r, c] = 3
                    changed = True

            for r, c in component["cells"]:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and output[nr, nc] == 0:
                            output[nr, nc] = 2

        output[grid == source] = source
        return output if changed else grid.copy()

    @staticmethod
    def seed_triangle_lattice(grid: np.ndarray) -> np.ndarray | None:
        if grid.shape[0] != 1:
            return None

        positions = np.argwhere(grid != 0)
        if len(positions) != 1:
            return None

        n = grid.shape[1]
        color = int(grid[0, int(positions[0][1])])
        center = int(positions[0][1])
        if center != n // 2:
            return None

        output = np.zeros((n, n), dtype=int)
        for r in range(center + 1):
            output[r, center - r] = color
            output[r, center + r] = color

        fill = 1
        offset = center % 4
        for r in range(n):
            left = center - r if r <= center else -1
            right = center + r if r <= center else n
            start = (r + offset) % 4
            for c in range(start, n, 4):
                if left < c < right and output[r, c] == 0:
                    output[r, c] = fill

        return output

    @staticmethod
    def nonoverlap_overlay_sections(grid: np.ndarray) -> np.ndarray | None:
        axis, idx = SectionRules.find_separator(grid)
        if axis is None:
            return None

        a, b = SectionRules.split_grid(grid, axis, idx)
        if a.shape != b.shape:
            return None

        if np.any((a != 0) & (b != 0)):
            return a.copy()

        result = a.copy()
        result[b != 0] = b[b != 0]
        return result

    @staticmethod
    def complete_missing_grid_cells(grid: np.ndarray) -> np.ndarray | None:
        sep = LearnedRules._separator_color(grid)
        if sep is None:
            return None

        row_bounds = LearnedRules._bounds_from_cuts(
            grid.shape[0],
            LearnedRules._full_indices(grid, axis=0, color=sep),
        )
        col_bounds = LearnedRules._bounds_from_cuts(
            grid.shape[1],
            LearnedRules._full_indices(grid, axis=1, color=sep),
        )
        output = grid.copy()
        changed = False

        for r in range(len(row_bounds) - 1):
            for c in range(len(col_bounds) - 1):
                cells = []
                for dr, dc in ((0, 0), (0, 1), (1, 0), (1, 1)):
                    rr, cc = r + dr, c + dc
                    r1, r2 = row_bounds[rr]
                    c1, c2 = col_bounds[cc]
                    region = grid[r1:r2, c1:c2]
                    vals = [int(v) for v in np.unique(region) if v != 0 and int(v) != sep]
                    cells.append((rr, cc, vals[0] if len(vals) == 1 else None, region))

                colors = [cell[2] for cell in cells if cell[2] is not None]
                if len(colors) != 3 or len(set(colors)) != 1:
                    continue

                missing = next(cell for cell in cells if cell[2] is None)
                opposite = next(
                    cell for cell in cells
                    if cell[0] != missing[0] and cell[1] != missing[1]
                )
                mr1, mr2 = row_bounds[missing[0]]
                mc1, mc2 = col_bounds[missing[1]]
                if opposite[3].shape != (mr2 - mr1, mc2 - mc1):
                    continue

                fill = np.rot90(opposite[3], 2)
                output[mr1:mr2, mc1:mc2] = np.where(fill == sep, sep, fill)
                changed = True

        return output if changed else None

    @staticmethod
    def complete_frame_symmetry(grid: np.ndarray) -> np.ndarray | None:
        output = grid.copy()
        changed = False

        for frame_color in [int(v) for v in np.unique(grid) if v != 0]:
            frame_grid = np.where(grid == frame_color, frame_color, 0)
            for component in Components.connected_components(frame_grid):
                if component["size"] < 4:
                    continue

                r1, r2, c1, c2 = component["bbox"]
                region = grid[r1:r2 + 1, c1:c2 + 1]
                markers = [
                    (r1 + int(r), c1 + int(c), int(region[r, c]))
                    for r, c in np.argwhere((region != 0) & (region != frame_color))
                ]
                if not markers:
                    continue

                for r, c, color in markers:
                    for nr, nc in ((r1 + r2 - r, c), (r, c1 + c2 - c)):
                        if (
                            0 <= nr < grid.shape[0] and
                            0 <= nc < grid.shape[1] and
                            output[nr, nc] == 0
                        ):
                            output[nr, nc] = color
                            changed = True

        return output if changed else None

    @staticmethod
    def pack_objects_into_floor_gaps(grid: np.ndarray) -> np.ndarray | None:
        floor_vals = [int(v) for v in np.unique(grid[-1]) if v != 0]
        if len(floor_vals) != 1 or not np.all(grid[-1] == floor_vals[0]):
            return None

        floor = floor_vals[0]
        support_row = grid.shape[0] - 2
        gaps = []
        c = 0
        while c < grid.shape[1]:
            if grid[support_row, c] != 0:
                c += 1
                continue
            start = c
            while c < grid.shape[1] and grid[support_row, c] == 0:
                c += 1
            gaps.append((start, c - 1, c - start))

        if not gaps:
            return None

        movable = np.where((grid != 0) & (grid != floor), grid, 0)
        objects = LearnedRules._eight_connected_objects(movable)
        if len(objects) != len(gaps):
            return None

        gaps.sort(key=lambda item: (item[2], item[0]))
        objects.sort(key=lambda obj: (len(obj["cells"]), obj["bbox"][2]))
        output = np.where(grid == floor, floor, 0)

        for obj, (c1, c2, width) in zip(objects, gaps):
            area = len(obj["cells"])
            if area % width != 0:
                return None
            height = area // width
            top = support_row - height + 1
            if top < 0:
                return None
            output[top:support_row + 1, c1:c2 + 1] = obj["color"]

        return output

    @staticmethod
    def _separator_color(grid: np.ndarray) -> int | None:
        candidates = []
        for color in [int(v) for v in np.unique(grid) if v != 0]:
            rows = LearnedRules._full_indices(grid, axis=0, color=color)
            cols = LearnedRules._full_indices(grid, axis=1, color=color)
            if rows or cols:
                candidates.append((len(rows) + len(cols), color))
        if not candidates:
            return None
        return max(candidates)[1]

    @staticmethod
    def _full_indices(grid: np.ndarray, axis: int, color: int) -> list[int]:
        indices = []
        length = grid.shape[0] if axis == 0 else grid.shape[1]
        for idx in range(length):
            values = grid[idx, :] if axis == 0 else grid[:, idx]
            if np.all(values == color):
                indices.append(idx)
        return indices

    @staticmethod
    def _bounds_from_cuts(size: int, cuts: list[int]) -> list[tuple[int, int]]:
        bounds = []
        prev = 0
        for cut in cuts + [size]:
            if prev < cut:
                bounds.append((prev, cut))
            prev = cut + 1
        return bounds

    @staticmethod
    def _outside_mask_with_walls(grid: np.ndarray, walls: set[int]) -> np.ndarray:
        rows, cols = grid.shape
        outside = np.zeros((rows, cols), dtype=bool)
        stack = []

        for r in range(rows):
            for c in (0, cols - 1):
                if int(grid[r, c]) not in walls:
                    stack.append((r, c))
        for c in range(cols):
            for r in (0, rows - 1):
                if int(grid[r, c]) not in walls:
                    stack.append((r, c))

        while stack:
            r, c = stack.pop()
            if outside[r, c] or int(grid[r, c]) in walls:
                continue
            outside[r, c] = True
            for dr, dc in Components.NEIGHBORS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    stack.append((nr, nc))
        return outside

    @staticmethod
    def _eight_connected_objects(grid: np.ndarray) -> list[dict]:
        rows, cols = grid.shape
        visited = np.zeros((rows, cols), dtype=bool)
        objects = []
        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == 0 or visited[r, c]:
                    continue

                color = int(grid[r, c])
                stack = [(r, c)]
                visited[r, c] = True
                cells = []

                while stack:
                    cr, cc = stack.pop()
                    cells.append((cr, cc))
                    for dr, dc in neighbors:
                        nr, nc = cr + dr, cc + dc
                        if (
                            0 <= nr < rows and 0 <= nc < cols and
                            not visited[nr, nc] and int(grid[nr, nc]) == color
                        ):
                            visited[nr, nc] = True
                            stack.append((nr, nc))

                objects.append({
                    "color": color,
                    "cells": cells,
                    "bbox": Components.cells_bounding_box(cells),
                })

        return objects

    @staticmethod
    def _enclosed_cells_for_component(grid: np.ndarray, component: dict) -> list[tuple[int, int]]:
        r1, r2, c1, c2 = component["bbox"]
        r1 = max(0, r1 - 1)
        r2 = min(grid.shape[0] - 1, r2 + 1)
        c1 = max(0, c1 - 1)
        c2 = min(grid.shape[1] - 1, c2 + 1)
        walls = set(component["cells"])
        outside = set()
        stack = []

        for r in range(r1, r2 + 1):
            stack.extend([(r, c1), (r, c2)])
        for c in range(c1, c2 + 1):
            stack.extend([(r1, c), (r2, c)])

        while stack:
            r, c = stack.pop()
            if (r, c) in outside or (r, c) in walls:
                continue
            outside.add((r, c))
            for dr, dc in Components.NEIGHBORS:
                nr, nc = r + dr, c + dc
                if r1 <= nr <= r2 and c1 <= nc <= c2:
                    stack.append((nr, nc))

        enclosed = []
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                if (r, c) not in walls and (r, c) not in outside:
                    enclosed.append((r, c))
        return enclosed

    @staticmethod
    def _outside_zero_mask(grid: np.ndarray) -> np.ndarray:
        rows, cols = grid.shape
        outside = np.zeros((rows, cols), dtype=bool)
        stack = []

        for r in range(rows):
            for c in (0, cols - 1):
                if grid[r, c] == 0:
                    stack.append((r, c))
        for c in range(cols):
            for r in (0, rows - 1):
                if grid[r, c] == 0:
                    stack.append((r, c))

        while stack:
            r, c = stack.pop()
            if outside[r, c]:
                continue
            outside[r, c] = True
            for dr, dc in Components.NEIGHBORS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0 and not outside[nr, nc]:
                    stack.append((nr, nc))

        return outside

    @staticmethod
    def _container_ray_distances(
        grid: np.ndarray,
        cells: list[tuple[int, int]],
        container: int,
    ) -> dict[str, float]:
        rows, cols = grid.shape
        distances = {
            "top": float("inf"),
            "bottom": float("inf"),
            "left": float("inf"),
            "right": float("inf"),
        }

        for r, c in cells:
            for nr in range(r - 1, -1, -1):
                if grid[nr, c] == container:
                    distances["top"] = min(distances["top"], r - nr)
                    break
            for nr in range(r + 1, rows):
                if grid[nr, c] == container:
                    distances["bottom"] = min(distances["bottom"], nr - r)
                    break
            for nc in range(c - 1, -1, -1):
                if grid[r, nc] == container:
                    distances["left"] = min(distances["left"], c - nc)
                    break
            for nc in range(c + 1, cols):
                if grid[r, nc] == container:
                    distances["right"] = min(distances["right"], nc - c)
                    break

        return distances

    @staticmethod
    def _connector_positions(start: int, end: int) -> list[int]:
        if start > end:
            return []

        length = end - start + 1
        if length % 2 == 1:
            return list(range(start, end + 1, 2))

        middle = (start + end) / 2
        left = []
        pos = start
        while pos < middle:
            left.append(pos)
            pos += 2

        right = []
        pos = end
        while pos > middle:
            right.append(pos)
            pos -= 2

        return sorted(set(left + right))

    @staticmethod
    def _complex_recolor_matches(
        inp: np.ndarray,
        out: np.ndarray,
        source: int,
        target: int,
    ) -> bool:
        expected = LearnedRules.recolor_complex_components(
            inp,
            {"source": source, "target": target},
        )
        return expected is not None and np.array_equal(expected, out)

    @staticmethod
    def _is_complex_component(component: dict, shape: tuple[int, int]) -> bool:
        rows, cols = shape
        cell_set = set(component["cells"])

        for r, c in cell_set:
            if (
                (r + 1, c) in cell_set and
                (r, c + 1) in cell_set and
                (r + 1, c + 1) in cell_set
            ):
                return True

        r1, r2, c1, c2 = component["bbox"]
        outside = set()
        stack = []

        for r in range(r1, r2 + 1):
            for c in (c1, c2):
                if (r, c) not in cell_set:
                    stack.append((r, c))
        for c in range(c1, c2 + 1):
            for r in (r1, r2):
                if (r, c) not in cell_set:
                    stack.append((r, c))

        while stack:
            r, c = stack.pop()
            if (r, c) in outside or (r, c) in cell_set:
                continue
            outside.add((r, c))
            for dr, dc in Components.NEIGHBORS:
                nr, nc = r + dr, c + dc
                if r1 <= nr <= r2 and c1 <= nc <= c2:
                    stack.append((nr, nc))

        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                if (r, c) not in cell_set and (r, c) not in outside:
                    return True

        return False


def print_grid(grid: np.ndarray):
    for row in grid:
        print(" ".join(map(str, row)))
    print()


# Public function wrappers kept for ArcAgent and future milestones.
rotate90 = GridOps.rotate90
rotate180 = GridOps.rotate180
rotate270 = GridOps.rotate270
flip_horizontal = GridOps.flip_horizontal
flip_vertical = GridOps.flip_vertical
transpose = GridOps.transpose
same_shape = GridOps.same_shape
same_content = GridOps.same_content
get_nonzero_positions = GridOps.nonzero_positions
get_bounding_box = GridOps.bounding_box
crop_bounding_box = GridOps.crop_bounding_box
create_empty_grid = GridOps.create_empty

get_unique_colors = ColorOps.unique_colors
get_nonzero_colors = ColorOps.nonzero_colors
replace_color = ColorOps.replace_color
count_color = ColorOps.count_color
count_nonzero = ColorOps.count_nonzero

detect_objects = Components.detect_objects
connected_components = Components.connected_components
get_cells_bounding_box = Components.cells_bounding_box
extract_object = Components.extract_object
find_connected_blobs = Components.connected_blobs

find_separator = SectionRules.find_separator
detect_two_section_rule = SectionRules.detect
apply_two_section_rule = SectionRules.apply

learn_color_mapping = ColorRules.learn_mapping
apply_color_mapping = ColorRules.apply_mapping
learn_masked_color_rule = ColorRules.learn_masked_color
apply_masked_color_rule = ColorRules.apply_masked_color

make_hollow = ShapeRules.make_hollow
construct_spiral_rectangle = ShapeRules.construct_spiral_rectangle
draw_border = ShapeRules.draw_border
invert_nested_colors = ShapeRules.invert_nested_colors
expand_dots_to_blocks = ShapeRules.expand_dots_to_blocks
mirror_bottom_half = ShapeRules.mirror_bottom_half
draw_x_diagonals = ShapeRules.draw_x_diagonals
pull_cells_to_matching_border = ShapeRules.pull_cells_to_matching_border

is_horizontal_symmetric = SymmetryRules.is_horizontal_symmetric
is_vertical_symmetric = SymmetryRules.is_vertical_symmetric

fill_matching_edge_rows = LearnedRules.fill_matching_edge_rows
extend_marker_ray = LearnedRules.extend_marker_ray
crop_between_corner_markers = LearnedRules.crop_between_corner_markers
learn_diagonal_block_rays = LearnedRules.learn_diagonal_block_rays
diagonal_block_rays = LearnedRules.diagonal_block_rays
mirror_expand_2x2 = LearnedRules.mirror_expand_2x2
overlay_split_sections = LearnedRules.overlay_split_sections
vertical_histogram = LearnedRules.vertical_histogram
growing_prefix_staircase = LearnedRules.growing_prefix_staircase
overlay_panels = LearnedRules.overlay_panels
learn_dot_move = LearnedRules.learn_dot_move
move_dot_toward_anchor = LearnedRules.move_dot_toward_anchor
learn_region_fill = LearnedRules.learn_region_fill
fill_regions = LearnedRules.fill_regions
learn_complex_component_recolor = LearnedRules.learn_complex_component_recolor
recolor_complex_components = LearnedRules.recolor_complex_components
move_marks_outside_container = LearnedRules.move_marks_outside_container
learn_corner_dot_expansion = LearnedRules.learn_corner_dot_expansion
expand_corner_dots = LearnedRules.expand_corner_dots
learn_equal_half_rule = LearnedRules.learn_equal_half_rule
apply_equal_half_rule = LearnedRules.apply_equal_half_rule
learn_straight_path = LearnedRules.learn_straight_path
draw_straight_path = LearnedRules.draw_straight_path
learn_point_connector = LearnedRules.learn_point_connector
connect_points = LearnedRules.connect_points
learn_repeated_mirror_tile = LearnedRules.learn_repeated_mirror_tile
apply_repeated_mirror_tile = LearnedRules.apply_repeated_mirror_tile
block_frequency_histogram = LearnedRules.block_frequency_histogram
pack_enclosed_markers = LearnedRules.pack_enclosed_markers
drop_objects_to_floor = LearnedRules.drop_objects_to_floor
learn_aligned_object_connector = LearnedRules.learn_aligned_object_connector
connect_aligned_objects = LearnedRules.connect_aligned_objects
external_mapping_crop = LearnedRules.external_mapping_crop
fill_seeded_outlines = LearnedRules.fill_seeded_outlines
decorate_closed_shapes = LearnedRules.decorate_closed_shapes
seed_triangle_lattice = LearnedRules.seed_triangle_lattice
nonoverlap_overlay_sections = LearnedRules.nonoverlap_overlay_sections
complete_missing_grid_cells = LearnedRules.complete_missing_grid_cells
complete_frame_symmetry = LearnedRules.complete_frame_symmetry
pack_objects_into_floor_gaps = LearnedRules.pack_objects_into_floor_gaps
