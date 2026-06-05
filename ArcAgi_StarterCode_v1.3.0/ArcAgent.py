import numpy as np

from ArcProblem import ArcProblem
from utils import (
    apply_color_mapping,
    apply_equal_half_rule,
    apply_masked_color_rule,
    apply_repeated_mirror_tile,
    apply_two_section_rule,
    block_frequency_histogram,
    complete_frame_symmetry,
    complete_missing_grid_cells,
    connect_aligned_objects,
    connect_points,
    construct_spiral_rectangle,
    crop_bounding_box,
    decorate_closed_shapes,
    detect_two_section_rule,
    diagonal_block_rays,
    draw_x_diagonals,
    draw_straight_path,
    drop_objects_to_floor,
    expand_corner_dots,
    expand_dots_to_blocks,
    extend_marker_ray,
    external_mapping_crop,
    fill_matching_edge_rows,
    fill_regions,
    fill_seeded_outlines,
    flip_horizontal,
    flip_vertical,
    crop_between_corner_markers,
    growing_prefix_staircase,
    invert_nested_colors,
    learn_color_mapping,
    learn_diagonal_block_rays,
    learn_dot_move,
    learn_equal_half_rule,
    learn_complex_component_recolor,
    learn_corner_dot_expansion,
    learn_aligned_object_connector,
    learn_masked_color_rule,
    learn_point_connector,
    learn_region_fill,
    learn_repeated_mirror_tile,
    learn_straight_path,
    make_hollow,
    mirror_expand_2x2,
    mirror_bottom_half,
    move_marks_outside_container,
    move_dot_toward_anchor,
    nonoverlap_overlay_sections,
    overlay_panels,
    overlay_split_sections,
    pack_enclosed_markers,
    pack_objects_into_floor_gaps,
    pull_cells_to_matching_border,
    recolor_complex_components,
    rotate90,
    rotate180,
    rotate270,
    same_content,
    seed_triangle_lattice,
    transpose,
    vertical_histogram,
)


class Rule:
    def __init__(self, name, predictor):
        self.name = name
        self.predict = predictor


class ArcAgent:
    def __init__(self):
        """
        The agent is a small rule engine. Each rule must prove itself against
        every training pair before it is allowed to predict the test output.

        You may add additional variables to this init method. Be aware that it gets called only once
        and then the make_predictions method will get called several times.
        """
        self.rules = [
            Rule("rotate90", self._simple_rule(rotate90)),
            Rule("rotate180", self._simple_rule(rotate180)),
            Rule("rotate270", self._simple_rule(rotate270)),
            Rule("flip_horizontal", self._simple_rule(flip_horizontal)),
            Rule("flip_vertical", self._simple_rule(flip_vertical)),
            Rule("transpose", self._simple_rule(transpose)),
            Rule("crop_bounding_box", self._simple_rule(crop_bounding_box)),
            Rule("hollow_rectangle", self._simple_rule(make_hollow)),
            Rule("spiral_rectangle", self._spiral_rule),
            Rule("two_section", self._two_section_rule),
            Rule("invert_nested_colors", self._simple_rule(invert_nested_colors)),
            Rule("color_substitution", self._color_substitution_rule),
            Rule("masked_color", self._masked_color_rule),
            Rule("expand_dots_to_blocks", self._expand_dots_rule),
            Rule("mirror_bottom_half", self._simple_rule(mirror_bottom_half)),
            Rule("pull_cells_to_matching_border", self._simple_rule(pull_cells_to_matching_border)),
            Rule("draw_x_diagonals", self._simple_rule(draw_x_diagonals)),
            Rule("fill_matching_edge_rows", self._simple_rule(fill_matching_edge_rows)),
            Rule("extend_marker_ray", self._simple_rule(extend_marker_ray)),
            Rule("crop_between_corner_markers", self._simple_rule(crop_between_corner_markers)),
            Rule("diagonal_block_rays", self._diagonal_block_ray_rule),
            Rule("mirror_expand_2x2", self._simple_rule(mirror_expand_2x2)),
            Rule("overlay_split_sections", self._simple_rule(overlay_split_sections)),
            Rule("vertical_histogram", self._simple_rule(vertical_histogram)),
            Rule("growing_prefix_staircase", self._simple_rule(growing_prefix_staircase)),
            Rule("overlay_panels", self._simple_rule(overlay_panels)),
            Rule("move_dot_toward_anchor", self._dot_move_rule),
            Rule("fill_regions", self._region_fill_rule),
            Rule("recolor_complex_components", self._complex_component_recolor_rule),
            Rule("move_marks_outside_container", self._simple_rule(move_marks_outside_container)),
            Rule("expand_corner_dots", self._corner_dot_expansion_rule),
            Rule("equal_half_rule", self._equal_half_rule),
            Rule("straight_path", self._straight_path_rule),
            Rule("point_connector", self._point_connector_rule),
            Rule("repeated_mirror_tile", self._repeated_mirror_tile_rule),
            Rule("block_frequency_histogram", self._simple_rule(block_frequency_histogram)),
            Rule("pack_enclosed_markers", self._pack_enclosed_markers_rule),
            Rule("drop_objects_to_floor", self._simple_rule(drop_objects_to_floor)),
            Rule("connect_aligned_objects", self._aligned_object_connector_rule),
            Rule("external_mapping_crop", self._simple_rule(external_mapping_crop)),
            Rule("fill_seeded_outlines", self._simple_rule(fill_seeded_outlines)),
            Rule("decorate_closed_shapes", self._simple_rule(decorate_closed_shapes)),
            Rule("seed_triangle_lattice", self._simple_rule(seed_triangle_lattice)),
            Rule("nonoverlap_overlay_sections", self._simple_rule(nonoverlap_overlay_sections)),
            Rule("complete_missing_grid_cells", self._simple_rule(complete_missing_grid_cells)),
            Rule("complete_frame_symmetry", self._simple_rule(complete_frame_symmetry)),
            Rule("pack_objects_into_floor_gaps", self._simple_rule(pack_objects_into_floor_gaps)),
        ]

    def make_predictions(self, arc_problem: ArcProblem) -> list[np.ndarray]:
        train_pairs = arc_problem.training_set()
        test_input = arc_problem.test_set().get_input_data().data()

        for rule in self.rules:
            prediction = rule.predict(train_pairs, test_input)
            if prediction is not None:
                return [prediction]

        return []

    def _simple_rule(self, transform):
        def predictor(train_pairs, test_input):
            return self._predict_if_matches(train_pairs, test_input, transform)

        return predictor

    def _spiral_rule(self, train_pairs, test_input):
        color = self._output_fill_value(train_pairs)
        return self._predict_if_matches(
            train_pairs,
            test_input,
            lambda grid: construct_spiral_rectangle(grid, color),
        )

    def _two_section_rule(self, train_pairs, test_input):
        pairs = self._array_pairs(train_pairs)
        rule, marker = detect_two_section_rule(pairs)
        if rule is None:
            return None

        return self._predict_if_matches(
            train_pairs,
            test_input,
            lambda grid: apply_two_section_rule(grid, rule, marker),
        )

    def _color_substitution_rule(self, train_pairs, test_input):
        mapping = learn_color_mapping(self._array_pairs(train_pairs))
        if mapping is None:
            return None

        return self._predict_if_matches(
            train_pairs,
            test_input,
            lambda grid: apply_color_mapping(grid, mapping),
        )

    def _masked_color_rule(self, train_pairs, test_input):
        rule = learn_masked_color_rule(self._array_pairs(train_pairs))
        if rule is None:
            return None

        return self._predict_if_matches(
            train_pairs,
            test_input,
            lambda grid: apply_masked_color_rule(grid, rule),
        )

    def _expand_dots_rule(self, train_pairs, test_input):
        out_val = self._output_fill_value(train_pairs)
        return self._predict_if_matches(
            train_pairs,
            test_input,
            lambda grid: expand_dots_to_blocks(grid, out_val),
        )

    def _diagonal_block_ray_rule(self, train_pairs, test_input):
        rule = learn_diagonal_block_rays(self._array_pairs(train_pairs))
        if rule is None:
            return None

        return self._predict_if_matches(
            train_pairs,
            test_input,
            lambda grid: diagonal_block_rays(grid, rule),
        )

    def _dot_move_rule(self, train_pairs, test_input):
        rule = learn_dot_move(self._array_pairs(train_pairs))
        if rule is None:
            return None

        return self._predict_if_matches(
            train_pairs,
            test_input,
            lambda grid: move_dot_toward_anchor(grid, rule),
        )

    def _region_fill_rule(self, train_pairs, test_input):
        rule = learn_region_fill(self._array_pairs(train_pairs))
        if rule is None:
            return None

        return self._predict_if_matches(
            train_pairs,
            test_input,
            lambda grid: fill_regions(grid, rule),
        )

    def _complex_component_recolor_rule(self, train_pairs, test_input):
        rule = learn_complex_component_recolor(self._array_pairs(train_pairs))
        if rule is None:
            return None

        return self._predict_if_matches(
            train_pairs,
            test_input,
            lambda grid: recolor_complex_components(grid, rule),
        )

    def _corner_dot_expansion_rule(self, train_pairs, test_input):
        rule = learn_corner_dot_expansion(self._array_pairs(train_pairs))
        if rule is None:
            return None

        return self._predict_if_matches(
            train_pairs,
            test_input,
            lambda grid: expand_corner_dots(grid, rule),
        )

    def _equal_half_rule(self, train_pairs, test_input):
        rule = learn_equal_half_rule(self._array_pairs(train_pairs))
        if rule is None:
            return None

        return self._predict_if_matches(
            train_pairs,
            test_input,
            lambda grid: apply_equal_half_rule(grid, rule),
        )

    def _straight_path_rule(self, train_pairs, test_input):
        rule = learn_straight_path(self._array_pairs(train_pairs))
        if rule is None:
            return None

        return self._predict_if_matches(
            train_pairs,
            test_input,
            lambda grid: draw_straight_path(grid, rule),
        )

    def _point_connector_rule(self, train_pairs, test_input):
        rule = learn_point_connector(self._array_pairs(train_pairs))
        if rule is None:
            return None

        return self._predict_if_matches(
            train_pairs,
            test_input,
            lambda grid: connect_points(grid, rule),
        )

    def _repeated_mirror_tile_rule(self, train_pairs, test_input):
        rule = learn_repeated_mirror_tile(self._array_pairs(train_pairs))
        if rule is None:
            return None

        return self._predict_if_matches(
            train_pairs,
            test_input,
            lambda grid: apply_repeated_mirror_tile(grid, rule),
        )

    def _pack_enclosed_markers_rule(self, train_pairs, test_input):
        shapes = {pair.get_output_data().data().shape for pair in train_pairs}
        if len(shapes) != 1:
            return None

        output_shape = next(iter(shapes))
        return self._predict_if_matches(
            train_pairs,
            test_input,
            lambda grid: pack_enclosed_markers(grid, output_shape),
        )

    def _aligned_object_connector_rule(self, train_pairs, test_input):
        rule = learn_aligned_object_connector(self._array_pairs(train_pairs))
        if rule is None:
            return None

        return self._predict_if_matches(
            train_pairs,
            test_input,
            lambda grid: connect_aligned_objects(grid, rule),
        )

    def _predict_if_matches(self, train_pairs, test_input, transform):
        for inp, out in self._array_pairs(train_pairs):
            predicted = transform(inp)
            if predicted is None or not same_content(predicted, out):
                return None

        return transform(test_input)

    @staticmethod
    def _array_pairs(train_pairs):
        return [
            (pair.get_input_data().data(), pair.get_output_data().data())
            for pair in train_pairs
        ]

    @staticmethod
    def _output_fill_value(train_pairs) -> int:
        out = train_pairs[0].get_output_data().data()
        vals = [int(v) for v in np.unique(out) if v != 0]
        return vals[0] if vals else 1
