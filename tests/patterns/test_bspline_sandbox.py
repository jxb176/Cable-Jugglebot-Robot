from __future__ import annotations

import numpy as np

from jugglebot.patterns.bspline_sandbox import BSplineSandboxModel


def test_default_curve_interpolates_endpoints() -> None:
    model = BSplineSandboxModel()

    sample = model.sample(samples=128)

    np.testing.assert_allclose(sample.curve[0], model.start_point, atol=1e-9)
    np.testing.assert_allclose(sample.curve[-1], model.end_point, atol=1e-9)
    assert sample.length > 0.0


def test_degree_increase_expands_control_count_floor() -> None:
    model = BSplineSandboxModel()

    model.set_control_count(4)
    model.set_degree(5)

    assert model.degree == 5
    assert model.control_count == 6


def test_moving_endpoint_keeps_boundary_handle_independent() -> None:
    model = BSplineSandboxModel()
    original_start_handle = model.start_handle.copy()

    model.move_start_point((-1.3, 0.7))

    np.testing.assert_allclose(model.start_handle, original_start_handle, atol=1e-9)


def test_end_tangent_tip_move_updates_end_handle_symmetrically() -> None:
    model = BSplineSandboxModel()
    end = model.end_point
    new_tip = end + np.array([0.6, 0.25], dtype=float)

    model.move_end_tangent_tip(new_tip)

    np.testing.assert_allclose(model.end_tangent_tip, new_tip, atol=1e-9)
    np.testing.assert_allclose(model.end_handle, end - (new_tip - end), atol=1e-9)


def test_control_count_change_preserves_boundary_constraints() -> None:
    model = BSplineSandboxModel()
    boundary = (
        model.start_point.copy(),
        model.start_handle.copy(),
        model.end_handle.copy(),
        model.end_point.copy(),
    )

    model.set_control_count(9)

    assert model.control_count == 9
    np.testing.assert_allclose(model.start_point, boundary[0], atol=1e-9)
    np.testing.assert_allclose(model.start_handle, boundary[1], atol=1e-9)
    np.testing.assert_allclose(model.end_handle, boundary[2], atol=1e-9)
    np.testing.assert_allclose(model.end_point, boundary[3], atol=1e-9)
