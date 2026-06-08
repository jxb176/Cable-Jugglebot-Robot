from __future__ import annotations

from jugglebot.controlui.decimation import decimate_xy


def test_decimation_returns_raw_points_when_sample_count_is_small() -> None:
    times = [0.0, 1.0, 2.0, 3.0]
    values = [1.0, 2.0, 3.0, 4.0]

    x_data, y_data = decimate_xy(times, values, pixel_width=10)

    assert x_data == times
    assert y_data == values


def test_decimation_preserves_spike_extrema() -> None:
    times = [float(index) for index in range(100)]
    values = [0.0] * 100
    values[50] = 10.0

    x_data, y_data = decimate_xy(times, values, pixel_width=10)

    assert len(x_data) <= 20
    assert 50.0 in x_data
    assert 10.0 in y_data


def test_decimation_uses_zoom_window_but_retains_full_detail_inside_it() -> None:
    times = [float(index) for index in range(40)]
    values = [float(index) for index in range(40)]

    x_data, y_data = decimate_xy(times, values, x_range=(10.2, 14.8), pixel_width=20)

    assert x_data == [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    assert y_data == [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]


def test_decimation_respects_explicit_max_points_budget() -> None:
    times = [float(index) for index in range(100)]
    values = [float(index) for index in range(100)]

    x_data, y_data = decimate_xy(times, values, max_points=24)

    assert len(x_data) <= 24
    assert len(y_data) <= 24
