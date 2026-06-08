"""Viewport-aware plot decimation utilities."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
import math
from typing import Sequence


DEFAULT_POINTS_PER_PIXEL = 1


def decimate_xy(
    times: Sequence[float],
    values: Sequence[float],
    *,
    x_range: tuple[float, float] | None = None,
    pixel_width: int = 0,
    points_per_pixel: int = DEFAULT_POINTS_PER_PIXEL,
    max_points: int | None = None,
) -> tuple[list[float], list[float]]:
    count = min(len(times), len(values))
    if count <= 2:
        return _materialize(times, 0, count), _materialize(values, 0, count)

    start_idx, end_idx = _visible_index_range(times, count, x_range)
    visible_count = end_idx - start_idx
    if visible_count <= 0:
        return [], []

    if max_points is None:
        max_points = max(2, int(pixel_width) * max(1, int(points_per_pixel)))
    else:
        max_points = max(2, int(max_points))

    if max_points <= 0 or visible_count <= max_points:
        return _materialize(times, start_idx, end_idx), _materialize(values, start_idx, end_idx)

    bucket_count = max(1, min(max_points // 2, visible_count // 2))
    return _minmax_bucket_decimate(times, values, start_idx, end_idx, bucket_count)


def _visible_index_range(
    times: Sequence[float],
    count: int,
    x_range: tuple[float, float] | None,
) -> tuple[int, int]:
    if x_range is None:
        return 0, count

    x_min = float(x_range[0])
    x_max = float(x_range[1])
    if not (math.isfinite(x_min) and math.isfinite(x_max) and x_max > x_min):
        return 0, count

    start_idx = max(0, bisect_left(times, x_min, 0, count) - 1)
    end_idx = min(count, bisect_right(times, x_max, start_idx, count) + 1)
    return start_idx, end_idx


def _minmax_bucket_decimate(
    times: Sequence[float],
    values: Sequence[float],
    start_idx: int,
    end_idx: int,
    bucket_count: int,
) -> tuple[list[float], list[float]]:
    visible_count = end_idx - start_idx
    out_times: list[float] = []
    out_values: list[float] = []
    last_index = -1

    for bucket in range(bucket_count):
        bucket_start = start_idx + (visible_count * bucket) // bucket_count
        bucket_end = start_idx + (visible_count * (bucket + 1)) // bucket_count
        if bucket_end <= bucket_start:
            continue

        min_index = bucket_start
        max_index = bucket_start
        min_value = float(values[bucket_start])
        max_value = min_value

        for idx in range(bucket_start + 1, bucket_end):
            value = float(values[idx])
            if value < min_value:
                min_value = value
                min_index = idx
            if value > max_value:
                max_value = value
                max_index = idx

        if min_index == max_index:
            last_index = _append_point(out_times, out_values, times, values, min_index, last_index)
            continue

        first_index, second_index = sorted((min_index, max_index))
        last_index = _append_point(out_times, out_values, times, values, first_index, last_index)
        last_index = _append_point(out_times, out_values, times, values, second_index, last_index)

    return out_times, out_values


def _append_point(
    out_times: list[float],
    out_values: list[float],
    times: Sequence[float],
    values: Sequence[float],
    index: int,
    last_index: int,
) -> int:
    if index == last_index:
        return last_index
    out_times.append(float(times[index]))
    out_values.append(float(values[index]))
    return index


def _materialize(data: Sequence[float], start_idx: int, end_idx: int) -> list[float]:
    if start_idx == 0 and end_idx == len(data) and isinstance(data, list):
        return data
    return list(data[start_idx:end_idx])
