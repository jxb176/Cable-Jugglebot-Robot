#!/usr/bin/env python3
"""Launch a standalone B-spline geometry sandbox."""

from __future__ import annotations

import argparse
import math
import tkinter as tk
from tkinter import ttk

import numpy as np

from jugglebot.patterns.bspline_sandbox import BSplineSandboxModel
from jugglebot.patterns.model import ValidationError


_BACKGROUND = "#F6F0E4"
_PANEL = "#FFFDF7"
_GRID_MINOR = "#E6DDCF"
_GRID_MAJOR = "#CDBEA9"
_AXIS = "#8E7C66"
_CURVE = "#0F766E"
_POLYGON = "#A16207"
_ENDPOINT = "#1D4ED8"
_INTERIOR = "#B45309"
_HANDLE = "#DC2626"
_HANDLE_POINT = "#FB7185"
_TEXT = "#1F2937"
_CANVAS_BG = "#FFFCF5"
_DRAG_RADIUS_PX = 12.0


class BSplineSandboxApp:
    """Minimal GUI for sandboxing B-spline geometry."""

    def __init__(self, root: tk.Tk, *, degree: int = 3, control_count: int = 6) -> None:
        self.root = root
        self.model = BSplineSandboxModel()
        self.model.set_degree(degree)
        self.model.set_control_count(control_count)

        self.degree_var = tk.StringVar(value=str(self.model.degree))
        self.control_count_var = tk.StringVar(value=str(self.model.control_count))
        self.samples_var = tk.StringVar(value="256")
        self.status_var = tk.StringVar(value="Left-drag points. Right-drag to pan. Mouse wheel to zoom.")
        self.arc_length_var = tk.StringVar()
        self.start_tangent_var = tk.StringVar()
        self.end_tangent_var = tk.StringVar()
        self.knots_var = tk.StringVar()

        self.view_center = np.zeros(2, dtype=float)
        self.view_scale = 240.0
        self._drag_target: dict[str, object] | None = None
        self._pan_anchor: tuple[int, int] | None = None
        self._pan_center_at_press: np.ndarray | None = None

        self.root.title("Jugglebot B-Spline Sandbox")
        self.root.geometry("1440x920")
        self.root.configure(background=_BACKGROUND)

        self.style = ttk.Style(self.root)
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass
        self.style.configure("Sandbox.TFrame", background=_BACKGROUND)
        self.style.configure("SandboxPanel.TFrame", background=_PANEL)
        self.style.configure("Sandbox.TLabelframe", background=_PANEL)
        self.style.configure("Sandbox.TLabelframe.Label", background=_PANEL, foreground=_TEXT)
        self.style.configure("Sandbox.TLabel", background=_PANEL, foreground=_TEXT)

        self._build_layout()
        self._bind_canvas_events()
        self.root.after(50, self._fit_view)
        self._draw_scene()

    def _build_layout(self) -> None:
        main = ttk.Frame(self.root, style="Sandbox.TFrame")
        main.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        sidebar = ttk.Frame(main, style="SandboxPanel.TFrame", width=320)
        sidebar.grid(row=0, column=0, sticky="nsw", padx=(0, 12))
        sidebar.grid_propagate(False)
        sidebar.columnconfigure(0, weight=1)

        ttk.Label(
            sidebar,
            text="B-Spline Sandbox",
            style="Sandbox.TLabel",
            font=("TkDefaultFont", 14, "bold"),
        ).grid(row=0, column=0, sticky="w", padx=14, pady=(14, 4))
        ttk.Label(
            sidebar,
            text="Geometry-only editor using the repo's De Boor evaluator.",
            style="Sandbox.TLabel",
            wraplength=280,
            justify=tk.LEFT,
        ).grid(row=1, column=0, sticky="w", padx=14, pady=(0, 14))

        controls = ttk.LabelFrame(sidebar, text="Controls", style="Sandbox.TLabelframe")
        controls.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 10))
        controls.columnconfigure(1, weight=1)

        self._add_spinbox_row(controls, "Degree", self.degree_var, 0, self._apply_degree)
        self._add_spinbox_row(controls, "Control Points", self.control_count_var, 1, self._apply_control_count)
        self._add_spinbox_row(controls, "Curve Samples", self.samples_var, 2, self._apply_samples)

        actions = ttk.Frame(controls, style="SandboxPanel.TFrame")
        actions.grid(row=3, column=0, columnspan=2, sticky="ew", padx=10, pady=(8, 10))
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)
        ttk.Button(actions, text="Reset Default", command=self._reset_default).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(actions, text="Reset Interior", command=self._reset_interior).grid(row=0, column=1, sticky="ew", padx=(4, 0))
        ttk.Button(actions, text="Fit View", command=self._fit_view).grid(row=1, column=0, sticky="ew", padx=(0, 4), pady=(8, 0))
        ttk.Button(actions, text="Straighten", command=self._straighten).grid(row=1, column=1, sticky="ew", padx=(4, 0), pady=(8, 0))

        metrics = ttk.LabelFrame(sidebar, text="Metrics", style="Sandbox.TLabelframe")
        metrics.grid(row=3, column=0, sticky="ew", padx=12, pady=(0, 10))
        metrics.columnconfigure(0, weight=1)
        for row, variable in enumerate(
            (self.arc_length_var, self.start_tangent_var, self.end_tangent_var, self.knots_var)
        ):
            ttk.Label(
                metrics,
                textvariable=variable,
                style="Sandbox.TLabel",
                wraplength=280,
                justify=tk.LEFT,
            ).grid(row=row, column=0, sticky="w", padx=10, pady=(8 if row == 0 else 2, 2))

        instructions = ttk.LabelFrame(sidebar, text="Interactions", style="Sandbox.TLabelframe")
        instructions.grid(row=4, column=0, sticky="ew", padx=12, pady=(0, 10))
        ttk.Label(
            instructions,
            text=(
                "Blue circles are endpoints.\n"
                "Red arrows are endpoint tangent vectors.\n"
                "Amber circles are interior control points.\n"
                "Pink square near the end is the actual end-adjacent control point.\n"
                "Planner UI currently only exposes degrees 3 and 5, but this sandbox lets you probe more."
            ),
            style="Sandbox.TLabel",
            wraplength=280,
            justify=tk.LEFT,
        ).grid(row=0, column=0, sticky="w", padx=10, pady=10)

        ttk.Label(
            sidebar,
            textvariable=self.status_var,
            style="Sandbox.TLabel",
            wraplength=280,
            justify=tk.LEFT,
        ).grid(row=5, column=0, sticky="sw", padx=14, pady=(4, 14))

        canvas_frame = ttk.Frame(main, style="SandboxPanel.TFrame")
        canvas_frame.grid(row=0, column=1, sticky="nsew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            canvas_frame,
            background=_CANVAS_BG,
            highlightthickness=0,
            relief=tk.FLAT,
            cursor="crosshair",
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")

    def _add_spinbox_row(
        self,
        parent: ttk.LabelFrame,
        label: str,
        variable: tk.StringVar,
        row: int,
        callback,
    ) -> None:
        ttk.Label(parent, text=label, style="Sandbox.TLabel").grid(row=row, column=0, sticky="w", padx=10, pady=8)
        spin = ttk.Spinbox(parent, textvariable=variable, width=10, command=callback)
        spin.grid(row=row, column=1, sticky="ew", padx=10, pady=8)
        spin.bind("<Return>", lambda _event: callback())
        spin.bind("<FocusOut>", lambda _event: callback())

    def _bind_canvas_events(self) -> None:
        self.canvas.bind("<Configure>", lambda _event: self._draw_scene())
        self.canvas.bind("<ButtonPress-1>", self._on_left_press)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_release)
        self.canvas.bind("<ButtonPress-3>", self._on_pan_press)
        self.canvas.bind("<B3-Motion>", self._on_pan_drag)
        self.canvas.bind("<ButtonRelease-3>", self._on_pan_release)
        self.canvas.bind("<ButtonPress-2>", self._on_pan_press)
        self.canvas.bind("<B2-Motion>", self._on_pan_drag)
        self.canvas.bind("<ButtonRelease-2>", self._on_pan_release)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", lambda event: self._zoom_at(1.1, event.x, event.y))
        self.canvas.bind("<Button-5>", lambda event: self._zoom_at(1.0 / 1.1, event.x, event.y))

    def _apply_degree(self) -> None:
        try:
            degree = int(self.degree_var.get().strip())
            self.model.set_degree(degree)
        except (ValueError, ValidationError) as exc:
            self.status_var.set(str(exc))
            self.degree_var.set(str(self.model.degree))
            return
        self.control_count_var.set(str(self.model.control_count))
        self.status_var.set(f"Degree set to {self.model.degree}.")
        self._draw_scene()

    def _apply_control_count(self) -> None:
        try:
            count = int(self.control_count_var.get().strip())
            self.model.set_control_count(count)
        except (ValueError, ValidationError) as exc:
            self.status_var.set(str(exc))
            self.control_count_var.set(str(self.model.control_count))
            return
        self.control_count_var.set(str(self.model.control_count))
        self.status_var.set(f"Control count set to {self.model.control_count}.")
        self._draw_scene()

    def _apply_samples(self) -> None:
        try:
            samples = self._sample_count()
        except ValueError as exc:
            self.status_var.set(str(exc))
            self.samples_var.set("256")
            return
        self.samples_var.set(str(samples))
        self._draw_scene()

    def _sample_count(self) -> int:
        return max(32, int(self.samples_var.get().strip()))

    def _reset_default(self) -> None:
        self.model.reset_default()
        self.degree_var.set(str(self.model.degree))
        self.control_count_var.set(str(self.model.control_count))
        self.status_var.set("Reset to the default sandbox curve.")
        self._fit_view()

    def _reset_interior(self) -> None:
        self.model.reset_interior_control_points()
        self.status_var.set("Interior control points regenerated from the boundary handles.")
        self._draw_scene()

    def _straighten(self) -> None:
        start = self.model.start_point
        end = self.model.end_point
        chord = end - start
        self.model.move_start_tangent_tip(start + 0.25 * chord)
        self.model.move_end_handle(end - 0.25 * chord)
        self.model.reset_interior_control_points()
        self.status_var.set("Boundary tangents aligned to the endpoint chord.")
        self._draw_scene()

    def _on_left_press(self, event: tk.Event[tk.Canvas]) -> None:
        self._drag_target = self._pick_drag_target(event.x, event.y)
        if self._drag_target is None:
            self.status_var.set("No draggable target under the cursor.")
            return
        self._update_drag_target(event.x, event.y)

    def _on_left_drag(self, event: tk.Event[tk.Canvas]) -> None:
        if self._drag_target is None:
            return
        self._update_drag_target(event.x, event.y)

    def _on_left_release(self, _event: tk.Event[tk.Canvas]) -> None:
        self._drag_target = None

    def _on_pan_press(self, event: tk.Event[tk.Canvas]) -> None:
        self._pan_anchor = (event.x, event.y)
        self._pan_center_at_press = self.view_center.copy()

    def _on_pan_drag(self, event: tk.Event[tk.Canvas]) -> None:
        if self._pan_anchor is None or self._pan_center_at_press is None:
            return
        dx = event.x - self._pan_anchor[0]
        dy = event.y - self._pan_anchor[1]
        self.view_center = self._pan_center_at_press + np.array([-dx / self.view_scale, dy / self.view_scale], dtype=float)
        self._draw_scene()

    def _on_pan_release(self, _event: tk.Event[tk.Canvas]) -> None:
        self._pan_anchor = None
        self._pan_center_at_press = None

    def _on_mousewheel(self, event: tk.Event[tk.Canvas]) -> None:
        factor = 1.1 if event.delta > 0 else 1.0 / 1.1
        self._zoom_at(factor, event.x, event.y)

    def _zoom_at(self, factor: float, x: int, y: int) -> None:
        factor = float(max(0.2, min(5.0, factor)))
        before = self._canvas_to_world(x, y)
        self.view_scale = float(max(40.0, min(1600.0, self.view_scale * factor)))
        after = self._canvas_to_world(x, y)
        self.view_center = self.view_center + (before - after)
        self._draw_scene()

    def _pick_drag_target(self, x: int, y: int) -> dict[str, object] | None:
        best: dict[str, object] | None = None
        best_distance = float("inf")
        for target in self._drag_targets():
            sx, sy = self._world_to_canvas(target["point"])
            distance = math.hypot(x - sx, y - sy)
            if distance <= float(target["radius"]) and distance < best_distance:
                best = target
                best_distance = distance
        return best

    def _drag_targets(self) -> list[dict[str, object]]:
        targets: list[dict[str, object]] = [
            {"kind": "start", "point": self.model.start_point, "radius": 14.0},
            {"kind": "end", "point": self.model.end_point, "radius": 14.0},
            {"kind": "start_tangent", "point": self.model.start_tangent_tip, "radius": 14.0},
            {"kind": "end_tangent", "point": self.model.end_tangent_tip, "radius": 14.0},
            {"kind": "end_handle", "point": self.model.end_handle, "radius": 12.0},
        ]
        for index in range(2, self.model.control_count - 2):
            targets.append({"kind": "control", "point": self.model.control_points[index].copy(), "index": index, "radius": 11.0})
        return targets

    def _update_drag_target(self, x: int, y: int) -> None:
        if self._drag_target is None:
            return
        world = self._canvas_to_world(x, y)
        kind = str(self._drag_target["kind"])
        if kind == "start":
            self.model.move_start_point(world)
            self.status_var.set("Dragging start endpoint.")
        elif kind == "end":
            self.model.move_end_point(world)
            self.status_var.set("Dragging end endpoint.")
        elif kind == "start_tangent":
            self.model.move_start_tangent_tip(world)
            self.status_var.set("Dragging start tangent.")
        elif kind == "end_tangent":
            self.model.move_end_tangent_tip(world)
            self.status_var.set("Dragging end tangent.")
        elif kind == "end_handle":
            self.model.move_end_handle(world)
            self.status_var.set("Dragging end-adjacent control point.")
        elif kind == "control":
            self.model.move_control_point(int(self._drag_target["index"]), world)
            self.status_var.set(f"Dragging interior control point P{int(self._drag_target['index'])}.")
        self._draw_scene()

    def _fit_view(self) -> None:
        width = max(100, int(self.canvas.winfo_width()))
        height = max(100, int(self.canvas.winfo_height()))
        points = [self.model.control_points, np.vstack((self.model.start_tangent_tip, self.model.end_tangent_tip))]
        stacked = np.vstack(points)
        minimum = stacked.min(axis=0)
        maximum = stacked.max(axis=0)
        span = np.maximum(maximum - minimum, np.array([0.4, 0.4], dtype=float))
        self.view_center = 0.5 * (minimum + maximum)
        self.view_scale = float(min((width - 120.0) / span[0], (height - 120.0) / span[1]))
        self.view_scale = float(max(40.0, min(1200.0, self.view_scale)))
        self._draw_scene()

    def _world_to_canvas(self, point: np.ndarray) -> tuple[float, float]:
        width = max(1, int(self.canvas.winfo_width()))
        height = max(1, int(self.canvas.winfo_height()))
        x = width * 0.5 + (float(point[0]) - float(self.view_center[0])) * self.view_scale
        y = height * 0.5 - (float(point[1]) - float(self.view_center[1])) * self.view_scale
        return x, y

    def _canvas_to_world(self, x: float, y: float) -> np.ndarray:
        width = max(1, int(self.canvas.winfo_width()))
        height = max(1, int(self.canvas.winfo_height()))
        world_x = self.view_center[0] + (float(x) - width * 0.5) / self.view_scale
        world_y = self.view_center[1] - (float(y) - height * 0.5) / self.view_scale
        return np.array([world_x, world_y], dtype=float)

    def _grid_step(self) -> float:
        target_world = 90.0 / self.view_scale
        magnitude = 10.0 ** math.floor(math.log10(max(target_world, 1e-6)))
        for factor in (1.0, 2.0, 5.0, 10.0):
            step = factor * magnitude
            if step >= target_world:
                return step
        return 10.0 * magnitude

    def _draw_grid(self) -> None:
        width = max(1, int(self.canvas.winfo_width()))
        height = max(1, int(self.canvas.winfo_height()))
        x_min, y_min = self._canvas_to_world(0, height)
        x_max, y_max = self._canvas_to_world(width, 0)
        step = self._grid_step()
        major_step = step * 5.0

        x = math.floor(x_min / step) * step
        while x <= x_max + step:
            sx, _ = self._world_to_canvas(np.array([x, 0.0], dtype=float))
            is_major = abs((x / major_step) - round(x / major_step)) < 1e-6
            self.canvas.create_line(sx, 0, sx, height, fill=_GRID_MAJOR if is_major else _GRID_MINOR)
            x += step

        y = math.floor(y_min / step) * step
        while y <= y_max + step:
            _, sy = self._world_to_canvas(np.array([0.0, y], dtype=float))
            is_major = abs((y / major_step) - round(y / major_step)) < 1e-6
            self.canvas.create_line(0, sy, width, sy, fill=_GRID_MAJOR if is_major else _GRID_MINOR)
            y += step

        axis_x, axis_y = self._world_to_canvas(np.array([0.0, 0.0], dtype=float))
        self.canvas.create_line(axis_x, 0, axis_x, height, fill=_AXIS, width=2)
        self.canvas.create_line(0, axis_y, width, axis_y, fill=_AXIS, width=2)

    def _draw_scene(self) -> None:
        self.canvas.delete("all")
        self._draw_grid()

        try:
            sample_count = self._sample_count()
        except ValueError:
            sample_count = 256
            self.samples_var.set("256")
        sample = self.model.sample(samples=sample_count)
        curve_pixels = [coord for point in sample.curve for coord in self._world_to_canvas(point)]
        polygon_pixels = [coord for point in self.model.control_points for coord in self._world_to_canvas(point)]
        start = self.model.start_point
        end = self.model.end_point
        start_tip = self.model.start_tangent_tip
        end_tip = self.model.end_tangent_tip

        self.canvas.create_line(*curve_pixels, fill=_CURVE, width=4, smooth=True)
        self.canvas.create_line(*polygon_pixels, fill=_POLYGON, width=2, dash=(6, 4))
        self.canvas.create_line(*self._world_to_canvas(start), *self._world_to_canvas(start_tip), fill=_HANDLE, width=3, arrow=tk.LAST)
        self.canvas.create_line(*self._world_to_canvas(end), *self._world_to_canvas(end_tip), fill=_HANDLE, width=3, arrow=tk.LAST)

        for index, point in enumerate(self.model.control_points):
            x, y = self._world_to_canvas(point)
            if index == 0 or index == self.model.control_count - 1:
                radius = 9
                self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=_ENDPOINT, outline="white", width=2)
            elif index == 1:
                radius = 8
                self.canvas.create_polygon(
                    x,
                    y - radius,
                    x + radius,
                    y,
                    x,
                    y + radius,
                    x - radius,
                    y,
                    fill=_HANDLE,
                    outline="white",
                    width=2,
                )
            elif index == self.model.control_count - 2:
                radius = 7
                self.canvas.create_rectangle(
                    x - radius,
                    y - radius,
                    x + radius,
                    y + radius,
                    fill=_HANDLE_POINT,
                    outline="white",
                    width=2,
                )
            else:
                radius = 7
                self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=_INTERIOR, outline="white", width=2)
            self.canvas.create_text(x + 14, y - 12, text=f"P{index}", fill=_TEXT, font=("TkDefaultFont", 9, "bold"))

        ex, ey = self._world_to_canvas(end_tip)
        self.canvas.create_polygon(
            ex,
            ey - 8,
            ex + 8,
            ey,
            ex,
            ey + 8,
            ex - 8,
            ey,
            fill=_HANDLE,
            outline="white",
            width=2,
        )

        self.arc_length_var.set(f"Arc length: {sample.length:.3f}")
        self.start_tangent_var.set(
            f"Start tangent: angle {self.model.tangent_angle_deg('start'):.1f} deg, length {self.model.tangent_length('start'):.3f}"
        )
        self.end_tangent_var.set(
            f"End tangent: angle {self.model.tangent_angle_deg('end'):.1f} deg, length {self.model.tangent_length('end'):.3f}"
        )
        self.knots_var.set("Knots: " + " ".join(f"{value:.2f}" for value in sample.knots))


def launch_bspline_sandbox(*, degree: int = 3, control_count: int = 6) -> None:
    root = tk.Tk()
    BSplineSandboxApp(root, degree=degree, control_count=control_count)
    root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive B-spline geometry sandbox")
    parser.add_argument("--degree", type=int, default=3, help="Initial B-spline degree")
    parser.add_argument("--control-points", type=int, default=6, help="Initial control point count")
    args = parser.parse_args()
    launch_bspline_sandbox(degree=args.degree, control_count=args.control_points)


if __name__ == "__main__":
    main()
