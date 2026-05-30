"""Tk desktop editor for unconstrained juggling pattern authoring."""

from __future__ import annotations

from pathlib import Path
import string
import time
import tkinter as tk
from tkinter import filedialog, ttk
from typing import Callable

import numpy as np

from .model import (
    HAND_NAMES,
    HandKeyframe,
    PatternProject,
    ThrowEvent,
    ValidationError,
    build_three_ball_cascade_pattern,
    load_pattern_project,
    save_pattern_project,
)


_BALL_PALETTE = [
    "#D44D3E",
    "#0E7490",
    "#F59E0B",
    "#5B8C5A",
    "#8B5CF6",
    "#D97706",
    "#2563EB",
    "#BE185D",
]
_HAND_COLORS = {"left": "#C2410C", "right": "#1D4ED8"}
_WAYPOINT_HIT_RADIUS_PX = 12.0
_VELOCITY_HANDLE_TIME_SCALE_S = 0.2


class PatternStudioApp:
    """Interactive unconstrained pattern editor."""

    def __init__(self, root: tk.Tk, project: PatternProject, initial_path: Path | None = None) -> None:
        self.root = root
        self.project = project.copy()
        self.current_path = initial_path
        self.current_time = 0.0
        self.playing = False
        self.last_tick = time.perf_counter()
        self.form_source_event_id: str | None = None
        self.hand_point_source_id: str | None = None
        self._suspend_event_slider_callbacks = False
        self._waypoint_hit_targets: list[dict[str, object]] = []
        self._projection_specs: dict[str, dict[str, object]] = {}
        self._drag_waypoint: dict[str, object] | None = None

        self.status_var = tk.StringVar(value="Ready.")
        self.name_var = tk.StringVar()
        self.mode_var = tk.StringVar()
        self.loop_period_var = tk.StringVar()
        self.gravity_var = tk.StringVar()
        self.hand_editor_hand_var = tk.StringVar(value="right")
        self.play_speed_var = tk.DoubleVar(value=1.0)
        self.time_var = tk.DoubleVar(value=0.0)
        self.time_label_var = tk.StringVar(value="t = 0.00 s")

        self.event_vars = {
            "id": tk.StringVar(),
            "ball": tk.StringVar(),
            "throw_hand": tk.StringVar(value="left"),
            "catch_hand": tk.StringVar(value="right"),
            "throw_time": tk.StringVar(),
            "catch_time": tk.StringVar(),
            "catch_velocity_scale": tk.StringVar(value="0.350"),
            "throw_x": tk.StringVar(),
            "throw_y": tk.StringVar(),
            "throw_z": tk.StringVar(),
            "catch_x": tk.StringVar(),
            "catch_y": tk.StringVar(),
            "catch_z": tk.StringVar(),
        }
        self.event_slider_vars = {
            "throw_time": tk.DoubleVar(value=0.0),
            "catch_time": tk.DoubleVar(value=1.0),
            "catch_velocity_scale": tk.DoubleVar(value=0.35),
            "throw_x": tk.DoubleVar(value=0.0),
            "throw_y": tk.DoubleVar(value=0.0),
            "throw_z": tk.DoubleVar(value=1.0),
            "catch_x": tk.DoubleVar(value=0.0),
            "catch_y": tk.DoubleVar(value=0.0),
            "catch_z": tk.DoubleVar(value=1.0),
        }
        self.event_slider_ranges = {
            "throw_time": (0.0, 4.0),
            "catch_time": (0.0, 4.5),
            "catch_velocity_scale": (0.0, 1.5),
            "throw_x": (-1.0, 1.0),
            "throw_y": (-1.0, 1.0),
            "throw_z": (0.0, 2.0),
            "catch_x": (-1.0, 1.0),
            "catch_y": (-1.0, 1.0),
            "catch_z": (0.0, 2.0),
        }
        self.event_slider_value_vars = {key: tk.StringVar(value="0.000") for key in self.event_slider_vars}
        self.event_slider_scales: dict[str, ttk.Scale] = {}
        self.hand_point_vars = {
            "id": tk.StringVar(),
            "time": tk.StringVar(),
            "x": tk.StringVar(),
            "y": tk.StringVar(),
            "z": tk.StringVar(),
            "vx": tk.StringVar(),
            "vy": tk.StringVar(),
            "vz": tk.StringVar(),
            "path_speed": tk.StringVar(),
            "bspline_degree": tk.StringVar(value="3"),
            "bspline_control_points": tk.StringVar(value="6"),
            "spline_to_next": tk.StringVar(value="quintic"),
        }

        self.style = ttk.Style(self.root)
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass
        self.style.configure("Studio.TFrame", background="#F6F0E4")
        self.style.configure("Studio.TLabelframe", background="#F6F0E4")
        self.style.configure("Studio.TLabelframe.Label", background="#F6F0E4", foreground="#1F2937")
        self.style.configure("Studio.TLabel", background="#F6F0E4", foreground="#1F2937")
        self.style.configure("Accent.TButton", padding=(10, 6))

        self.root.configure(background="#F6F0E4")
        self.root.title("Jugglebot Pattern Studio")
        self.root.geometry("1680x980")

        self._build_layout()
        self._load_project_into_controls(select_event_id=self.project.sorted_events()[0].id if self.project.events else None)
        self._draw_scene()
        self.root.after(33, self._tick)

    def _build_layout(self) -> None:
        main = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        left = ttk.Frame(main, style="Studio.TFrame", width=560)
        right = ttk.Frame(main, style="Studio.TFrame")
        main.add(left, weight=0)
        main.add(right, weight=1)

        self._build_editor_panel(left)
        self._build_preview_panel(right)

    def _build_editor_panel(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)

        project_frame = ttk.LabelFrame(parent, text="Project", style="Studio.TLabelframe")
        project_frame.grid(row=0, column=0, sticky="ew", padx=4, pady=(0, 8))
        project_frame.columnconfigure(1, weight=1)

        self._add_labeled_entry(project_frame, "Name", self.name_var, 0)
        self._add_labeled_combobox(project_frame, "Mode", self.mode_var, ("loop", "single_run"), 1)
        self._add_labeled_entry(project_frame, "Loop Period [s]", self.loop_period_var, 2)
        self._add_labeled_entry(project_frame, "Gravity [m/s^2]", self.gravity_var, 3)

        apply_settings = ttk.Button(project_frame, text="Apply Settings", command=self._apply_project_settings, style="Accent.TButton")
        apply_settings.grid(row=4, column=0, columnspan=2, sticky="ew", padx=8, pady=(6, 8))

        files_frame = ttk.LabelFrame(parent, text="Files", style="Studio.TLabelframe")
        files_frame.grid(row=1, column=0, sticky="ew", padx=4, pady=(0, 8))
        for col in range(3):
            files_frame.columnconfigure(col, weight=1)
        ttk.Button(files_frame, text="Load YAML", command=self._load_from_file).grid(row=0, column=0, sticky="ew", padx=6, pady=8)
        ttk.Button(files_frame, text="Save YAML", command=self._save_to_file).grid(row=0, column=1, sticky="ew", padx=6, pady=8)
        ttk.Button(files_frame, text="Reset Sample", command=self._reset_sample).grid(row=0, column=2, sticky="ew", padx=6, pady=8)

        events_frame = ttk.LabelFrame(parent, text="Throws", style="Studio.TLabelframe")
        events_frame.grid(row=2, column=0, sticky="nsew", padx=4, pady=(0, 8))
        parent.rowconfigure(2, weight=1)
        events_frame.columnconfigure(0, weight=1)
        events_frame.rowconfigure(0, weight=1)

        list_frame = ttk.Frame(events_frame, style="Studio.TFrame")
        list_frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=(8, 4))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        self.event_listbox = tk.Listbox(
            list_frame,
            activestyle="none",
            bg="#FFFDF8",
            fg="#111827",
            selectbackground="#D7E8F7",
            selectforeground="#111827",
            selectmode=tk.EXTENDED,
            exportselection=False,
            height=12,
            relief=tk.FLAT,
        )
        self.event_listbox.grid(row=0, column=0, sticky="nsew")
        event_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.event_listbox.yview)
        event_scroll.grid(row=0, column=1, sticky="ns")
        self.event_listbox.configure(yscrollcommand=event_scroll.set)
        self.event_listbox.bind("<<ListboxSelect>>", self._on_event_selected)

        buttons = ttk.Frame(events_frame, style="Studio.TFrame")
        buttons.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=1)
        buttons.columnconfigure(2, weight=1)
        ttk.Button(buttons, text="New Event", command=self._prepare_new_event).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(buttons, text="Apply Event", command=self._apply_event).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(buttons, text="Delete Selected", command=self._delete_selected_events).grid(row=0, column=2, sticky="ew", padx=(4, 0))

        form_frame = ttk.LabelFrame(parent, text="Selected Event", style="Studio.TLabelframe")
        form_frame.grid(row=3, column=0, sticky="ew", padx=4, pady=(0, 8))
        for col in range(4):
            form_frame.columnconfigure(col, weight=1)

        self._add_form_row(form_frame, "Event ID", "id", 0)
        self._add_form_row(form_frame, "Ball", "ball", 1)
        self._add_form_row(form_frame, "Throw Hand", "throw_hand", 2, values=HAND_NAMES)
        self._add_form_row(form_frame, "Catch Hand", "catch_hand", 3, values=HAND_NAMES)
        self._add_form_row(form_frame, "Throw Time [s]", "throw_time", 4)
        self._add_form_row(form_frame, "Catch Time [s]", "catch_time", 5)
        self._add_form_row(form_frame, "Catch Vel Scale", "catch_velocity_scale", 6)
        self._add_vector_row(form_frame, "Throw Pos [m]", "throw", 7)
        self._add_vector_row(form_frame, "Catch Pos [m]", "catch", 8)

        slider_frame = ttk.LabelFrame(parent, text="Live Event Adjust", style="Studio.TLabelframe")
        slider_frame.grid(row=4, column=0, sticky="ew", padx=4, pady=(0, 8))
        for col in range(3):
            slider_frame.columnconfigure(col, weight=1)
        self._add_event_slider_cell(slider_frame, "Throw Time", "throw_time", 0, 0)
        self._add_event_slider_cell(slider_frame, "Catch Time", "catch_time", 0, 1)
        self._add_event_slider_cell(slider_frame, "Catch Vel", "catch_velocity_scale", 0, 2)
        self._add_event_slider_cell(slider_frame, "Throw X", "throw_x", 1, 0)
        self._add_event_slider_cell(slider_frame, "Throw Y", "throw_y", 1, 1)
        self._add_event_slider_cell(slider_frame, "Throw Z", "throw_z", 1, 2)
        self._add_event_slider_cell(slider_frame, "Catch X", "catch_x", 2, 0)
        self._add_event_slider_cell(slider_frame, "Catch Y", "catch_y", 2, 1)
        self._add_event_slider_cell(slider_frame, "Catch Z", "catch_z", 2, 2)

        hand_frame = ttk.LabelFrame(parent, text="Hand Trajectory", style="Studio.TLabelframe")
        hand_frame.grid(row=5, column=0, sticky="ew", padx=4, pady=(0, 8))
        for col in range(4):
            hand_frame.columnconfigure(col, weight=1)

        ttk.Label(hand_frame, text="Hand", style="Studio.TLabel").grid(row=0, column=0, sticky="w", padx=8, pady=4)
        hand_combo = ttk.Combobox(hand_frame, textvariable=self.hand_editor_hand_var, values=HAND_NAMES, state="readonly")
        hand_combo.grid(row=0, column=1, columnspan=3, sticky="ew", padx=8, pady=4)
        hand_combo.bind("<<ComboboxSelected>>", self._on_hand_editor_changed)

        hand_list_frame = ttk.Frame(hand_frame, style="Studio.TFrame")
        hand_list_frame.grid(row=1, column=0, columnspan=4, sticky="nsew", padx=8, pady=(4, 4))
        hand_list_frame.columnconfigure(0, weight=1)
        hand_list_frame.rowconfigure(0, weight=1)
        hand_frame.rowconfigure(1, weight=1)

        self.hand_point_listbox = tk.Listbox(
            hand_list_frame,
            activestyle="none",
            bg="#FFFDF8",
            fg="#111827",
            selectbackground="#D7E8F7",
            selectforeground="#111827",
            selectmode=tk.EXTENDED,
            exportselection=False,
            height=6,
            relief=tk.FLAT,
        )
        self.hand_point_listbox.grid(row=0, column=0, sticky="nsew")
        hand_scroll = ttk.Scrollbar(hand_list_frame, orient=tk.VERTICAL, command=self.hand_point_listbox.yview)
        hand_scroll.grid(row=0, column=1, sticky="ns")
        self.hand_point_listbox.configure(yscrollcommand=hand_scroll.set)
        self.hand_point_listbox.bind("<<ListboxSelect>>", self._on_hand_point_selected)

        hand_buttons = ttk.Frame(hand_frame, style="Studio.TFrame")
        hand_buttons.grid(row=2, column=0, columnspan=4, sticky="ew", padx=8, pady=(0, 4))
        for col in range(3):
            hand_buttons.columnconfigure(col, weight=1)
        ttk.Button(hand_buttons, text="New Point", command=self._prepare_new_hand_point).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(hand_buttons, text="Apply Point", command=self._apply_hand_point).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(hand_buttons, text="Delete Point", command=self._delete_selected_hand_points).grid(row=0, column=2, sticky="ew", padx=(4, 0))

        self._add_hand_point_row(hand_frame, "Point ID", "id", 3)
        self._add_hand_point_row(hand_frame, "Time [s]", "time", 4)
        self._add_hand_point_combobox_row(hand_frame, "Spline To Next", "spline_to_next", ("cubic", "quintic", "bspline"), 5)
        self._add_hand_point_combobox_row(hand_frame, "B-Spline Degree", "bspline_degree", ("3", "5"), 6)
        self._add_hand_point_row(hand_frame, "Control Points", "bspline_control_points", 7)
        self._add_hand_point_row(hand_frame, "Path Speed [m/s]", "path_speed", 8)
        self._add_hand_point_vector_row(hand_frame, "Pos [m]", 9)
        self._add_hand_point_vector_row(hand_frame, "Vel [m/s]", 10, prefix="v")

        status = ttk.Label(parent, textvariable=self.status_var, style="Studio.TLabel", anchor="w", justify=tk.LEFT)
        status.grid(row=6, column=0, sticky="ew", padx=8, pady=(0, 4))

    def _build_preview_panel(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        transport = ttk.LabelFrame(parent, text="Preview", style="Studio.TLabelframe")
        transport.grid(row=0, column=0, sticky="ew", padx=4, pady=(0, 8))
        transport.columnconfigure(1, weight=1)
        transport.columnconfigure(4, weight=1)

        self.play_button = ttk.Button(transport, text="Play", command=self._toggle_playback, style="Accent.TButton")
        self.play_button.grid(row=0, column=0, sticky="w", padx=8, pady=8)

        self.time_scale = ttk.Scale(
            transport,
            orient=tk.HORIZONTAL,
            from_=0.0,
            to=max(0.001, self.project.timeline_duration()),
            variable=self.time_var,
            command=self._on_scrubbed,
        )
        self.time_scale.grid(row=0, column=1, columnspan=3, sticky="ew", padx=8, pady=8)

        speed_label = ttk.Label(transport, text="Speed", style="Studio.TLabel")
        speed_label.grid(row=0, column=4, sticky="e", padx=(8, 4), pady=8)
        speed_scale = ttk.Scale(transport, orient=tk.HORIZONTAL, from_=0.25, to=2.0, variable=self.play_speed_var)
        speed_scale.grid(row=0, column=5, sticky="ew", padx=(0, 8), pady=8)

        time_label = ttk.Label(transport, textvariable=self.time_label_var, style="Studio.TLabel")
        time_label.grid(row=0, column=6, sticky="e", padx=8, pady=8)

        self.canvas = tk.Canvas(parent, background="#FFF8EC", highlightthickness=0, relief=tk.FLAT)
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=4, pady=(0, 4))
        self.canvas.bind("<Configure>", lambda _event: self._draw_scene())
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)

    def _add_labeled_entry(self, parent: ttk.Widget, label: str, variable: tk.StringVar, row: int) -> None:
        ttk.Label(parent, text=label, style="Studio.TLabel").grid(row=row, column=0, sticky="w", padx=8, pady=4)
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", padx=8, pady=4)

    def _add_labeled_combobox(self, parent: ttk.Widget, label: str, variable: tk.StringVar, values: tuple[str, ...], row: int) -> None:
        ttk.Label(parent, text=label, style="Studio.TLabel").grid(row=row, column=0, sticky="w", padx=8, pady=4)
        combo = ttk.Combobox(parent, textvariable=variable, values=values, state="readonly")
        combo.grid(row=row, column=1, sticky="ew", padx=8, pady=4)

    def _add_form_row(self, parent: ttk.Widget, label: str, key: str, row: int, values: tuple[str, ...] | None = None) -> None:
        ttk.Label(parent, text=label, style="Studio.TLabel").grid(row=row, column=0, sticky="w", padx=8, pady=4)
        if values is None:
            widget = ttk.Entry(parent, textvariable=self.event_vars[key])
        else:
            widget = ttk.Combobox(parent, textvariable=self.event_vars[key], values=values, state="readonly")
        widget.grid(row=row, column=1, columnspan=3, sticky="ew", padx=8, pady=4)

    def _add_vector_row(self, parent: ttk.Widget, label: str, prefix: str, row: int) -> None:
        ttk.Label(parent, text=label, style="Studio.TLabel").grid(row=row, column=0, sticky="w", padx=8, pady=4)
        for idx, axis in enumerate(("x", "y", "z"), start=1):
            entry = ttk.Entry(parent, textvariable=self.event_vars[f"{prefix}_{axis}"])
            entry.grid(row=row, column=idx, sticky="ew", padx=4, pady=4)

    def _add_event_slider_cell(self, parent: ttk.Widget, label: str, key: str, row: int, column: int) -> None:
        cell = ttk.Frame(parent, style="Studio.TFrame")
        cell.grid(row=row, column=column, sticky="ew", padx=6, pady=4)
        cell.columnconfigure(0, weight=1)
        ttk.Label(cell, text=label, style="Studio.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(cell, textvariable=self.event_slider_value_vars[key], style="Studio.TLabel").grid(row=0, column=1, sticky="e")
        scale = ttk.Scale(
            cell,
            orient=tk.HORIZONTAL,
            from_=self.event_slider_ranges[key][0],
            to=self.event_slider_ranges[key][1],
            variable=self.event_slider_vars[key],
            command=lambda _value, field=key: self._on_event_slider_changed(field),
        )
        scale.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(2, 0))
        self.event_slider_scales[key] = scale

    def _add_hand_point_row(self, parent: ttk.Widget, label: str, key: str, row: int) -> None:
        ttk.Label(parent, text=label, style="Studio.TLabel").grid(row=row, column=0, sticky="w", padx=8, pady=4)
        widget = ttk.Entry(parent, textvariable=self.hand_point_vars[key])
        widget.grid(row=row, column=1, columnspan=3, sticky="ew", padx=8, pady=4)

    def _add_hand_point_combobox_row(
        self,
        parent: ttk.Widget,
        label: str,
        key: str,
        values: tuple[str, ...],
        row: int,
    ) -> None:
        ttk.Label(parent, text=label, style="Studio.TLabel").grid(row=row, column=0, sticky="w", padx=8, pady=4)
        widget = ttk.Combobox(parent, textvariable=self.hand_point_vars[key], values=values, state="readonly")
        widget.grid(row=row, column=1, columnspan=3, sticky="ew", padx=8, pady=4)

    def _add_hand_point_vector_row(self, parent: ttk.Widget, label: str, row: int, prefix: str = "") -> None:
        ttk.Label(parent, text=label, style="Studio.TLabel").grid(row=row, column=0, sticky="w", padx=8, pady=4)
        for idx, axis in enumerate(("x", "y", "z"), start=1):
            entry = ttk.Entry(parent, textvariable=self.hand_point_vars[f"{prefix}{axis}"])
            entry.grid(row=row, column=idx, sticky="ew", padx=4, pady=4)

    def _load_project_into_controls(self, select_event_id: str | None = None) -> None:
        self.project.validate()
        self.playing = False
        self.play_button.configure(text="Play")
        self.name_var.set(self.project.name)
        self.mode_var.set(self.project.mode)
        self.loop_period_var.set(f"{self.project.loop_period:.3f}")
        self.gravity_var.set(f"{self.project.gravity:.3f}")
        self._populate_event_list(select_event_id=select_event_id)
        self._populate_hand_point_list(select_point_id=None)
        self._sync_time_controls(reset_time=True)
        self._set_status(self._project_summary())

    def _project_summary(self) -> str:
        mode = "looped" if self.project.is_loop else "single run"
        hand_points = sum(len(self.project.sorted_hand_trajectory(hand)) for hand in HAND_NAMES)
        path = f"  File: {self.current_path}" if self.current_path else ""
        return (
            f"{self.project.name}: {len(self.project.events)} events, "
            f"{len(self.project.ball_ids())} balls, {hand_points} hand points, {mode}, "
            f"timeline {self.project.timeline_duration():.2f}s.{path}"
        )

    def _populate_event_list(self, select_event_id: str | None = None) -> None:
        self.event_listbox.delete(0, tk.END)
        self.event_ids: list[str] = []
        for event in self.project.sorted_events():
            self.event_ids.append(event.id)
            label = (
                f"{event.id} | {event.ball} "
                f"{event.throw_hand[0].upper()}->{event.catch_hand[0].upper()} "
                f"{event.throw_time:.2f}s -> {event.catch_time:.2f}s"
            )
            self.event_listbox.insert(tk.END, label)

        if not self.event_ids:
            self.form_source_event_id = None
            self._clear_event_form()
            return

        target_id = select_event_id if select_event_id in self.event_ids else self.event_ids[0]
        index = self.event_ids.index(target_id)
        self.event_listbox.selection_clear(0, tk.END)
        self.event_listbox.selection_set(index)
        self.event_listbox.see(index)
        self._load_event_into_form(self.project.sorted_events()[index])

    def _load_event_into_form(self, event: ThrowEvent) -> None:
        self.form_source_event_id = event.id
        self.event_vars["id"].set(event.id)
        self.event_vars["ball"].set(event.ball)
        self.event_vars["throw_hand"].set(event.throw_hand)
        self.event_vars["catch_hand"].set(event.catch_hand)
        self.event_vars["throw_time"].set(f"{event.throw_time:.3f}")
        self.event_vars["catch_time"].set(f"{event.catch_time:.3f}")
        self.event_vars["catch_velocity_scale"].set(f"{event.catch_velocity_scale:.3f}")
        for axis, value in zip(("x", "y", "z"), event.throw_pos):
            self.event_vars[f"throw_{axis}"].set(f"{value:.3f}")
        for axis, value in zip(("x", "y", "z"), event.catch_pos):
            self.event_vars[f"catch_{axis}"].set(f"{value:.3f}")
        self._sync_event_sliders_from_event(event)

    def _clear_event_form(self) -> None:
        for key, variable in self.event_vars.items():
            if key in {"throw_hand", "catch_hand"}:
                variable.set("left" if key == "throw_hand" else "right")
            else:
                variable.set("")
        self._suspend_event_slider_callbacks = True
        for key, variable in self.event_slider_vars.items():
            variable.set(1.0 if key.endswith("_z") else 0.0)
            self.event_slider_value_vars[key].set(f"{float(variable.get()):.3f}")
        self._suspend_event_slider_callbacks = False

    def _selected_event_ids(self) -> list[str]:
        selection = [int(index) for index in self.event_listbox.curselection()]
        return [self.event_ids[index] for index in selection if 0 <= index < len(self.event_ids)]

    def _sync_event_sliders_from_event(self, event: ThrowEvent) -> None:
        self._configure_event_slider_ranges()
        values = {
            "throw_time": event.throw_time,
            "catch_time": event.catch_time,
            "catch_velocity_scale": event.catch_velocity_scale,
            "throw_x": event.throw_pos[0],
            "throw_y": event.throw_pos[1],
            "throw_z": event.throw_pos[2],
            "catch_x": event.catch_pos[0],
            "catch_y": event.catch_pos[1],
            "catch_z": event.catch_pos[2],
        }
        self._suspend_event_slider_callbacks = True
        for key, value in values.items():
            self.event_slider_vars[key].set(float(value))
            self.event_slider_value_vars[key].set(f"{float(value):.3f}")
        self._suspend_event_slider_callbacks = False

    def _configure_event_slider_ranges(self) -> None:
        duration_max = max(self.project.sequence_end_time(), self.project.loop_period, 1.5)
        scene_bounds = self._scene_bounds()
        self.event_slider_ranges["throw_time"] = (0.0, max(duration_max, 0.25))
        self.event_slider_ranges["catch_time"] = (0.05, max(duration_max + 1.0, 0.5))
        self.event_slider_ranges["catch_velocity_scale"] = (0.0, 1.5)
        self.event_slider_ranges["throw_x"] = (scene_bounds["x"][0] - 0.15, scene_bounds["x"][1] + 0.15)
        self.event_slider_ranges["catch_x"] = self.event_slider_ranges["throw_x"]
        self.event_slider_ranges["throw_y"] = (scene_bounds["y"][0] - 0.15, scene_bounds["y"][1] + 0.15)
        self.event_slider_ranges["catch_y"] = self.event_slider_ranges["throw_y"]
        self.event_slider_ranges["throw_z"] = (min(0.0, scene_bounds["z"][0] - 0.1), scene_bounds["z"][1] + 0.2)
        self.event_slider_ranges["catch_z"] = self.event_slider_ranges["throw_z"]
        for key, scale in self.event_slider_scales.items():
            scale.configure(from_=self.event_slider_ranges[key][0], to=self.event_slider_ranges[key][1])

    def _selected_hand(self) -> str:
        hand = self.hand_editor_hand_var.get().strip() or "right"
        return hand if hand in HAND_NAMES else "right"

    def _populate_hand_point_list(self, select_point_id: str | None) -> None:
        hand = self._selected_hand()
        self.hand_point_listbox.delete(0, tk.END)
        self.hand_point_ids: list[str] = []
        for keyframe in self.project.sorted_hand_trajectory(hand):
            self.hand_point_ids.append(keyframe.id)
            metric_label = "|s|" if self._keyframe_uses_path_speed(keyframe) else "|v|"
            segment_label = keyframe.spline_to_next
            if keyframe.spline_to_next == "bspline":
                segment_label = f"bs{keyframe.bspline_degree}/{keyframe.bspline_control_points}"
            label = (
                f"{keyframe.id} | {segment_label} | "
                f"t={keyframe.time:.2f}s | "
                f"p=({keyframe.pos[0]:.2f}, {keyframe.pos[1]:.2f}, {keyframe.pos[2]:.2f}) | "
                f"{metric_label}={self._resolved_keyframe_speed(keyframe):.2f}"
            )
            self.hand_point_listbox.insert(tk.END, label)

        if not self.hand_point_ids:
            self.hand_point_source_id = None
            self._clear_hand_point_form()
            return

        target_id = select_point_id if select_point_id in self.hand_point_ids else self.hand_point_ids[0]
        index = self.hand_point_ids.index(target_id)
        self.hand_point_listbox.selection_clear(0, tk.END)
        self.hand_point_listbox.selection_set(index)
        self.hand_point_listbox.see(index)
        point = self.project.sorted_hand_trajectory(hand)[index]
        self._load_hand_point_into_form(point)

    def _load_hand_point_into_form(self, keyframe: HandKeyframe) -> None:
        self.hand_editor_hand_var.set(keyframe.hand)
        self.hand_point_source_id = keyframe.id
        self.hand_point_vars["id"].set(keyframe.id)
        self.hand_point_vars["time"].set(f"{keyframe.time:.3f}")
        self.hand_point_vars["spline_to_next"].set(keyframe.spline_to_next)
        self.hand_point_vars["bspline_degree"].set("" if keyframe.bspline_degree is None else str(keyframe.bspline_degree))
        self.hand_point_vars["bspline_control_points"].set(
            "" if keyframe.bspline_control_points is None else str(keyframe.bspline_control_points)
        )
        self.hand_point_vars["path_speed"].set(f"{self._resolved_keyframe_speed(keyframe):.3f}")
        for axis, value in zip(("x", "y", "z"), keyframe.pos):
            self.hand_point_vars[axis].set(f"{value:.3f}")
        for axis, value in zip(("x", "y", "z"), self._resolved_keyframe_velocity(keyframe)):
            self.hand_point_vars[f"v{axis}"].set(f"{value:.3f}")

    def _clear_hand_point_form(self) -> None:
        for variable in self.hand_point_vars.values():
            variable.set("")

    def _selected_hand_point_ids(self) -> list[str]:
        selection = [int(index) for index in self.hand_point_listbox.curselection()]
        return [self.hand_point_ids[index] for index in selection if 0 <= index < len(self.hand_point_ids)]

    def _prepare_new_event(self) -> None:
        event_id = self._next_event_id()
        ball_id = self._next_ball_id()
        start_time, end_time = self._default_new_times()
        self.form_source_event_id = None
        self.event_vars["id"].set(event_id)
        self.event_vars["ball"].set(ball_id)
        self.event_vars["throw_hand"].set("left")
        self.event_vars["catch_hand"].set("right")
        self.event_vars["throw_time"].set(f"{start_time:.3f}")
        self.event_vars["catch_time"].set(f"{end_time:.3f}")
        self.event_vars["catch_velocity_scale"].set("0.350")
        defaults = {
            "throw_x": "-0.320",
            "throw_y": "-0.080",
            "throw_z": "1.000",
            "catch_x": "0.220",
            "catch_y": "-0.060",
            "catch_z": "0.780",
        }
        for key, value in defaults.items():
            self.event_vars[key].set(value)
        try:
            self._sync_event_sliders_from_event(self._build_event_from_form())
        except Exception:
            pass
        self.event_listbox.selection_clear(0, tk.END)
        self._set_status("Prepared a new event. Apply it to add it to the project.")
        self._draw_scene()

    def _prepare_new_hand_point(self) -> None:
        hand = self._selected_hand()
        self.hand_point_source_id = None
        time_s = 0.5 * self.project.timeline_duration()
        self.hand_point_vars["id"].set(self._next_hand_point_id(hand))
        self.hand_point_vars["time"].set(f"{time_s:.3f}")
        defaults = (-0.35, 0.00, 0.95) if hand == "left" else (0.35, 0.00, 0.95)
        self.hand_point_vars["spline_to_next"].set("quintic")
        self.hand_point_vars["bspline_degree"].set("3")
        self.hand_point_vars["bspline_control_points"].set("6")
        for axis, value in zip(("x", "y", "z"), defaults):
            self.hand_point_vars[axis].set(f"{value:.3f}")
        default_velocity = self.project.hand_state(hand, time_s).velocity
        self.hand_point_vars["path_speed"].set(f"{float(np.linalg.norm(default_velocity)):.3f}")
        for axis, value in zip(("x", "y", "z"), default_velocity):
            self.hand_point_vars[f"v{axis}"].set(f"{value:.3f}")
        self.hand_point_listbox.selection_clear(0, tk.END)
        self._set_status(f"Prepared a new {hand} hand point. Apply it to add it.")
        self._draw_scene()

    def _default_new_times(self) -> tuple[float, float]:
        if self.project.is_loop:
            start = min(0.15 * self.project.loop_period, max(0.0, self.project.loop_period - 1.0))
            duration = min(1.0, max(0.4, 0.35 * self.project.loop_period))
            end = min(self.project.loop_period - 0.1, start + duration)
            if end <= start + 0.05:
                end = start + 0.5
            return start, end

        start = self.project.sequence_end_time() + 0.1
        end = start + 1.0
        return start, end

    def _next_event_id(self) -> str:
        existing = {event.id for event in self.project.events}
        index = 1
        while True:
            candidate = f"E{index}"
            if candidate not in existing:
                return candidate
            index += 1

    def _next_ball_id(self) -> str:
        existing = {event.ball for event in self.project.events}
        for letter in string.ascii_uppercase:
            if letter not in existing:
                return letter
        index = len(existing) + 1
        return f"Ball{index}"

    def _next_hand_point_id(self, hand: str) -> str:
        prefix = "L" if hand == "left" else "R"
        existing = set(self.project.hand_waypoint_ids(hand))
        index = 1
        while True:
            candidate = f"{prefix}{index}"
            if candidate not in existing:
                return candidate
            index += 1

    def _read_project_form(self) -> tuple[str, str, float, float]:
        return (
            self.name_var.get().strip() or "untitled_pattern",
            self.mode_var.get().strip() or "loop",
            float(self.loop_period_var.get()),
            float(self.gravity_var.get()),
        )

    def _build_event_from_form(self) -> ThrowEvent:
        return ThrowEvent(
            id=self.event_vars["id"].get().strip(),
            ball=self.event_vars["ball"].get().strip(),
            throw_hand=self.event_vars["throw_hand"].get().strip(),
            catch_hand=self.event_vars["catch_hand"].get().strip(),
            throw_time=float(self.event_vars["throw_time"].get()),
            catch_time=float(self.event_vars["catch_time"].get()),
            catch_velocity_scale=float(self.event_vars["catch_velocity_scale"].get()),
            throw_pos=(
                float(self.event_vars["throw_x"].get()),
                float(self.event_vars["throw_y"].get()),
                float(self.event_vars["throw_z"].get()),
            ),
            catch_pos=(
                float(self.event_vars["catch_x"].get()),
                float(self.event_vars["catch_y"].get()),
                float(self.event_vars["catch_z"].get()),
            ),
        )

    def _build_hand_point_from_form(self) -> HandKeyframe:
        hand = self._selected_hand()
        path_speed_text = self.hand_point_vars["path_speed"].get().strip()
        bspline_degree_text = self.hand_point_vars["bspline_degree"].get().strip()
        bspline_control_points_text = self.hand_point_vars["bspline_control_points"].get().strip()
        return HandKeyframe(
            id=self.hand_point_vars["id"].get().strip(),
            hand=hand,
            time=float(self.hand_point_vars["time"].get()),
            pos=(
                float(self.hand_point_vars["x"].get()),
                float(self.hand_point_vars["y"].get()),
                float(self.hand_point_vars["z"].get()),
            ),
            spline_to_next=self.hand_point_vars["spline_to_next"].get().strip() or "quintic",
            velocity=(
                float(self.hand_point_vars["vx"].get()),
                float(self.hand_point_vars["vy"].get()),
                float(self.hand_point_vars["vz"].get()),
            ),
            path_speed=(None if not path_speed_text else float(path_speed_text)),
            bspline_degree=(None if not bspline_degree_text else int(bspline_degree_text)),
            bspline_control_points=(None if not bspline_control_points_text else int(bspline_control_points_text)),
        )

    def _apply_project_settings(self) -> None:
        try:
            name, mode, loop_period, gravity = self._read_project_form()
            candidate = self.project.copy()
            candidate.name = name
            candidate.mode = mode  # type: ignore[assignment]
            candidate.loop_period = loop_period
            candidate.gravity = gravity
            candidate.validate()
        except (ValueError, ValidationError) as exc:
            self._set_status(f"Settings rejected: {exc}", error=True)
            return

        self.project = candidate
        self._load_project_into_controls(select_event_id=self.form_source_event_id)
        self._draw_scene()

    def _apply_event(self) -> None:
        if not self._commit_event_form(live=False):
            return
        self._load_project_into_controls(select_event_id=self.form_source_event_id)
        self._draw_scene()

    def _commit_event_form(self, *, live: bool) -> bool:
        try:
            event = self._build_event_from_form()
            name, mode, loop_period, gravity = self._read_project_form()
            candidate = self.project.copy()
            candidate.name = name
            candidate.mode = mode  # type: ignore[assignment]
            candidate.loop_period = loop_period
            candidate.gravity = gravity

            if self.form_source_event_id is None:
                candidate.events.append(event)
            else:
                replaced = False
                for idx, existing in enumerate(candidate.events):
                    if existing.id == self.form_source_event_id:
                        candidate.events[idx] = event
                        replaced = True
                        break
                if not replaced:
                    candidate.events.append(event)

            candidate.validate()
        except (ValueError, ValidationError) as exc:
            self._set_status(f"{'Live update rejected' if live else 'Event rejected'}: {exc}", error=True)
            return False

        self.project = candidate
        self.form_source_event_id = event.id
        if live:
            self._sync_event_sliders_from_event(event)
            self._refresh_event_list_labels(selected_ids={event.id})
            self._sync_time_controls(reset_time=False)
            self._set_status(f"Live update: {event.id}")
        else:
            self._set_status(f"Updated {event.id}.")
        return True

    def _apply_hand_point(self) -> None:
        if not self._commit_hand_point_form(live=False):
            return
        self._draw_scene()

    def _commit_hand_point_form(self, *, live: bool) -> bool:
        try:
            keyframe = self._build_hand_point_from_form()
            name, mode, loop_period, gravity = self._read_project_form()
            candidate = self.project.copy()
            candidate.name = name
            candidate.mode = mode  # type: ignore[assignment]
            candidate.loop_period = loop_period
            candidate.gravity = gravity

            for hand in HAND_NAMES:
                candidate.hand_trajectories[hand] = [
                    existing for existing in candidate.hand_trajectories.get(hand, [])
                    if existing.id != self.hand_point_source_id
                ]
            candidate.hand_trajectories[keyframe.hand] = list(candidate.hand_trajectories.get(keyframe.hand, [])) + [keyframe]
            candidate.validate()
        except (ValueError, ValidationError) as exc:
            if live:
                existing = self._find_hand_point(self._selected_hand(), self.hand_point_source_id)
                if existing is not None:
                    self._load_hand_point_into_form(existing)
            self._set_status(f"Hand point rejected: {exc}", error=True)
            return False

        self.project = candidate
        self.hand_editor_hand_var.set(keyframe.hand)
        self._populate_hand_point_list(select_point_id=keyframe.id)
        self._refresh_event_list_labels()
        self._sync_time_controls(reset_time=False)
        self._set_status(f"{'Live update' if live else 'Updated'} {keyframe.id}.")
        return True

    def _delete_selected_events(self) -> None:
        selected_ids = self._selected_event_ids()
        if not selected_ids:
            self._set_status("No events are selected to delete.", error=True)
            return

        candidate = self.project.copy()
        candidate.events = [event for event in candidate.events if event.id not in selected_ids]
        try:
            candidate.validate()
        except ValidationError as exc:
            self._set_status(f"Delete rejected: {exc}", error=True)
            return

        self.project = candidate
        next_id = candidate.sorted_events()[0].id if candidate.events else None
        self._load_project_into_controls(select_event_id=next_id)
        count = len(selected_ids)
        self._set_status(f"Deleted {count} event{'s' if count != 1 else ''}.")
        self._draw_scene()

    def _delete_selected_hand_points(self) -> None:
        selected_ids = self._selected_hand_point_ids()
        if not selected_ids:
            self._set_status("No hand points are selected to delete.", error=True)
            return

        hand = self._selected_hand()
        candidate = self.project.copy()
        candidate.hand_trajectories[hand] = [
            keyframe for keyframe in candidate.hand_trajectories.get(hand, [])
            if keyframe.id not in selected_ids
        ]
        try:
            candidate.validate()
        except ValidationError as exc:
            self._set_status(f"Delete rejected: {exc}", error=True)
            return

        self.project = candidate
        self._load_project_into_controls(select_event_id=self.form_source_event_id)
        self._populate_hand_point_list(select_point_id=None)
        count = len(selected_ids)
        self._set_status(f"Deleted {count} hand point{'s' if count != 1 else ''}.")
        self._draw_scene()

    def _reset_sample(self) -> None:
        self.project = build_three_ball_cascade_pattern()
        self.current_path = None
        self._load_project_into_controls(select_event_id=self.project.sorted_events()[0].id)
        self._draw_scene()

    def _load_from_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Open pattern YAML",
            filetypes=[("YAML Files", "*.yaml *.yml"), ("All Files", "*.*")],
        )
        if not path:
            return
        try:
            self.project = load_pattern_project(path)
        except Exception as exc:
            self._set_status(f"Load failed: {exc}", error=True)
            return

        self.current_path = Path(path)
        select_event_id = self.project.sorted_events()[0].id if self.project.events else None
        self._load_project_into_controls(select_event_id=select_event_id)
        self._draw_scene()

    def _save_to_file(self) -> None:
        try:
            name, mode, loop_period, gravity = self._read_project_form()
            candidate = self.project.copy()
            candidate.name = name
            candidate.mode = mode  # type: ignore[assignment]
            candidate.loop_period = loop_period
            candidate.gravity = gravity
            candidate.validate()
            self.project = candidate
        except (ValueError, ValidationError) as exc:
            self._set_status(f"Save failed: settings are invalid: {exc}", error=True)
            return

        if self.current_path is None:
            path = filedialog.asksaveasfilename(
                title="Save pattern YAML",
                defaultextension=".yaml",
                filetypes=[("YAML Files", "*.yaml *.yml"), ("All Files", "*.*")],
                initialfile=f"{self.project.name}.yaml",
            )
            if not path:
                return
            self.current_path = Path(path)

        try:
            save_pattern_project(self.project, self.current_path)
        except Exception as exc:
            self._set_status(f"Save failed: {exc}", error=True)
            return

        self._set_status(f"Saved {self.project.name} to {self.current_path}")

    def _on_event_selected(self, _event: object | None = None) -> None:
        selected_ids = self._selected_event_ids()
        if not selected_ids:
            return
        event_map = {event.id: event for event in self.project.sorted_events()}
        event = event_map[selected_ids[0]]
        self._load_event_into_form(event)
        if len(selected_ids) > 1:
            self._set_status(
                f"{len(selected_ids)} events selected. Delete removes all selected events; Apply edits the first selected event."
            )
        self._draw_scene()

    def _on_event_slider_changed(self, field: str) -> None:
        value = float(self.event_slider_vars[field].get())
        self.event_slider_value_vars[field].set(f"{value:.3f}")
        if self._suspend_event_slider_callbacks or self.form_source_event_id is None:
            return
        self.event_vars[field].set(f"{value:.3f}")
        if self._commit_event_form(live=True):
            self._draw_scene()

    def _on_hand_editor_changed(self, _event: object | None = None) -> None:
        self._populate_hand_point_list(select_point_id=None)
        self._draw_scene()

    def _on_hand_point_selected(self, _event: object | None = None) -> None:
        selected_ids = self._selected_hand_point_ids()
        if not selected_ids:
            return
        point_map = {point.id: point for point in self.project.sorted_hand_trajectory(self._selected_hand())}
        point = point_map[selected_ids[0]]
        self._load_hand_point_into_form(point)
        if len(selected_ids) > 1:
            self._set_status(
                f"{len(selected_ids)} hand points selected. Delete removes all selected points; Apply edits the first selected point."
            )
        self._draw_scene()

    def _on_canvas_press(self, event: tk.Event[tk.Canvas]) -> None:
        target = self._find_waypoint_hit_target(float(event.x), float(event.y))
        if target is None:
            self._drag_waypoint = None
            return

        hand = str(target["hand"])
        point_id = str(target["point_id"])
        projection = str(target["projection"])
        kind = str(target.get("kind", "position"))
        descriptor = "velocity handle" if kind == "velocity" else "waypoint"
        self._select_hand_point(hand, point_id, redraw=True, status=f"Selected {point_id} {descriptor} on the {projection} view.")
        self._drag_waypoint = {"hand": hand, "point_id": point_id, "projection": projection, "kind": kind}
        self.canvas.configure(cursor="fleur")

    def _on_canvas_drag(self, event: tk.Event[tk.Canvas]) -> None:
        if self._drag_waypoint is None:
            return
        projection = str(self._drag_waypoint["projection"])
        kind = str(self._drag_waypoint.get("kind", "position"))
        if (
            self._update_dragged_waypoint_from_canvas(float(event.x), float(event.y), projection)
            if kind == "position"
            else self._update_dragged_velocity_from_canvas(float(event.x), float(event.y), projection)
        ):
            self.canvas.configure(cursor="fleur")

    def _on_canvas_release(self, event: tk.Event[tk.Canvas]) -> None:
        if self._drag_waypoint is not None:
            self._on_canvas_drag(event)
        self._drag_waypoint = None
        self.canvas.configure(cursor="")

    def _toggle_playback(self) -> None:
        self.playing = not self.playing
        self.play_button.configure(text="Pause" if self.playing else "Play")
        self.last_tick = time.perf_counter()

    def _on_scrubbed(self, _value: str) -> None:
        self.current_time = float(self.time_var.get())
        self.playing = False
        self.play_button.configure(text="Play")
        self._update_time_label()
        self._draw_scene()

    def _sync_time_controls(self, reset_time: bool = False) -> None:
        duration = self.project.timeline_duration()
        self.time_scale.configure(to=max(0.001, duration))
        if reset_time:
            self.current_time = 0.0
            self.time_var.set(self.current_time)
        else:
            self.current_time = min(float(self.time_var.get()), duration)
            self.time_var.set(self.current_time)
        self._update_time_label()

    def _update_time_label(self) -> None:
        self.time_label_var.set(f"t = {self.current_time:.2f} s")

    def _tick(self) -> None:
        now = time.perf_counter()
        dt = now - self.last_tick
        self.last_tick = now

        if self.playing:
            duration = self.project.timeline_duration()
            self.current_time += dt * float(self.play_speed_var.get())
            if self.project.is_loop:
                self.current_time %= duration
            elif self.current_time >= duration:
                self.current_time = duration
                self.playing = False
                self.play_button.configure(text="Play")
            self.time_var.set(self.current_time)
            self._update_time_label()
            self._draw_scene()

        self.root.after(33, self._tick)

    def _set_status(self, message: str, error: bool = False) -> None:
        prefix = "Error: " if error else ""
        self.status_var.set(prefix + message)

    def _refresh_event_list_labels(self, selected_ids: set[str] | None = None) -> None:
        selected_ids = set() if selected_ids is None else set(selected_ids)
        current_ids = selected_ids or set(self._selected_event_ids())
        self.event_listbox.delete(0, tk.END)
        self.event_ids = []
        for event in self.project.sorted_events():
            self.event_ids.append(event.id)
            label = (
                f"{event.id} | {event.ball} "
                f"{event.throw_hand[0].upper()}->{event.catch_hand[0].upper()} "
                f"{event.throw_time:.2f}s -> {event.catch_time:.2f}s"
            )
            self.event_listbox.insert(tk.END, label)
        self.event_listbox.selection_clear(0, tk.END)
        for index, event_id in enumerate(self.event_ids):
            if event_id in current_ids:
                self.event_listbox.selection_set(index)
        if self.form_source_event_id in self.event_ids:
            self.event_listbox.see(self.event_ids.index(self.form_source_event_id))

    def _ball_color(self, ball: str) -> str:
        idx = sum(ord(ch) for ch in ball) % len(_BALL_PALETTE)
        return _BALL_PALETTE[idx]

    def _draw_scene(self) -> None:
        self.canvas.delete("all")
        self._waypoint_hit_targets = []
        self._projection_specs = {}
        width = max(200, self.canvas.winfo_width())
        height = max(200, self.canvas.winfo_height())
        pad = 18

        view_bottom = int(height * 0.56)
        mid_x = width // 2
        mid_y = pad + (view_bottom - pad) // 2
        xy_rect = (pad, pad, mid_x - pad // 2, mid_y - 6)
        xz_rect = (pad, mid_y + 6, mid_x - pad // 2, view_bottom)
        iso_rect = (mid_x + pad // 2, pad, width - pad, mid_y - 6)
        yz_rect = (mid_x + pad // 2, mid_y + 6, width - pad, view_bottom)
        plot_rect = (pad, view_bottom + 12, width - pad, int(height * 0.79))
        timeline_rect = (pad, plot_rect[3] + 12, width - pad, height - pad)

        for rect, title in (
            (xy_rect, "Top View (x / y)"),
            (xz_rect, "Front View (x / z)"),
            (iso_rect, "Isometric View"),
            (yz_rect, "Side View (y / z)"),
            (plot_rect, f"{self._selected_hand().title()} Hand Kinematics"),
            (timeline_rect, "Timing View"),
        ):
            self.canvas.create_rectangle(*rect, fill="#FFFCF5", outline="#D5CAB8", width=1)
            self.canvas.create_text(rect[0] + 12, rect[1] + 12, anchor="w", text=title, fill="#374151", font=("TkDefaultFont", 10, "bold"))

        try:
            self.project.validate()
            state = self.project.sample(self.current_time)
            bounds = self._scene_bounds()
        except ValidationError as exc:
            self.canvas.create_text(width / 2, height / 2, text=str(exc), fill="#991B1B", font=("TkDefaultFont", 12, "bold"))
            return

        self._projection_specs = {
            "xy": {"rect": xy_rect, "horizontal": "x", "vertical": "y", "h_bounds": bounds["x"], "v_bounds": bounds["y"]},
            "xz": {"rect": xz_rect, "horizontal": "x", "vertical": "z", "h_bounds": bounds["x"], "v_bounds": bounds["z"]},
            "yz": {"rect": yz_rect, "horizontal": "y", "vertical": "z", "h_bounds": bounds["y"], "v_bounds": bounds["z"]},
        }
        xy_map = self._make_mapper(xy_rect, bounds["x"], bounds["y"])
        xz_map = self._make_mapper(xz_rect, bounds["x"], bounds["z"])
        yz_map = self._make_mapper(yz_rect, bounds["y"], bounds["z"])
        iso_map = self._make_mapper(iso_rect, *self._iso_bounds())

        self._draw_axes(xy_rect, xy_map, x0=0.0, y0=0.0, x_label="x", y_label="y")
        self._draw_axes(xz_rect, xz_map, x0=0.0, y0=0.0, x_label="x", y_label="z")
        self._draw_axes(yz_rect, yz_map, x0=0.0, y0=0.0, x_label="y", y_label="z")
        self._draw_iso_axes(iso_map)

        selected_ids = set(self._selected_event_ids())
        for hand in HAND_NAMES:
            path = self.project.sample_hand_path(hand, samples=200)
            self._draw_polyline(xy_map, path[:, [1, 2]], fill=_HAND_COLORS[hand], width=3)
            self._draw_polyline(xz_map, path[:, [1, 3]], fill=_HAND_COLORS[hand], width=3)
            self._draw_polyline(yz_map, path[:, [2, 3]], fill=_HAND_COLORS[hand], width=3)
            self._draw_iso_polyline(iso_map, path[:, 1:4], fill=_HAND_COLORS[hand], width=3)
        self._draw_selected_hand_segments(xy_map, xz_map, yz_map, iso_map)
        self._draw_selected_bspline_controls(xy_map, xz_map, yz_map, iso_map)
        self._draw_hand_waypoints(xy_map, xz_map, yz_map, iso_map)

        for event in self.project.sorted_events():
            path = self.project.sample_event_flight(event, samples=72)
            color = self._ball_color(event.ball)
            width_px = 4 if event.id in selected_ids else 2
            dash = () if event.id in selected_ids else (5, 5)
            self._draw_polyline(xy_map, path[:, [1, 2]], fill=color, width=width_px, dash=dash)
            self._draw_polyline(xz_map, path[:, [1, 3]], fill=color, width=width_px, dash=dash)
            self._draw_polyline(yz_map, path[:, [2, 3]], fill=color, width=width_px, dash=dash)
            self._draw_iso_polyline(iso_map, path[:, 1:4], fill=color, width=width_px, dash=dash)
        self._draw_hand_velocity_arrows(xy_map, xz_map, yz_map, iso_map)

        for hand, position in state.hand_positions.items():
            self._draw_marker(xy_map, position[[0, 1]], _HAND_COLORS[hand], radius=7)
            self._draw_marker(xz_map, position[[0, 2]], _HAND_COLORS[hand], label=f"{hand[0].upper()}H", radius=7)
            self._draw_marker(yz_map, position[[1, 2]], _HAND_COLORS[hand], radius=7)
            self._draw_iso_marker(iso_map, position, _HAND_COLORS[hand], radius=7)

        for ball, position in state.ball_positions.items():
            color = self._ball_color(ball)
            self._draw_marker(xy_map, position[[0, 1]], color, radius=9)
            self._draw_marker(xz_map, position[[0, 2]], color, label=ball, radius=9)
            self._draw_marker(yz_map, position[[1, 2]], color, radius=9)
            self._draw_iso_marker(iso_map, position, color, radius=9)

        self._draw_hand_kinematics(plot_rect, self._selected_hand())
        self._draw_timeline(timeline_rect, selected_ids)

    def _scene_bounds(self) -> dict[str, tuple[float, float]]:
        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []
        for hand in HAND_NAMES:
            hand_path = self.project.sample_hand_path(hand, samples=160)
            xs.extend(hand_path[:, 1].tolist())
            ys.extend(hand_path[:, 2].tolist())
            zs.extend(hand_path[:, 3].tolist())
        for event in self.project.sorted_events():
            flight = self.project.sample_event_flight(event, samples=64)
            xs.extend(flight[:, 1].tolist())
            ys.extend(flight[:, 2].tolist())
            zs.extend(flight[:, 3].tolist())

        if not xs:
            xs = [-1.0, 1.0]
            ys = [-0.6, 0.6]
            zs = [0.0, 1.2]

        return {
            "x": self._pad_bounds(xs, min_span=1.0, pad_ratio=0.18),
            "y": self._pad_bounds(ys, min_span=0.6, pad_ratio=0.18),
            "z": self._pad_bounds(zs, min_span=1.0, pad_ratio=0.12),
        }

    def _pad_bounds(self, values: list[float], min_span: float, pad_ratio: float) -> tuple[float, float]:
        low = float(min(values))
        high = float(max(values))
        span = max(high - low, min_span)
        mid = 0.5 * (low + high)
        half = 0.5 * span * (1.0 + 2.0 * pad_ratio)
        return mid - half, mid + half

    def _make_mapper(
        self,
        rect: tuple[int, int, int, int],
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
    ) -> Callable[[float, float], tuple[float, float]]:
        x0, y0, x1, y1 = rect
        inset = 28
        draw_x0 = x0 + inset
        draw_x1 = x1 - inset
        draw_y0 = y0 + inset
        draw_y1 = y1 - inset
        x_span = max(x_bounds[1] - x_bounds[0], 1e-6)
        y_span = max(y_bounds[1] - y_bounds[0], 1e-6)

        def mapper(px: float, py: float) -> tuple[float, float]:
            sx = (px - x_bounds[0]) / x_span
            sy = (py - y_bounds[0]) / y_span
            cx = draw_x0 + sx * (draw_x1 - draw_x0)
            cy = draw_y1 - sy * (draw_y1 - draw_y0)
            return cx, cy

        return mapper

    def _project_iso_coords(self, x: float, y: float, z: float) -> tuple[float, float]:
        return (
            0.8660254 * (float(x) - float(y)),
            float(z) - 0.5 * (float(x) + float(y)),
        )

    def _iso_bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        u_values: list[float] = []
        v_values: list[float] = []

        for hand in HAND_NAMES:
            hand_path = self.project.sample_hand_path(hand, samples=160)
            for point in hand_path[:, 1:4]:
                u, v = self._project_iso_coords(point[0], point[1], point[2])
                u_values.append(u)
                v_values.append(v)
        for event in self.project.sorted_events():
            flight = self.project.sample_event_flight(event, samples=64)
            for point in flight[:, 1:4]:
                u, v = self._project_iso_coords(point[0], point[1], point[2])
                u_values.append(u)
                v_values.append(v)

        if not u_values:
            corners = [
                self._project_iso_coords(-1.0, -0.6, 0.0),
                self._project_iso_coords(1.0, 0.6, 1.2),
            ]
            u_values = [corner[0] for corner in corners]
            v_values = [corner[1] for corner in corners]

        return (
            self._pad_bounds(u_values, min_span=1.2, pad_ratio=0.18),
            self._pad_bounds(v_values, min_span=1.2, pad_ratio=0.18),
        )

    def _draw_iso_axes(self, mapper: Callable[[float, float], tuple[float, float]]) -> None:
        origin = self._project_iso_coords(0.0, 0.0, 0.0)
        x_end = self._project_iso_coords(0.35, 0.0, 0.0)
        y_end = self._project_iso_coords(0.0, 0.35, 0.0)
        z_end = self._project_iso_coords(0.0, 0.0, 0.45)
        ox, oy = mapper(*origin)
        xx, xy = mapper(*x_end)
        yx, yy = mapper(*y_end)
        zx, zy = mapper(*z_end)
        self.canvas.create_line(ox, oy, xx, xy, fill="#DC2626", width=2)
        self.canvas.create_line(ox, oy, yx, yy, fill="#059669", width=2)
        self.canvas.create_line(ox, oy, zx, zy, fill="#2563EB", width=2)
        self.canvas.create_text(xx + 6, xy, text="x", anchor="w", fill="#DC2626", font=("TkDefaultFont", 9, "bold"))
        self.canvas.create_text(yx - 6, yy, text="y", anchor="e", fill="#059669", font=("TkDefaultFont", 9, "bold"))
        self.canvas.create_text(zx, zy - 6, text="z", anchor="s", fill="#2563EB", font=("TkDefaultFont", 9, "bold"))

    def _draw_axes(
        self,
        rect: tuple[int, int, int, int],
        mapper: Callable[[float, float], tuple[float, float]],
        *,
        x0: float,
        y0: float,
        x_label: str,
        y_label: str,
    ) -> None:
        x_left, y_axis = mapper(x0 - 10.0, y0)
        x_right, _ = mapper(x0 + 10.0, y0)
        x_axis, y_bottom = mapper(x0, y0 - 10.0)
        _, y_top = mapper(x0, y0 + 10.0)
        if rect[1] < y_axis < rect[3]:
            self.canvas.create_line(rect[0] + 16, y_axis, rect[2] - 16, y_axis, fill="#D4D4D4")
            self.canvas.create_text(rect[2] - 18, y_axis - 10, text=x_label, fill="#6B7280", anchor="e")
        if rect[0] < x_axis < rect[2]:
            self.canvas.create_line(x_axis, rect[1] + 16, x_axis, rect[3] - 16, fill="#D4D4D4")
            self.canvas.create_text(x_axis + 10, rect[1] + 18, text=y_label, fill="#6B7280", anchor="w")

    def _draw_polyline(
        self,
        mapper: Callable[[float, float], tuple[float, float]],
        points: np.ndarray,
        *,
        fill: str,
        width: int,
        dash: tuple[int, int] | tuple[()] = (),
    ) -> None:
        coords: list[float] = []
        for point in points:
            x, y = mapper(float(point[0]), float(point[1]))
            coords.extend((x, y))
        if len(coords) >= 4:
            self.canvas.create_line(*coords, fill=fill, width=width, dash=dash, smooth=True)

    def _draw_iso_polyline(
        self,
        mapper: Callable[[float, float], tuple[float, float]],
        points: np.ndarray,
        *,
        fill: str,
        width: int,
        dash: tuple[int, int] | tuple[()] = (),
    ) -> None:
        projected = np.array([self._project_iso_coords(point[0], point[1], point[2]) for point in points], dtype=float)
        self._draw_polyline(mapper, projected, fill=fill, width=width, dash=dash)

    def _draw_arrow(
        self,
        mapper: Callable[[float, float], tuple[float, float]],
        start: np.ndarray,
        end: np.ndarray,
        *,
        fill: str,
        width: int,
        handle_fill: str,
        label: str | None = None,
    ) -> tuple[float, float]:
        sx, sy = mapper(float(start[0]), float(start[1]))
        ex, ey = mapper(float(end[0]), float(end[1]))
        self.canvas.create_line(sx, sy, ex, ey, fill=fill, width=width, arrow=tk.LAST, arrowshape=(10, 12, 4))
        self.canvas.create_oval(ex - 4, ey - 4, ex + 4, ey + 4, fill=handle_fill, outline=fill, width=2)
        if label:
            self.canvas.create_text(ex + 6, ey - 6, text=label, anchor="sw", fill=fill, font=("TkDefaultFont", 8, "bold"))
        return ex, ey

    def _draw_marker(
        self,
        mapper: Callable[[float, float], tuple[float, float]],
        point: np.ndarray,
        fill: str,
        *,
        radius: int,
        label: str | None = None,
    ) -> tuple[float, float]:
        cx, cy = mapper(float(point[0]), float(point[1]))
        self.canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, fill=fill, outline="#111827", width=1)
        if label:
            self.canvas.create_text(cx + radius + 6, cy - radius - 2, text=label, anchor="w", fill="#111827", font=("TkDefaultFont", 9, "bold"))
        return cx, cy

    def _draw_square_marker(
        self,
        mapper: Callable[[float, float], tuple[float, float]],
        point: np.ndarray,
        fill: str,
        *,
        size: int,
        label: str | None = None,
        outline: str = "#111827",
        width: int = 1,
    ) -> tuple[float, float]:
        cx, cy = mapper(float(point[0]), float(point[1]))
        self.canvas.create_rectangle(cx - size, cy - size, cx + size, cy + size, fill=fill, outline=outline, width=width)
        if label:
            self.canvas.create_text(cx + size + 6, cy - size - 2, text=label, anchor="w", fill="#111827", font=("TkDefaultFont", 8, "bold"))
        return cx, cy

    def _draw_iso_marker(
        self,
        mapper: Callable[[float, float], tuple[float, float]],
        point: np.ndarray,
        fill: str,
        *,
        radius: int,
        label: str | None = None,
    ) -> tuple[float, float]:
        return self._draw_marker(mapper, np.asarray(self._project_iso_coords(point[0], point[1], point[2]), dtype=float), fill, radius=radius, label=label)

    def _draw_iso_square_marker(
        self,
        mapper: Callable[[float, float], tuple[float, float]],
        point: np.ndarray,
        fill: str,
        *,
        size: int,
        label: str | None = None,
        outline: str = "#111827",
        width: int = 1,
    ) -> tuple[float, float]:
        return self._draw_square_marker(
            mapper,
            np.asarray(self._project_iso_coords(point[0], point[1], point[2]), dtype=float),
            fill,
            size=size,
            label=label,
            outline=outline,
            width=width,
        )

    def _selected_hand_point(self) -> HandKeyframe | None:
        return self._find_hand_point(self._selected_hand(), self.hand_point_source_id)

    def _keyframe_uses_path_speed(self, keyframe: HandKeyframe) -> bool:
        keyframes = self.project.sorted_hand_trajectory(keyframe.hand)
        for index, current in enumerate(keyframes):
            if current.id != keyframe.id:
                continue
            return current.spline_to_next == "bspline" or (
                index > 0 and keyframes[index - 1].spline_to_next == "bspline"
            )
        return keyframe.spline_to_next == "bspline"

    def _resolved_keyframe_vector_velocity(self, keyframe: HandKeyframe) -> np.ndarray:
        if keyframe.velocity is not None:
            return np.asarray(keyframe.velocity, dtype=float)
        return self.project.hand_state(keyframe.hand, keyframe.time).velocity

    def _resolved_keyframe_tangent(self, keyframe: HandKeyframe) -> np.ndarray:
        tangent = self.project.hand_keyframe_tangent(keyframe)
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm > 1e-9:
            return tangent
        velocity = self._resolved_keyframe_vector_velocity(keyframe)
        velocity_norm = float(np.linalg.norm(velocity))
        if velocity_norm > 1e-9:
            return velocity / velocity_norm
        return np.zeros(3, dtype=float)

    def _resolved_keyframe_speed(self, keyframe: HandKeyframe) -> float:
        if self._keyframe_uses_path_speed(keyframe):
            return (
                float(keyframe.path_speed)
                if keyframe.path_speed is not None
                else self.project.hand_keyframe_path_speed(keyframe)
            )
        return float(np.linalg.norm(self._resolved_keyframe_vector_velocity(keyframe)))

    def _resolved_keyframe_velocity(self, keyframe: HandKeyframe) -> np.ndarray:
        if self._keyframe_uses_path_speed(keyframe):
            return self._resolved_keyframe_tangent(keyframe) * self._resolved_keyframe_speed(keyframe)
        return self._resolved_keyframe_vector_velocity(keyframe)

    def _selected_bspline_segment(self) -> dict[str, object] | None:
        point = self._selected_hand_point()
        if point is None or point.spline_to_next != "bspline":
            return None
        return self.project.hand_bspline_segment(point.hand, point.time, point.pos, samples=96)

    def _find_hand_point(self, hand: str, point_id: str | None) -> HandKeyframe | None:
        if not point_id:
            return None
        hand = hand.strip() if hand in HAND_NAMES else self._selected_hand()
        for point in self.project.sorted_hand_trajectory(hand):
            if point.id == point_id:
                return point
        return None

    def _select_hand_point(self, hand: str, point_id: str, *, redraw: bool, status: str | None = None) -> None:
        self.hand_editor_hand_var.set(hand)
        self._populate_hand_point_list(select_point_id=point_id)
        if status:
            self._set_status(status)
        if redraw:
            self._draw_scene()

    def _find_waypoint_hit_target(self, canvas_x: float, canvas_y: float) -> dict[str, object] | None:
        best_target: dict[str, object] | None = None
        best_distance_sq = _WAYPOINT_HIT_RADIUS_PX * _WAYPOINT_HIT_RADIUS_PX
        for target in self._waypoint_hit_targets:
            dx = float(target["cx"]) - canvas_x
            dy = float(target["cy"]) - canvas_y
            distance_sq = dx * dx + dy * dy
            if distance_sq <= best_distance_sq:
                best_distance_sq = distance_sq
                best_target = target
        return best_target

    def _projection_value_at_canvas(self, projection: str, canvas_x: float, canvas_y: float) -> dict[str, float] | None:
        spec = self._projection_specs.get(projection)
        if spec is None:
            return None

        rect = spec["rect"]
        if not isinstance(rect, tuple) or len(rect) != 4:
            return None
        h_bounds = spec["h_bounds"]
        v_bounds = spec["v_bounds"]
        if not isinstance(h_bounds, tuple) or not isinstance(v_bounds, tuple):
            return None

        inset = 28.0
        draw_x0 = float(rect[0]) + inset
        draw_x1 = float(rect[2]) - inset
        draw_y0 = float(rect[1]) + inset
        draw_y1 = float(rect[3]) - inset
        if draw_x1 <= draw_x0 or draw_y1 <= draw_y0:
            return None

        clamped_x = min(max(canvas_x, draw_x0), draw_x1)
        clamped_y = min(max(canvas_y, draw_y0), draw_y1)
        sx = (clamped_x - draw_x0) / max(draw_x1 - draw_x0, 1e-9)
        sy = (draw_y1 - clamped_y) / max(draw_y1 - draw_y0, 1e-9)

        horizontal = str(spec["horizontal"])
        vertical = str(spec["vertical"])
        return {
            horizontal: float(h_bounds[0]) + sx * float(h_bounds[1] - h_bounds[0]),
            vertical: float(v_bounds[0]) + sy * float(v_bounds[1] - v_bounds[0]),
        }

    def _update_dragged_waypoint_from_canvas(self, canvas_x: float, canvas_y: float, projection: str) -> bool:
        if self._drag_waypoint is None:
            return False
        hand = str(self._drag_waypoint["hand"])
        point_id = str(self._drag_waypoint["point_id"])
        point = self._find_hand_point(hand, point_id)
        axes = self._projection_value_at_canvas(projection, canvas_x, canvas_y)
        if point is None or axes is None:
            return False

        values = {"x": float(point.pos[0]), "y": float(point.pos[1]), "z": float(point.pos[2])}
        values.update(axes)
        self.hand_point_vars["x"].set(f"{values['x']:.3f}")
        self.hand_point_vars["y"].set(f"{values['y']:.3f}")
        self.hand_point_vars["z"].set(f"{values['z']:.3f}")
        if self._commit_hand_point_form(live=True):
            self._draw_scene()
            return True
        return False

    def _update_dragged_velocity_from_canvas(self, canvas_x: float, canvas_y: float, projection: str) -> bool:
        if self._drag_waypoint is None:
            return False
        hand = str(self._drag_waypoint["hand"])
        point_id = str(self._drag_waypoint["point_id"])
        point = self._find_hand_point(hand, point_id)
        axes = self._projection_value_at_canvas(projection, canvas_x, canvas_y)
        if point is None or axes is None:
            return False

        pos_values = {"x": float(point.pos[0]), "y": float(point.pos[1]), "z": float(point.pos[2])}
        if self._keyframe_uses_path_speed(point):
            dragged = self._resolved_keyframe_velocity(point).copy()
            for axis, endpoint_value in axes.items():
                dragged["xyz".index(axis)] = (float(endpoint_value) - pos_values[axis]) / _VELOCITY_HANDLE_TIME_SCALE_S
            dragged_norm = float(np.linalg.norm(dragged))
            if dragged_norm <= 1e-9:
                return False
            speed = max(self._resolved_keyframe_speed(point), 1e-6)
            velocity_values = (dragged / dragged_norm) * speed
            self.hand_point_vars["path_speed"].set(f"{speed:.3f}")
            self.hand_point_vars["vx"].set(f"{velocity_values[0]:.3f}")
            self.hand_point_vars["vy"].set(f"{velocity_values[1]:.3f}")
            self.hand_point_vars["vz"].set(f"{velocity_values[2]:.3f}")
        else:
            velocity_values = {
                "x": float(self._resolved_keyframe_velocity(point)[0]),
                "y": float(self._resolved_keyframe_velocity(point)[1]),
                "z": float(self._resolved_keyframe_velocity(point)[2]),
            }
            for axis, endpoint_value in axes.items():
                velocity_values[axis] = (float(endpoint_value) - pos_values[axis]) / _VELOCITY_HANDLE_TIME_SCALE_S

            self.hand_point_vars["vx"].set(f"{velocity_values['x']:.3f}")
            self.hand_point_vars["vy"].set(f"{velocity_values['y']:.3f}")
            self.hand_point_vars["vz"].set(f"{velocity_values['z']:.3f}")
            self.hand_point_vars["path_speed"].set(
                f"{float(np.linalg.norm([velocity_values['x'], velocity_values['y'], velocity_values['z']])):.3f}"
            )
        if self._commit_hand_point_form(live=True):
            self._draw_scene()
            return True
        return False

    def _draw_selected_hand_segments(
        self,
        xy_map: Callable[[float, float], tuple[float, float]],
        xz_map: Callable[[float, float], tuple[float, float]],
        yz_map: Callable[[float, float], tuple[float, float]],
        iso_map: Callable[[float, float], tuple[float, float]],
    ) -> None:
        point = self._selected_hand_point()
        if point is None:
            return

        nodes = self.project.hand_keyframes(point.hand)
        target_index = None
        target_pos = np.asarray(point.pos, dtype=float)
        for idx, (timestamp, position) in enumerate(nodes):
            if abs(timestamp - point.time) <= 1e-6 and np.allclose(position, target_pos, atol=1e-6):
                target_index = idx
                break
        if target_index is None:
            return

        color = _HAND_COLORS[point.hand]
        segments: list[tuple[float, float]] = []
        if target_index > 0:
            segments.append((nodes[target_index - 1][0], nodes[target_index][0]))
        if target_index + 1 < len(nodes):
            segments.append((nodes[target_index][0], nodes[target_index + 1][0]))

        for start_time, end_time in segments:
            if end_time <= start_time:
                continue
            times = np.linspace(start_time, end_time, 48, dtype=float)
            positions = np.vstack([self.project.hand_state(point.hand, time_s).position for time_s in times])
            self._draw_polyline(xy_map, positions[:, [0, 1]], fill=color, width=6)
            self._draw_polyline(xz_map, positions[:, [0, 2]], fill=color, width=6)
            self._draw_polyline(yz_map, positions[:, [1, 2]], fill=color, width=6)
            self._draw_iso_polyline(iso_map, positions, fill=color, width=6)

    def _draw_selected_bspline_controls(
        self,
        xy_map: Callable[[float, float], tuple[float, float]],
        xz_map: Callable[[float, float], tuple[float, float]],
        yz_map: Callable[[float, float], tuple[float, float]],
        iso_map: Callable[[float, float], tuple[float, float]],
    ) -> None:
        segment = self._selected_bspline_segment()
        point = self._selected_hand_point()
        if segment is None or point is None:
            return

        control_points = np.asarray(segment["control_points"], dtype=float)
        label = f"B{segment['degree']}/{len(control_points)}"
        color = _HAND_COLORS[point.hand]
        self._draw_polyline(xy_map, control_points[:, [0, 1]], fill=color, width=1, dash=(4, 4))
        self._draw_polyline(xz_map, control_points[:, [0, 2]], fill=color, width=1, dash=(4, 4))
        self._draw_polyline(yz_map, control_points[:, [1, 2]], fill=color, width=1, dash=(4, 4))
        self._draw_iso_polyline(iso_map, control_points, fill=color, width=1, dash=(4, 4))

        for index, control in enumerate(control_points):
            control_label = label if index == 0 else None
            self._draw_marker(xy_map, control[[0, 1]], "#FEF3C7", radius=3)
            self._draw_marker(xz_map, control[[0, 2]], "#FEF3C7", radius=3, label=control_label)
            self._draw_marker(yz_map, control[[1, 2]], "#FEF3C7", radius=3)
            self._draw_iso_marker(iso_map, control, "#FEF3C7", radius=3)

    def _draw_hand_waypoints(
        self,
        xy_map: Callable[[float, float], tuple[float, float]],
        xz_map: Callable[[float, float], tuple[float, float]],
        yz_map: Callable[[float, float], tuple[float, float]],
        iso_map: Callable[[float, float], tuple[float, float]],
    ) -> None:
        selected_hand = self._selected_hand()
        selected_point_id = self.hand_point_source_id
        for hand in HAND_NAMES:
            color = _HAND_COLORS[hand]
            for point in self.project.sorted_hand_trajectory(hand):
                pos = np.asarray(point.pos, dtype=float)
                label = point.id if hand == selected_hand else None
                is_selected = hand == selected_hand and point.id == selected_point_id
                size = 7 if is_selected else 5
                width = 2 if is_selected else 1
                outline = "#111827" if is_selected else "#374151"
                self._draw_square_marker(xy_map, pos[[0, 1]], color, size=size, outline=outline, width=width)
                xz_center = self._draw_square_marker(xz_map, pos[[0, 2]], color, size=size, label=label, outline=outline, width=width)
                yz_center = self._draw_square_marker(yz_map, pos[[1, 2]], color, size=size, outline=outline, width=width)
                self._draw_iso_square_marker(iso_map, pos, color, size=size, outline=outline, width=width)
                self._waypoint_hit_targets.append(
                    {"hand": hand, "point_id": point.id, "projection": "xz", "kind": "position", "cx": xz_center[0], "cy": xz_center[1]}
                )
                self._waypoint_hit_targets.append(
                    {"hand": hand, "point_id": point.id, "projection": "yz", "kind": "position", "cx": yz_center[0], "cy": yz_center[1]}
                )

            for event in self.project.sorted_events():
                if event.throw_hand == hand:
                    pos = np.asarray(event.throw_pos, dtype=float)
                    self._draw_marker(xy_map, pos[[0, 1]], "#FFFFFF", radius=4)
                    self._draw_marker(xz_map, pos[[0, 2]], "#FFFFFF", radius=4, label=(f"{event.id}T" if hand == selected_hand else None))
                    self._draw_marker(yz_map, pos[[1, 2]], "#FFFFFF", radius=4)
                    self._draw_iso_marker(iso_map, pos, "#FFFFFF", radius=4)
                if event.catch_hand == hand:
                    pos = np.asarray(event.catch_pos, dtype=float)
                    self._draw_marker(xy_map, pos[[0, 1]], "#F8FAFC", radius=4)
                    self._draw_marker(xz_map, pos[[0, 2]], "#F8FAFC", radius=4, label=(f"{event.id}C" if hand == selected_hand else None))
                    self._draw_marker(yz_map, pos[[1, 2]], "#F8FAFC", radius=4)
                    self._draw_iso_marker(iso_map, pos, "#F8FAFC", radius=4)

    def _draw_hand_velocity_arrows(
        self,
        xy_map: Callable[[float, float], tuple[float, float]],
        xz_map: Callable[[float, float], tuple[float, float]],
        yz_map: Callable[[float, float], tuple[float, float]],
        iso_map: Callable[[float, float], tuple[float, float]],
    ) -> None:
        hand = self._selected_hand()
        selected_point_id = self.hand_point_source_id
        for point in self.project.sorted_hand_trajectory(hand):
            pos = np.asarray(point.pos, dtype=float)
            velocity = self._resolved_keyframe_velocity(point)
            tip = pos + (_VELOCITY_HANDLE_TIME_SCALE_S * velocity)
            is_selected = point.id == selected_point_id
            width = 3 if is_selected else 2
            handle_fill = "#FEF3C7" if is_selected else "#FFFFFF"
            label_suffix = "t" if self._keyframe_uses_path_speed(point) else "v"
            label = f"{point.id}{label_suffix}" if is_selected else None
            color = _HAND_COLORS[hand]

            self._draw_arrow(xy_map, pos[[0, 1]], tip[[0, 1]], fill=color, width=width, handle_fill=handle_fill)
            xz_tip = self._draw_arrow(xz_map, pos[[0, 2]], tip[[0, 2]], fill=color, width=width, handle_fill=handle_fill, label=label)
            yz_tip = self._draw_arrow(yz_map, pos[[1, 2]], tip[[1, 2]], fill=color, width=width, handle_fill=handle_fill)
            iso_start = np.asarray(self._project_iso_coords(pos[0], pos[1], pos[2]), dtype=float)
            iso_tip = np.asarray(self._project_iso_coords(tip[0], tip[1], tip[2]), dtype=float)
            self._draw_arrow(iso_map, iso_start, iso_tip, fill=color, width=width, handle_fill=handle_fill)

            self._waypoint_hit_targets.append(
                {"hand": hand, "point_id": point.id, "projection": "xz", "kind": "velocity", "cx": xz_tip[0], "cy": xz_tip[1]}
            )
            self._waypoint_hit_targets.append(
                {"hand": hand, "point_id": point.id, "projection": "yz", "kind": "velocity", "cx": yz_tip[0], "cy": yz_tip[1]}
            )

    def _draw_hand_kinematics(self, rect: tuple[int, int, int, int], hand: str) -> None:
        x0, y0, x1, y1 = rect
        left = x0 + 50
        right = x1 - 14
        top = y0 + 30
        bottom = y1 - 12
        gap = 10
        plot_height = max(28.0, (bottom - top - 2 * gap) / 3.0)
        duration = max(self.project.timeline_duration(), 1e-6)
        times = np.linspace(0.0, duration, 240, dtype=float)
        states = [self.project.hand_state(hand, ti) for ti in times]
        groups = [
            ("Position [m]", np.vstack([state.position for state in states])),
            ("Velocity [m/s]", np.vstack([state.velocity for state in states])),
            ("Acceleration [m/s^2]", np.vstack([state.acceleration for state in states])),
        ]
        axis_colors = ["#DC2626", "#059669", "#2563EB"]
        authored_times = [point.time for point in self.project.sorted_hand_trajectory(hand)]
        anchor_times = []
        for event in self.project.sorted_events():
            if event.throw_hand == hand:
                anchor_times.append(event.throw_time)
            if event.catch_hand == hand:
                anchor_times.append(event.catch_time)

        for idx, (label, data) in enumerate(groups):
            y_top = top + idx * (plot_height + gap)
            y_bottom = y_top + plot_height
            self.canvas.create_rectangle(left, y_top, right, y_bottom, outline="#E8DDCF", width=1)
            self.canvas.create_text(x0 + 12, y_top + 10, text=label, anchor="w", fill="#374151", font=("TkDefaultFont", 9, "bold"))

            ymin = float(np.min(data))
            ymax = float(np.max(data))
            if abs(ymax - ymin) < 1e-9:
                ymin -= 1.0
                ymax += 1.0
            pad = 0.12 * (ymax - ymin)
            ymin -= pad
            ymax += pad
            if ymin <= 0.0 <= ymax:
                zero_y = y_bottom - ((0.0 - ymin) / (ymax - ymin)) * (y_bottom - y_top)
                self.canvas.create_line(left, zero_y, right, zero_y, fill="#E5E7EB")

            def map_xy(time_s: float, value: float) -> tuple[float, float]:
                px = left + (time_s / duration) * (right - left)
                py = y_bottom - ((value - ymin) / (ymax - ymin)) * (y_bottom - y_top)
                return px, py

            for marker_time in authored_times:
                if 0.0 <= marker_time <= duration:
                    px = left + (marker_time / duration) * (right - left)
                    self.canvas.create_line(px, y_top, px, y_bottom, fill="#C084FC", dash=(2, 3))
            for marker_time in anchor_times:
                if 0.0 <= marker_time <= duration:
                    px = left + (marker_time / duration) * (right - left)
                    self.canvas.create_line(px, y_top, px, y_bottom, fill="#CBD5E1", dash=(3, 4))

            for axis_index, color in enumerate(axis_colors):
                coords: list[float] = []
                for time_s, row in zip(times, data):
                    px, py = map_xy(float(time_s), float(row[axis_index]))
                    coords.extend((px, py))
                self.canvas.create_line(*coords, fill=color, width=2, smooth=True)

            current_x = left + (min(max(self.current_time, 0.0), duration) / duration) * (right - left)
            self.canvas.create_line(current_x, y_top, current_x, y_bottom, fill="#111827", width=2)
            self.canvas.create_text(right - 8, y_top + 10, text="x  y  z", anchor="e", fill="#6B7280", font=("TkDefaultFont", 8))

    def _draw_timeline(self, rect: tuple[int, int, int, int], selected_ids: set[str]) -> None:
        duration = self.project.timeline_duration()
        x0, y0, x1, y1 = rect
        left_pad = 68
        right_pad = 20
        top_pad = 30
        bottom_pad = 18
        draw_x0 = x0 + left_pad
        draw_x1 = x1 - right_pad
        draw_y0 = y0 + top_pad
        draw_y1 = y1 - bottom_pad

        ball_rows = self.project.ball_ids()
        row_labels = ball_rows + ["LH", "RH"]
        row_count = max(1, len(row_labels))
        row_height = max(26.0, (draw_y1 - draw_y0) / row_count)

        def map_t(t: float) -> float:
            return draw_x0 + (t / max(duration, 1e-6)) * (draw_x1 - draw_x0)

        for row_index, label in enumerate(row_labels):
            cy = draw_y0 + row_height * (row_index + 0.5)
            self.canvas.create_line(draw_x0, cy, draw_x1, cy, fill="#EEE7DA")
            self.canvas.create_text(x0 + 18, cy, text=label, anchor="w", fill="#374151", font=("TkDefaultFont", 10, "bold"))

        for tick in np.linspace(0.0, duration, 6):
            tx = map_t(float(tick))
            self.canvas.create_line(tx, draw_y0 - 12, tx, draw_y1, fill="#E5DFD3")
            self.canvas.create_text(tx, y0 + 12, text=f"{tick:.2f}", fill="#6B7280", font=("TkDefaultFont", 9))

        ball_row_index = {ball: idx for idx, ball in enumerate(ball_rows)}
        for event in self.project.sorted_events():
            row = ball_row_index[event.ball]
            cy = draw_y0 + row_height * (row + 0.5)
            color = self._ball_color(event.ball)
            for seg_start, seg_end in self._display_segments(event.throw_time, event.catch_time, duration):
                sx = map_t(seg_start)
                ex = map_t(seg_end)
                self.canvas.create_rectangle(
                    sx,
                    cy - row_height * 0.26,
                    ex,
                    cy + row_height * 0.26,
                    fill=color,
                    outline="#111827" if event.id in selected_ids else color,
                    width=2 if event.id in selected_ids else 1,
                )
            self.canvas.create_text(
                map_t(event.throw_time % duration if self.project.is_loop else event.throw_time) + 4,
                cy - row_height * 0.34,
                text=event.id,
                anchor="sw",
                fill="#111827",
                font=("TkDefaultFont", 8, "bold"),
            )

        for row_offset, hand in enumerate(HAND_NAMES, start=len(ball_rows)):
            cy = draw_y0 + row_height * (row_offset + 0.5)
            color = _HAND_COLORS[hand]
            for timestamp, position in self.project.hand_keyframes(hand):
                for seg_start, _seg_end in self._display_segments(timestamp, min(timestamp + 0.01, duration), duration):
                    x = map_t(seg_start)
                    self.canvas.create_line(x, cy - row_height * 0.30, x, cy + row_height * 0.30, fill=color, width=2)
                    self.canvas.create_oval(x - 4, cy - 4, x + 4, cy + 4, fill=color, outline="#111827")

        current_x = map_t(self.current_time)
        self.canvas.create_line(current_x, draw_y0 - 10, current_x, draw_y1, fill="#111827", width=2)

    def _display_segments(self, start: float, end: float, duration: float) -> list[tuple[float, float]]:
        if duration <= 0.0:
            return []
        if not self.project.is_loop:
            clipped_start = max(0.0, start)
            clipped_end = min(duration, end)
            if clipped_end <= clipped_start:
                return []
            return [(clipped_start, clipped_end)]

        span = end - start
        if span <= 0.0:
            return []
        if span >= duration - 1e-9:
            return [(0.0, duration)]

        start_mod = start % duration
        if start_mod + span <= duration + 1e-9:
            return [(start_mod, min(duration, start_mod + span))]
        return [
            (start_mod, duration),
            (0.0, (start_mod + span) - duration),
        ]


def launch_pattern_studio(project: PatternProject | None = None, initial_path: str | Path | None = None) -> None:
    root = tk.Tk()
    path = None if initial_path is None else Path(initial_path)
    app_project = project.copy() if project is not None else build_three_ball_cascade_pattern()
    PatternStudioApp(root, app_project, initial_path=path)
    root.mainloop()
