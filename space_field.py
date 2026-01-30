"""
Space Field visualizer and geometry utilities.
Extracted from orb_gui for reuse by orchestrators/bridges.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Tuple

import numpy as np
from PySide6 import QtCore, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SpaceField")


class ConvergenceState(Enum):
    DIVERGENT = "divergent"
    CONVERGING = "converging"
    CONVERGED = "converged"
    TENSION = "tension"


@dataclass
class VizTraversalVerdict:
    core_id: str
    position: Tuple[float, float, float]
    confidence: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be within [0,1]")
        if len(self.position) != 3:
            raise ValueError("Position must be a 3D tuple")


@dataclass
class GhostTrace:
    position: np.ndarray
    opacity: float
    timestamp: float

    def decay(self, rate: float = 0.15) -> bool:
        self.opacity -= rate
        return self.opacity <= 0


class ConvergenceGeometry:
    def __init__(self, consensus_radius: float = 0.25, tension_threshold: float = 0.8):
        self.consensus_radius = consensus_radius
        self.tension_threshold = tension_threshold

    def centroid(self, positions: List[np.ndarray]) -> np.ndarray:
        if not positions:
            return np.array([0.0, 0.0, 0.0])
        return np.mean(positions, axis=0)

    def detect(self, verdicts: List[VizTraversalVerdict]) -> ConvergenceState:
        if len(verdicts) < 2:
            return ConvergenceState.DIVERGENT

        positions = np.array([v.position for v in verdicts])
        center = self.centroid(list(positions))
        distances = np.linalg.norm(positions - center, axis=1)
        max_disp = float(np.max(distances))

        if np.any(distances > self.tension_threshold):
            return ConvergenceState.TENSION
        if max_disp <= self.consensus_radius:
            return ConvergenceState.CONVERGED
        if max_disp <= self.consensus_radius * 2:
            return ConvergenceState.CONVERGING
        return ConvergenceState.DIVERGENT


class SpaceFieldVisualizer(QtWidgets.QWidget):
    verdicts_received = QtCore.Signal(list)
    convergence_updated = QtCore.Signal(str, float)

    CORE_PALETTE = {
        "caleon": {"main": "#00ffcc", "ghost": "#00ffcc", "glow": "#004d40"},
        "cali_x": {"main": "#ff00ff", "ghost": "#ff00ff", "glow": "#4a004a"},
        "kaygee": {"main": "#ffff00", "ghost": "#ffff00", "glow": "#4a4a00"},
        "ecm": {"main": "#ff4444", "ghost": "#ff4444", "glow": "#4a0000"},
        "centroid": {"main": "#ffffff", "ghost": "#888888", "glow": "#333333"},
    }

    def __init__(self, parent=None, max_history: int = 10, animation_steps: int = 15):
        super().__init__(parent)
        self.max_history = max_history
        self.animation_steps = animation_steps
        self.current_step = 0

        self.ghost_trails: Dict[str, deque] = {core: deque(maxlen=max_history) for core in self.CORE_PALETTE}
        self.current_verdicts: Dict[str, VizTraversalVerdict] = {}
        self.target_verdicts: Dict[str, VizTraversalVerdict] = {}
        self.interpolation_frames: List[Dict] = []

        self.geometry = ConvergenceGeometry()
        self.convergence_state = ConvergenceState.DIVERGENT
        self.integrity_map: Dict[str, Dict[str, Any]] = {}

        self.verdicts_received.connect(self._process_verdicts_slot)

        self._init_ui()
        self._init_animation_timer()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.status_label = QtWidgets.QLabel("STATE: INITIALIZING")
        self.status_label.setStyleSheet("color: #888888; padding: 5px; font-weight: bold;")
        layout.addWidget(self.status_label)

        self.fig = Figure(figsize=(6, 6), dpi=100, facecolor="#0d1117")
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet("background-color: #0d1117;")
        layout.addWidget(self.canvas)

        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_facecolor("#0d1117")
        self._configure_axes()

        self._stored_view = (30, -60)

    def _configure_axes(self):
        self.ax.grid(True, color="#21262d", linestyle="--", alpha=0.5)
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.xaxis.pane.set_edgecolor("#30363d")
        self.ax.yaxis.pane.set_edgecolor("#30363d")
        self.ax.zaxis.pane.set_edgecolor("#30363d")

        self.ax.set_xlabel("PATTERN [X]", color="#c9d1d9", fontsize=9, labelpad=10)
        self.ax.set_ylabel("CONTINUITY [Y]", color="#c9d1d9", fontsize=9, labelpad=10)
        self.ax.set_zlabel("GOVERNANCE [Z]", color="#c9d1d9", fontsize=9, labelpad=10)

        self.ax.tick_params(axis="x", colors="#8b949e", labelsize=8)
        self.ax.tick_params(axis="y", colors="#8b949e", labelsize=8)
        self.ax.tick_params(axis="z", colors="#8b949e", labelsize=8)

    def _init_animation_timer(self):
        self.anim_timer = QtCore.QTimer(self)
        self.anim_timer.timeout.connect(self._animation_step)
        self.anim_timer.setInterval(33)

    def update_manifold(self, verdicts: List[VizTraversalVerdict]):
        valid = []
        for v in verdicts:
            try:
                valid.append(
                    VizTraversalVerdict(
                        v.core_id,
                        tuple(v.position),
                        float(v.confidence),
                        v.timestamp,
                        v.metadata,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Invalid verdict: {exc}")
        if valid:
            self.verdicts_received.emit(valid)

    def update_integrity_overlay(self, health_map: Dict[str, Dict[str, Any]]):
        """Update per-core structural integrity map (used during rendering)."""
        self.integrity_map = health_map or {}

    @QtCore.Slot(list)
    def _process_verdicts_slot(self, verdicts: List[VizTraversalVerdict]):
        self.current_verdicts = {
            core_id: self.current_verdicts.get(core_id, VizTraversalVerdict(core_id, (0, 0, 0), 0.0))
            for core_id in [v.core_id for v in verdicts]
        }
        self.target_verdicts = {v.core_id: v for v in verdicts}

        self._build_interpolation()
        self._update_ghost_trails()

        self.convergence_state = self.geometry.detect(verdicts)
        self._update_status()

        self.current_step = 0
        self.anim_timer.start()

    def _build_interpolation(self):
        self.interpolation_frames = []
        for step in range(1, self.animation_steps + 1):
            alpha = step / self.animation_steps
            frame: Dict[str, Tuple[Tuple[float, float, float], float]] = {}
            for core_id, target in self.target_verdicts.items():
                current = self.current_verdicts.get(core_id)
                if not current:
                    continue
                new_pos = tuple(current.position[i] + (target.position[i] - current.position[i]) * alpha for i in range(3))
                new_conf = current.confidence + (target.confidence - current.confidence) * alpha
                frame[core_id] = (new_pos, new_conf)
            self.interpolation_frames.append(frame)

    def _update_ghost_trails(self):
        for core_id, verdict in self.current_verdicts.items():
            if core_id in self.target_verdicts:
                trail = self.ghost_trails.get(core_id, deque(maxlen=self.max_history))
                trail.append(GhostTrace(position=np.array(verdict.position), opacity=1.0, timestamp=time.time()))
                self.ghost_trails[core_id] = trail

        for core_id, trail in self.ghost_trails.items():
            surviving = deque(maxlen=self.max_history)
            for ghost in trail:
                if not ghost.decay(rate=0.1):
                    surviving.append(ghost)
            self.ghost_trails[core_id] = surviving

    def _animation_step(self):
        if self.current_step >= len(self.interpolation_frames):
            self.anim_timer.stop()
            self.current_verdicts = self.target_verdicts.copy()
            return

        frame = self.interpolation_frames[self.current_step]
        self._render_frame(frame)
        self.current_step += 1

    def _render_frame(self, frame_data: Dict[str, Tuple[Tuple[float, float, float], float]]):
        self.ax.clear()
        self._configure_axes()

        positions: List[Tuple[float, float, float]] = []

        for core_id, trail in self.ghost_trails.items():
            if not trail:
                continue
            color_base = self.CORE_PALETTE.get(core_id, self.CORE_PALETTE["centroid"])
            for ghost in trail:
                if ghost.opacity <= 0.1:
                    continue
                rgba = to_rgba(color_base["ghost"], alpha=ghost.opacity * 0.3)
                self.ax.scatter(*ghost.position, color=rgba, s=50, marker="o")
                self.ax.plot(
                    [0, ghost.position[0]],
                    [0, ghost.position[1]],
                    [0, ghost.position[2]],
                    color=rgba,
                    alpha=ghost.opacity * 0.2,
                    linewidth=1,
                )

        for core_id, (pos, conf) in frame_data.items():
            positions.append(pos)
            palette = self.CORE_PALETTE.get(core_id, self.CORE_PALETTE["centroid"])
            size = 100 + (conf * 200)

            integrity = float(self.integrity_map.get(core_id, {}).get("structural_integrity", 1.0))
            alpha = max(0.1, min(1.0, 0.3 + integrity * 0.7))
            color_main = palette["main"] if integrity >= 0.7 else "#ff0000"

            self.ax.scatter(*pos, color=color_main, s=size, edgecolors=palette["glow"], linewidths=2, alpha=alpha)
            if conf > 0.75:
                self.ax.scatter(*pos, color="none", s=size * 1.3, edgecolors="white", linewidths=1, alpha=0.5)
            if integrity < 1.0:
                self.ax.scatter(*pos, color="none", s=size * 1.5, edgecolors="#ff4444", linewidths=2, alpha=0.5)

            self.ax.text(pos[0], pos[1], pos[2], f" {core_id.upper()}\n{conf:.2f}", color="white", fontsize=8, weight="bold")

        if len(positions) >= 2:
            self._render_convergence_geometry(positions)

        self.ax.view_init(elev=self._stored_view[0], azim=self._stored_view[1])
        self._stored_view = (self.ax.elev, self.ax.azim)
        self.canvas.draw()

    def _render_convergence_geometry(self, positions: List[Tuple[float, float, float]]):
        pos_array = np.array(positions)
        centroid = self.geometry.centroid(list(pos_array))

        if self.convergence_state == ConvergenceState.CONVERGED:
            self.ax.scatter(*centroid, color="#00ff00", s=200, marker="*", alpha=0.8)
        elif self.convergence_state == ConvergenceState.TENSION:
            self.ax.scatter(*centroid, color="#ff0000", s=150, marker="x", alpha=0.8, linewidths=3)

        for i, pos1 in enumerate(positions):
            for pos2 in positions[i + 1 :]:
                self.ax.plot(
                    [pos1[0], pos2[0]],
                    [pos1[1], pos2[1]],
                    [pos1[2], pos2[2]],
                    color="#30363d",
                    alpha=0.3,
                    linewidth=1,
                )

    def _update_status(self):
        color_map = {
            ConvergenceState.CONVERGED: "#00ff00",
            ConvergenceState.CONVERGING: "#ffaa00",
            ConvergenceState.DIVERGENT: "#ff4444",
            ConvergenceState.TENSION: "#ff00ff",
        }
        color = color_map.get(self.convergence_state, "#888888")
        self.status_label.setStyleSheet(f"color: {color}; padding: 5px; font-weight: bold;")
        self.status_label.setText(f"STATE: {self.convergence_state.value.upper()} | CORES: {len(self.current_verdicts)}")
        avg_conf = float(np.mean([v.confidence for v in self.current_verdicts.values()])) if self.current_verdicts else 0.0
        self.convergence_updated.emit(self.convergence_state.value, avg_conf)

    def clear_history(self):
        for core_id in self.ghost_trails:
            self.ghost_trails[core_id].clear()
        self.update_manifold([])
