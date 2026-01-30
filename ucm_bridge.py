"""
UCM Event Loop Orchestrator bridging cognition (SpaceField) and DALS workers.
Keeps UI thread free by dispatching worker tasks to QThreadPool.
"""

import asyncio
import hashlib
import logging
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QObject, QRunnable, QThread, QThreadPool, Signal, Slot

from space_field import ConvergenceState, SpaceFieldVisualizer, VizTraversalVerdict
class SemanticDimension(Enum):
    PATTERN = "x"
    CONTINUITY = "y"
    SYNTHESIS = "xyz"
    GOVERNANCE = "z"


@dataclass
class CorePersonality:
    core_id: str
    primary_axis: SemanticDimension
    anchor_vector: Tuple[float, float, float]
    volatility_coefficient: float
    consensus_radius: float = 0.25

    def generate_initial_position(self, query_seed: str) -> Tuple[float, float, float]:
        seeded_random = random.Random(f"{query_seed}-{self.core_id}")
        base = np.array(self.anchor_vector)

        if self.primary_axis == SemanticDimension.SYNTHESIS:
            perturbation = np.array(
                [
                    seeded_random.uniform(-0.3, 0.3),
                    seeded_random.uniform(-0.3, 0.3),
                    seeded_random.uniform(-0.3, 0.3),
                ]
            )
            position = base + perturbation
        else:
            jitter = seeded_random.uniform(-0.1, 0.1)
            position = base + np.array([jitter, jitter, jitter])

        return tuple(np.clip(position, -1.0, 1.0))


class Core4Initializer:
    PROFILES: Dict[str, CorePersonality] = {
        "caleon": CorePersonality(
            core_id="caleon",
            primary_axis=SemanticDimension.PATTERN,
            anchor_vector=(0.8, 0.0, 0.0),
            volatility_coefficient=0.1,
            consensus_radius=0.2,
        ),
        "cali_x": CorePersonality(
            core_id="cali_x",
            primary_axis=SemanticDimension.CONTINUITY,
            anchor_vector=(0.0, 0.8, 0.0),
            volatility_coefficient=0.15,
            consensus_radius=0.25,
        ),
        "kaygee": CorePersonality(
            core_id="kaygee",
            primary_axis=SemanticDimension.SYNTHESIS,
            anchor_vector=(0.33, 0.33, 0.33),
            volatility_coefficient=0.8,
            consensus_radius=0.35,
        ),
        "ecm": CorePersonality(
            core_id="ecm",
            primary_axis=SemanticDimension.GOVERNANCE,
            anchor_vector=(0.0, 0.0, 0.8),
            volatility_coefficient=0.05,
            consensus_radius=0.15,
        ),
    }

    @classmethod
    def initialize_manifold_state(cls, query: str) -> Dict[str, Tuple[float, float, float]]:
        query_seed = hashlib.sha256(query.encode()).hexdigest()[:12]
        return {cid: profile.generate_initial_position(query_seed) for cid, profile in cls.PROFILES.items()}

    @classmethod
    def check_convergence_validity(cls, positions: Dict[str, Tuple[float, float, float]]) -> bool:
        pos_array = np.array(list(positions.values()))
        if np.any(np.abs(pos_array) > 1.0):
            return False

        centroid = np.mean(pos_array, axis=0)
        distances = np.linalg.norm(pos_array - centroid, axis=1)
        if np.all(distances < 0.01):
            logging.warning("False consensus detected: collapse to single point")
            return False
        return True

logger = logging.getLogger("UCM.Orchestrator")


class ConvergenceEvent(Enum):
    CONVERGED = "converged"
    DIVERGENT = "divergent"
    TENSION = "tension"
    TIMEOUT = "timeout"


@dataclass
class CoreVerdict:
    core_id: str
    position: Tuple[float, float, float]
    confidence: float
    consensus_vector: Tuple[float, float, float]
    is_outlier: bool = False


class WorkerDispatchTask(QRunnable):
    """I/O-only worker executed in QThreadPool."""

    def __init__(self, query: str, verdicts: List[CoreVerdict], callback: Callable, vault_path: str):
        super().__init__()
        self.query = query
        self.verdicts = verdicts
        self.callback = callback
        self.vault_path = vault_path
        self.setAutoDelete(True)

    def run(self):
        try:
            logger.info(f"Worker thread {QThread.currentThreadId()} writing to {self.vault_path}")
            time.sleep(0.5)
            result = {
                "query": self.query,
                "status": "committed",
                "vault": self.vault_path,
                "consensus_confidence": sum(v.confidence for v in self.verdicts) / max(1, len(self.verdicts)),
                "timestamp": time.time(),
            }
            self.callback(result)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Worker failed: {exc}")
            self.callback({"status": "failed", "error": str(exc)})


class UCMCognitiveBridge(QObject):
    cognition_phase_complete = Signal(list)  # verdicts
    convergence_detected = Signal(str, float)
    worker_phase_complete = Signal(dict)
    escalation_required = Signal(str, list)
    manifold_updated = Signal()

    def __init__(self, visualizer: SpaceFieldVisualizer, convergence_timeout: float = 5.0, parent=None):
        super().__init__(parent)
        self.visualizer = visualizer
        self.convergence_timeout = convergence_timeout
        self.thread_pool = QThreadPool.globalInstance()
        self.current_query: Optional[str] = None
        self.is_active = False

        self.initializer = Core4Initializer()

        self.visualizer.convergence_updated.connect(self._on_convergence_update)

    async def process_query(self, query: str, max_retries: int = 2):
        self.current_query = query
        self.is_active = True
        logger.info(f"Ingress: {query}")

        for attempt in range(max_retries):
            verdicts = await self._run_core_traversal(query)
            self.cognition_phase_complete.emit(verdicts)
            self.visualizer.update_manifold(verdicts)
            self.manifold_updated.emit()

            state = await self._wait_for_convergence_state(verdicts, timeout_sec=self.convergence_timeout)

            if state == ConvergenceEvent.CONVERGED:
                avg_conf = sum(v.confidence for v in verdicts) / max(1, len(verdicts))
                self.convergence_detected.emit("CONVERGED", avg_conf)
                result = await self._dispatch_to_workers(query, verdicts)
                self.is_active = False
                return result

            if state == ConvergenceEvent.TENSION:
                dissenters = self._identify_dissenters(verdicts)
                coords = [v.position for v in dissenters]
                self.escalation_required.emit(query, coords)
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5)
                    continue
                self.is_active = False
                return {"status": "escalated", "reason": "tension", "coords": coords}

            if state == ConvergenceEvent.TIMEOUT:
                logger.error("Convergence timeout")
                self.is_active = False
                return {"status": "timeout", "query": query}

        self.is_active = False
        return {"status": "aborted", "reason": "no_convergence"}

    async def _run_core_traversal(self, query: str) -> List[VizTraversalVerdict]:
        initial_positions = self.initializer.initialize_manifold_state(query)

        if not self.initializer.check_convergence_validity(initial_positions):
            logger.error("Manifold initialization failed; using safety anchors")
            initial_positions = {
                "caleon": (0.5, 0.0, 0.0),
                "cali_x": (-0.5, 0.0, 0.0),
                "kaygee": (0.0, 0.5, 0.0),
                "ecm": (0.0, -0.5, 0.0),
            }

        verdicts: List[VizTraversalVerdict] = []
        for core_id, start_pos in initial_positions.items():
            trajectory = self._simulate_cognitive_drift(core_id, start_pos, query)
            confidence = self._calculate_confidence(core_id, trajectory)
            verdicts.append(
                VizTraversalVerdict(
                    core_id=core_id,
                    position=trajectory,
                    confidence=confidence,
                )
            )

        await asyncio.sleep(0.05)
        return verdicts

    async def _wait_for_convergence_state(self, verdicts: List[VizTraversalVerdict], timeout_sec: float) -> ConvergenceEvent:
        start = time.time()
        while True:
            state = self.visualizer.geometry.detect(verdicts)
            if state == ConvergenceState.CONVERGED:
                return ConvergenceEvent.CONVERGED
            if state == ConvergenceState.TENSION:
                return ConvergenceEvent.TENSION
            if time.time() - start > timeout_sec:
                return ConvergenceEvent.TIMEOUT
            await asyncio.sleep(0.05)

    def _identify_dissenters(self, verdicts: List[VizTraversalVerdict]) -> List[VizTraversalVerdict]:
        positions = [np.array(v.position) for v in verdicts]
        centroid = np.mean(positions, axis=0)
        dissenters = []
        for v in verdicts:
            dist = float(np.linalg.norm(np.array(v.position) - centroid))
            if dist > 0.8:
                dissenters.append(v)
        return dissenters

    async def _dispatch_to_workers(self, query: str, verdicts: List[VizTraversalVerdict]):
        worker_verdicts = []
        centroid = np.mean([np.array(v.position) for v in verdicts], axis=0)
        for v in verdicts:
            worker_verdicts.append(
                CoreVerdict(
                    core_id=v.core_id,
                    position=v.position,
                    confidence=v.confidence,
                    consensus_vector=tuple(centroid - np.array(v.position)),
                    is_outlier=v in self._identify_dissenters(verdicts),
                )
            )

        worker_future: asyncio.Future = asyncio.Future()

        def on_done(result: Dict[str, any]):
            if not worker_future.done():
                worker_future.set_result(result)
            self.worker_phase_complete.emit(result)

        task = WorkerDispatchTask(query=query, verdicts=worker_verdicts, callback=on_done, vault_path="posteriori_vault/converged/")
        self.thread_pool.start(task)
        logger.info("Worker dispatched to QThreadPool")
        await worker_future
        return worker_future.result()

    @Slot(str, float)
    def _on_convergence_update(self, state: str, confidence: float):
        logger.debug(f"Manifold state: {state} @ {confidence:.2f}")

    def emergency_stop(self):
        self.is_active = False
        self.thread_pool.clear()
        logger.warning("Emergency stop triggered")

    def _simulate_cognitive_drift(self, core_id: str, start: Tuple[float, float, float], query: str) -> Tuple[float, float, float]:
        profile = Core4Initializer.PROFILES[core_id]
        q_hash = hashlib.sha256(f"{query}-{core_id}".encode()).hexdigest()
        rng = random.Random(q_hash)

        query_vector = np.array(
            [
                rng.uniform(-1.0, 1.0),
                rng.uniform(-1.0, 1.0),
                rng.uniform(-1.0, 1.0),
            ]
        )

        bias_weight = max(0.0, 1.0 - profile.volatility_coefficient)
        drift = np.array(start) * bias_weight + query_vector * (1 - bias_weight)

        return tuple(np.clip(drift, -1.0, 1.0))

    def _calculate_confidence(self, core_id: str, position: Tuple[float, float, float]) -> float:
        profile = Core4Initializer.PROFILES[core_id]
        distance_from_anchor = float(np.linalg.norm(np.array(position) - np.array(profile.anchor_vector)))
        base = 0.65 + (1.0 - profile.volatility_coefficient) * 0.25
        penalty = min(distance_from_anchor, 1.0) * 0.2
        return max(0.1, min(1.0, base - penalty))


class ORBXController:
    def __init__(self, visualizer: SpaceFieldVisualizer):
        self.bridge = UCMCognitiveBridge(visualizer)
        self.bridge.escalation_required.connect(self._on_escalation)
        self.bridge.worker_phase_complete.connect(self._on_worker_done)

    async def execute_query(self, query: str):
        return await self.bridge.process_query(query)

    def _on_escalation(self, query: str, coords: List):
        logger.warning(f"Escalation for query: {query} | coords: {coords}")

    def _on_worker_done(self, result: Dict[str, any]):
        logger.info(f"Worker phase complete: {result}")
