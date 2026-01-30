"""
PySide6 desktop shell for CALI ORB.
Provides dashboard, query panel, and log console.
"""

import sys
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

from space_field import ConvergenceState, SpaceFieldVisualizer, VizTraversalVerdict
from ucm_bridge import ORBXController

from orb_main import CALI


class ORBWindow(QtWidgets.QMainWindow):
    def __init__(self, cali: CALI):
        super().__init__()
        self.cali = cali
        self.setWindowTitle("CALI ORB - PySide6")
        self.resize(960, 640)

        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        self.dashboard_tab = self._build_dashboard_tab()
        self.cali_tab = self._build_cali_tab()
        self.worker_tab = self._build_worker_tab()
        self.command_tab = self._build_command_tab()
        self.log_tab = self._build_log_tab()

        self.tabs.addTab(self.dashboard_tab, "Dashboard")
        self.tabs.addTab(self.cali_tab, "CALI")
        self.tabs.addTab(self.worker_tab, "Workers")
        self.tabs.addTab(self.command_tab, "Commands")
        self.tabs.addTab(self.log_tab, "Logs")

        self.escalations: List[Dict[str, Any]] = []
        self.last_verdict_points: List[Tuple[str, Tuple[float, float, float], float]] = []

        self.controller = ORBXController(self.sf_visualizer)
        self.controller.bridge.convergence_detected.connect(self._on_convergence_change)  # type: ignore[arg-type]
        self.controller.bridge.escalation_required.connect(self._on_bridge_escalation)  # type: ignore[arg-type]
        self.controller.bridge.worker_phase_complete.connect(self._on_worker_phase_complete)  # type: ignore[arg-type]

        self._init_health_timer()
        self._log("CALI ORB UI ready")

    # ----------------------- UI builders -----------------------
    def _build_dashboard_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(widget)

        self.label_cali_status = QtWidgets.QLabel("—")
        self.label_attractors = QtWidgets.QLabel("—")
        self.label_traces = QtWidgets.QLabel("—")
        self.label_convergence = QtWidgets.QLabel("—")
        self.label_escalations = QtWidgets.QLabel("—")
        self.label_forge = QtWidgets.QLabel("—")
        self.label_last_position = QtWidgets.QLabel("—")

        layout.addRow("CALI Status", self.label_cali_status)
        layout.addRow("Space Field Attractors", self.label_attractors)
        layout.addRow("Ghost Traces", self.label_traces)
        layout.addRow("Convergence Events", self.label_convergence)
        layout.addRow("Worker Escalations", self.label_escalations)
        layout.addRow("Forge Pending", self.label_forge)
        layout.addRow("Last Convergence", self.label_last_position)

        refresh_btn = QtWidgets.QPushButton("Refresh Now")
        refresh_btn.clicked.connect(self.update_health)  # type: ignore[arg-type]
        layout.addRow(refresh_btn)
        return widget

    def _build_cali_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(widget)

        self.sf_visualizer = SpaceFieldVisualizer()
        self.sf_visualizer.convergence_updated.connect(self._on_convergence_change)  # type: ignore[arg-type]
        vbox.addWidget(self.sf_visualizer)

        self.escalation_list = QtWidgets.QListWidget()
        self.escalation_list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        vbox.addWidget(QtWidgets.QLabel("Escalations (Rule B/D)"))
        vbox.addWidget(self.escalation_list)

        promote_btn = QtWidgets.QPushButton("Promote to Apriori (manual gate)")
        promote_btn.clicked.connect(self._on_manual_promote)  # type: ignore[arg-type]
        vbox.addWidget(promote_btn)
        return widget

    def _build_worker_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(widget)

        self.worker_table = QtWidgets.QTableWidget(0, 4)
        self.worker_table.setHorizontalHeaderLabels(["Worker", "Domain", "Calls", "Success Rate"])
        self.worker_table.horizontalHeader().setStretchLastSection(True)
        vbox.addWidget(self.worker_table)

        refresh_btn = QtWidgets.QPushButton("Refresh Workers")
        refresh_btn.clicked.connect(self.refresh_workers)  # type: ignore[arg-type]
        vbox.addWidget(refresh_btn)
        return widget

    def _build_command_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(widget)

        self.query_edit = QtWidgets.QPlainTextEdit()
        self.query_edit.setPlaceholderText("Ask CALI or request a procedure...")
        vbox.addWidget(self.query_edit)

        send_btn = QtWidgets.QPushButton("Send to CALI")
        send_btn.clicked.connect(self._on_send_clicked)  # type: ignore[arg-type]
        vbox.addWidget(send_btn)

        self.result_view = QtWidgets.QPlainTextEdit()
        self.result_view.setReadOnly(True)
        vbox.addWidget(self.result_view)

        self.proposal_box = QtWidgets.QGroupBox("Forge Proposals (Rule E)")
        p_layout = QtWidgets.QHBoxLayout(self.proposal_box)
        self.proposal_id_input = QtWidgets.QLineEdit()
        self.proposal_id_input.setPlaceholderText("proposal id")
        approve_btn = QtWidgets.QPushButton("Approve")
        reject_btn = QtWidgets.QPushButton("Reject")
        approve_btn.clicked.connect(lambda: self._handle_proposal(True))  # type: ignore[arg-type]
        reject_btn.clicked.connect(lambda: self._handle_proposal(False))  # type: ignore[arg-type]
        p_layout.addWidget(self.proposal_id_input)
        p_layout.addWidget(approve_btn)
        p_layout.addWidget(reject_btn)
        vbox.addWidget(self.proposal_box)

        return widget

    def _build_log_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(widget)
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        vbox.addWidget(self.log_view)
        return widget

    # ----------------------- Actions -----------------------
    def _on_send_clicked(self):
        query = self.query_edit.toPlainText().strip()
        if not query:
            return
        self._log(f"→ {query}")
        self.query_edit.clear()
        asyncio.create_task(self._run_query(query))

    async def _run_query(self, query: str):
        try:
            result = await self.controller.execute_query(query)
            if result is not None:
                self._on_result(result)
        except Exception as exc:  # noqa: BLE001
            self._on_error(str(exc))

    def _on_result(self, result: Dict[str, Any]):
        self._log(f"✓ result: {result.get('status')}")
        pretty = json_dumps(result)
        self.result_view.setPlainText(pretty)

        if result.get("status") == "pending":
            self._register_escalation({"reason": "non-convergence", "detail": result})

        if "detailed_verdicts" in result:
            self._update_visualizer(result["detailed_verdicts"], result.get("position"))

        # Capture escalations returned from worker path
        if result.get("status") == "escalated" or result.get("reason") == "novelty_high_value":
            self._register_escalation(result)

    def _on_error(self, message: str):
        self._log(f"⚠ error: {message}")

    def update_health(self):
        try:
            health = self.cali.get_health_report()
        except Exception as exc:  # noqa: BLE001
            self._log(f"Health check failed: {exc}")
            return

        self.label_cali_status.setText(str(health.get("cali_status")))
        self.label_attractors.setText(str(health.get("space_field_attractors")))
        self.label_traces.setText(str(health.get("ghost_trace_count")))
        self.label_convergence.setText(str(health.get("convergence_events")))
        self.label_escalations.setText(str(health.get("worker_escalations")))
        self.label_forge.setText(str(health.get("forge_pending")))

    def refresh_workers(self):
        registry = self.cali.workers.worker_registry
        self.worker_table.setRowCount(len(registry))
        for row, (wid, meta) in enumerate(registry.items()):
            self.worker_table.setItem(row, 0, QtWidgets.QTableWidgetItem(wid))
            self.worker_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(meta.get("domain", ""))))
            self.worker_table.setItem(row, 2, QtWidgets.QTableWidgetItem(str(meta.get("call_count", 0))))
            self.worker_table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{meta.get('success_rate', 0):.2f}"))

    def _handle_proposal(self, approved: bool):
        proposal_id = self.proposal_id_input.text().strip()
        if not proposal_id:
            return
        self.cali.forges.approve_proposal(proposal_id, approved, cali_notes="UI decision")
        self._log(f"Forge proposal {proposal_id} {'approved' if approved else 'rejected'}")

    def _on_manual_promote(self):
        self._log("Manual Apriori promotion requested (hook VaultSystem here).")

    def _register_escalation(self, payload: Dict[str, Any]):
        self.escalations.append(payload)
        display = json_dumps(payload)
        self.escalation_list.addItem(display)

    def _init_health_timer(self):
        self.health_timer = QtCore.QTimer(self)
        self.health_timer.setInterval(5000)
        self.health_timer.timeout.connect(self.update_health)  # type: ignore[arg-type]
        self.health_timer.start()
        self.update_health()
        self.refresh_workers()

    def _log(self, message: str):
        timestamp = QtCore.QDateTime.currentDateTime().toString("hh:mm:ss")
        self.log_view.appendPlainText(f"[{timestamp}] {message}")

    def _update_visualizer(self, verdicts_payload: List[Dict[str, Any]], position_hint: Any):
        verdicts: List[VizTraversalVerdict] = []
        for item in verdicts_payload:
            try:
                verdicts.append(
                    VizTraversalVerdict(
                        core_id=item.get("core_id", "?"),
                        position=tuple(item.get("position", (0.0, 0.0, 0.0))),
                        confidence=float(item.get("confidence", 0.0)),
                        timestamp=float(item.get("timestamp", time.time())),
                        metadata=item,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                self._log(f"Visualizer verdict skipped: {exc}")
        if verdicts:
            self.sf_visualizer.update_manifold(verdicts)
            self.label_last_position.setText(str(position_hint))

    def _on_convergence_change(self, state: str, confidence: float):
        self._log(f"Convergence: {state} @ {confidence:.2f}")

    def _on_bridge_escalation(self, query: str, coords: List):
        payload = {"status": "escalated", "query": query, "coords": coords}
        self._register_escalation(payload)
        self._log(f"Escalation signaled for query '{query}'")

    def _on_worker_phase_complete(self, result: Dict[str, Any]):
        self._log(f"Worker phase complete: {result.get('status')}")
        self.result_view.setPlainText(json_dumps(result))


def json_dumps(data: Dict[str, Any]) -> str:
    try:
        import json

        return json.dumps(data, indent=2, default=str)
    except Exception:
        return str(data)


def build_cali() -> CALI:
    orb_dir = Path("cali_orb")
    orb_dir.mkdir(parents=True, exist_ok=True)
    return CALI(orb_dir)


def main():
    app = QtWidgets.QApplication(sys.argv)
    cali = build_cali()
    window = ORBWindow(cali)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
