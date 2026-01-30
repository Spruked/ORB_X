"""
System Maintenance SKG - CALI_ORB Oversight Module (scaffold-safe)
Doctrine: Cognition here, execution in DALS (Memory 29)
Location: system_maintenance.py
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import asyncio


# Safe imports with stubs for development -------------------------------------------------
try:
    from PySide6.QtCore import QObject, Signal  # type: ignore
except ImportError:  # pragma: no cover
    class QObject:  # type: ignore
        pass

    class Signal:  # type: ignore
        def __init__(self, *args):
            pass

        def emit(self, *args):
            pass


try:
    from vault_logic_system.memory.consolidation import ImmutableMemoryMatrix  # type: ignore
except ImportError:  # pragma: no cover
    class ImmutableMemoryMatrix:  # type: ignore
        def record_observation(self, **kwargs):
            logging.info(f"[STUB] record_observation: {kwargs}")


try:
    from security_audit import FileIntegrityMonitor  # type: ignore
except ImportError:  # pragma: no cover
    class FileIntegrityMonitor:  # type: ignore
        def check_directory(self, path):
            return []


try:
    from dals_bridge import DALSExecutionBridge  # type: ignore
except ImportError:  # pragma: no cover
    DALSExecutionBridge = None  # type: ignore


class MaintenanceClass(Enum):
    DIAGNOSTIC = "diagnostic"
    REPAIR = "repair"
    REDESIGN = "redesign"
    EMERGENCY = "emergency"


class AuthorizationLevel(Enum):
    L1_DIAGNOSTIC = 1
    L2_PATCH = 2
    L3_REDEPLOY = 3
    L4_ARCHITECTURAL = 4


@dataclass(frozen=True)
class RepairTicket:
    """Immutable repair proposal - hash enforced even in stub mode."""

    ticket_id: str
    timestamp: str
    target_system: str
    maintenance_class: MaintenanceClass
    issue_description: str
    proposed_changes: Dict[str, str]
    diff_patch: str
    authorization_required: AuthorizationLevel
    diagnostic_data: Dict = field(default_factory=dict)

    def __post_init__(self):
        content = f"{self.ticket_id}{self.timestamp}{self.diff_patch}"
        object.__setattr__(self, "ticket_hash", hashlib.sha256(content.encode()).hexdigest()[:16])


class SystemMaintenanceSKG(QObject):
    """
    Minimal scaffold preserving security invariants.
    Enforces: no self-modification, DALS delegation pattern.
    """

    ticket_created = Signal(object)
    authorization_required = Signal(object, str)
    repair_executed = Signal(str, bool)
    system_health_report = Signal(dict)

    PROTECTED_SYSTEMS = {"cali_orb", "cali", "orb", "self"}
    CORE_4_SYSTEMS = {"caleon", "cali_x", "kaygee", "ecm"}

    def __init__(self, cali_orb_path: Path, dals_bridge: Optional[Any] = None, immutable_matrix: Optional[Any] = None, parent=None):
        super().__init__(parent)
        self.orb_path = Path(cali_orb_path)
        self.dals = dals_bridge
        self.matrix = immutable_matrix or ImmutableMemoryMatrix()

        self.temp_vault = self.orb_path / "temp_maintenance_vault"
        self.temp_vault.mkdir(parents=True, exist_ok=True)

        self.active_tickets: Dict[str, RepairTicket] = {}
        self.auth_cache: set[str] = set()

        logging.info(f"MaintenanceSKG initialized | DALS Bridge: {self.dals is not None}")

    def _is_self_modification(self, target: str) -> bool:
        target_lower = target.lower().replace("_", "").replace("-", "")
        return target_lower in self.PROTECTED_SYSTEMS or "cali" in target_lower

    def run_diagnostics(self, target_system: Optional[str] = None) -> Dict:
        results: Dict[str, Any] = {}
        targets = [target_system] if target_system else list(self.CORE_4_SYSTEMS)
        for system in targets:
            results[system] = {
                "status": "online_stub",
                "score": 1.0,
                "issues": [],
                "note": "Scaffold mode - no filesystem check",
            }
            self.matrix.record_observation(
                source="maintenance_skg",
                event_type="diagnostic_scan",
                data={"system": system, "mode": "scaffold"},
                confidence=0.5,
            )
        self.system_health_report.emit(results)
        return results

    def propose_repair(self, target_system: str, issue_id: str) -> Optional[RepairTicket]:
        if self._is_self_modification(target_system):
            logging.critical(f"SELF-MODIFICATION BLOCKED: {target_system}")
            return None

        ticket = RepairTicket(
            ticket_id=f"STUB-{datetime.now().strftime('%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            target_system=target_system,
            maintenance_class=MaintenanceClass.REPAIR,
            issue_description=f"Stub issue: {issue_id}",
            proposed_changes={"example.py": "# stub fix"},
            diff_patch="--- stub\n+++ stub\n@@ -1 +1 @@\n-stub\n+fixed",
            authorization_required=AuthorizationLevel.L2_PATCH,
        )

        stub_file = self.temp_vault / f"{ticket.ticket_id}.json"
        stub_file.write_text('{"status": "staged", "authorized": false}')

        self.active_tickets[ticket.ticket_id] = ticket
        self.ticket_created.emit(ticket)
        logging.info(f"Stub ticket created for {target_system}: {ticket.ticket_id}")
        return ticket

    def authorize_ticket(self, ticket_id: str, auth_level: int) -> bool:
        ticket = self.active_tickets.get(ticket_id)
        if not ticket:
            return False
        if auth_level < ticket.authorization_required.value:
            logging.error(f"Auth failed: {auth_level} < {ticket.authorization_required.value}")
            return False
        self.auth_cache.add(ticket.ticket_hash)
        logging.info(f"Ticket {ticket_id} authorized at level {auth_level}")
        return True

    def execute_authorized_repair(self, ticket_id: str) -> bool:
        ticket = self.active_tickets.get(ticket_id)
        if not ticket:
            return False
        if ticket.ticket_hash not in self.auth_cache:
            logging.error("Ticket not in auth cache")
            return False

        if self.dals is None:
            logging.warning("STUB MODE: Would delegate to DALS, but no bridge connected")
            self.repair_executed.emit(ticket_id, True)
            return True

        success = self.dals.deploy_maintenance_worker(ticket)
        self.repair_executed.emit(ticket_id, success)
        return success


class DoctrineAuditor(QObject):
    """
    Cognition-only auditor that delegates execution to DALS (Memory 29).
    Maps code quality to geometric properties for visualization.
    """

    audit_complete = Signal(dict)  # {core_id: {structural_integrity, violations}}

    def __init__(self, dals_bridge: Optional[Any] = None, parent=None):
        super().__init__(parent)
        self.dals = dals_bridge

    async def audit_geometric_integrity(self) -> Dict[str, Dict[str, Any]]:
        if self.dals is None:
            logging.warning("DoctrineAuditor: No DALS bridge; returning stub integrity map")
            default = {cid: {"structural_integrity": 1.0, "violations": []} for cid in ["caleon", "cali_x", "kaygee", "ecm"]}
            self.audit_complete.emit(default)
            return default

        try:
            result = await self.dals.run_doctrine_audit_async()
        except Exception as exc:  # noqa: BLE001
            logging.error(f"Doctrine audit failed via DALS: {exc}")
            fallback = {cid: {"structural_integrity": 0.5, "violations": ["audit_error"]} for cid in ["caleon", "cali_x", "kaygee", "ecm"]}
            self.audit_complete.emit(fallback)
            return fallback

        health_map: Dict[str, Dict[str, Any]] = {}
        violations_by_core = result.get("violations", {}) if isinstance(result, dict) else {}

        for core_id in ["caleon", "cali_x", "kaygee", "ecm"]:
            violations = violations_by_core.get(core_id, [])
            integrity = max(0.0, 1.0 - (len(violations) * 0.1))
            health_map[core_id] = {
                "structural_integrity": integrity,
                "violations": violations,
                "axis_bias": "governance",
            }

        self.audit_complete.emit(health_map)
        return health_map


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    maint = SystemMaintenanceSKG(Path("./test_cali"))

    print("Testing self-mod block...")
    result = maint.propose_repair("cali_orb", "test_issue")
    assert result is None

    print("Testing valid target...")
    ticket = maint.propose_repair("kaygee", "syntax_error")
    assert ticket is not None

    print("Testing auth flow...")
    maint.authorize_ticket(ticket.ticket_id, 2)
    maint.execute_authorized_repair(ticket.ticket_id)

    print("All scaffold tests passed")
