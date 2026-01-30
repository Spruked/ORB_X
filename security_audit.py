"""
Security Integrity Guardian (SIG) for ORB_X
Memory 27: File integrity, process auditing, architectural verification
Protects against: IDE/Copilot injection, unauthorized modifications,
architectural drift (hallucinated endpoints), and self-modification attacks.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
import time
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    import psutil  # type: ignore

    PSUTIL_AVAILABLE = True
except ImportError:  # pragma: no cover
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available - process auditing disabled")

try:
    from vault_logic_system.memory.consolidation import ImmutableMemoryMatrix  # type: ignore
except ImportError:  # pragma: no cover
    class ImmutableMemoryMatrix:  # type: ignore
        def record_observation(self, **kwargs):
            logging.info(f"[Memory] {kwargs}")


@dataclass
class IntegrityViolation:
    """Immutable record of security breach attempt."""

    timestamp: str
    file_path: str
    expected_hash: str
    actual_hash: str
    process_name: Optional[str]
    username: str
    violation_type: str  # modification, permission, architectural, file_deletion
    auto_rollback: bool = False


class FileIntegrityMonitor:
    """
    Memory 27: SHA256-based integrity monitoring.
    Windows-compatible (no Unix-specific permissions required).
    """

    PROTECTED_FILES = [
        "system_maintenance.py",
        "ucm_bridge.py",
        "orb_main.py",
        "space_field.py",
        "orb_gui.py",
    ]

    SEAL_FILE = ".integrity_seals.json"

    def __init__(self, watch_path: Path, memory_matrix: Optional[ImmutableMemoryMatrix] = None):
        self.watch_path = Path(watch_path)
        self.memory = memory_matrix
        self.seals: Dict[str, str] = {}
        self.violations: List[IntegrityViolation] = []
        self._load_seals()

        self.whitelist_processes: Set[str] = {"python.exe", "pythonw.exe", "git.exe"}
        self.blacklist_processes: Set[str] = {"Code.exe", "code-tunnel.exe"}

    def _load_seals(self):
        seal_path = self.watch_path / self.SEAL_FILE
        if seal_path.exists():
            self.seals = json.loads(seal_path.read_text())
        else:
            self._initialize_seals()

    def _initialize_seals(self):
        logging.info("Initializing integrity seals...")
        for filename in self.PROTECTED_FILES:
            file_path = self.watch_path / filename
            if file_path.exists():
                self.seals[str(file_path)] = self._calculate_hash(file_path)
        (self.watch_path / self.SEAL_FILE).write_text(json.dumps(self.seals, indent=2))
        logging.info(f"Sealed {len(self.seals)} files")

    def _calculate_hash(self, file_path: Path) -> str:
        h = hashlib.sha256()
        h.update(file_path.read_bytes())
        return h.hexdigest()

    def verify_integrity(self) -> Tuple[bool, List[IntegrityViolation]]:
        violations: List[IntegrityViolation] = []
        for file_path_str, expected_hash in self.seals.items():
            file_path = Path(file_path_str)
            if not file_path.exists():
                violation = IntegrityViolation(
                    timestamp=datetime.now().isoformat(),
                    file_path=str(file_path),
                    expected_hash=expected_hash,
                    actual_hash="FILE_MISSING",
                    process_name=None,
                    username=os.getlogin(),
                    violation_type="file_deletion",
                )
                violations.append(violation)
                continue

            actual_hash = self._calculate_hash(file_path)
            if actual_hash != expected_hash:
                process_info = self._detect_modifying_process(file_path)
                violation = IntegrityViolation(
                    timestamp=datetime.now().isoformat(),
                    file_path=str(file_path),
                    expected_hash=expected_hash,
                    actual_hash=actual_hash,
                    process_name=process_info.get("name") if process_info else None,
                    username=os.getlogin(),
                    violation_type="unauthorized_modification",
                )
                violations.append(violation)

                if self.memory:
                    self.memory.record_observation(
                        source="security_audit",
                        event_type="integrity_violation",
                        data=asdict(violation),
                        confidence=1.0,
                    )

        self.violations.extend(violations)
        return len(violations) == 0, violations

    def _detect_modifying_process(self, file_path: Path) -> Optional[Dict]:
        if not PSUTIL_AVAILABLE:
            return None
        try:
            current_pid = os.getpid()
            for proc in psutil.process_iter(["pid", "name", "username", "open_files"]):
                try:
                    if proc.pid == current_pid:
                        continue
                    if proc.info.get("open_files"):
                        for opened in proc.info["open_files"]:
                            if str(file_path) in str(opened.path):
                                return {"name": proc.info.get("name"), "pid": proc.pid, "user": proc.info.get("username")}
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as exc:  # pragma: no cover
            logging.error(f"Process detection error: {exc}")
        return None

    def update_seal(self, file_path: Path, authorized: bool = False):
        if not authorized:
            raise PermissionError("Seal update requires explicit authorization")
        file_path = Path(file_path)
        new_hash = self._calculate_hash(file_path)
        self.seals[str(file_path)] = new_hash
        (self.watch_path / self.SEAL_FILE).write_text(json.dumps(self.seals, indent=2))
        if self.memory:
            self.memory.record_observation(
                source="security_audit",
                event_type="seal_updated",
                data={"file": str(file_path), "new_hash": new_hash},
                confidence=1.0,
            )

    def generate_report(self) -> Dict:
        return {
            "total_files_monitored": len(self.seals),
            "violations_detected": len(self.violations),
            "last_check": datetime.now().isoformat(),
            "system_integrity": "COMPROMISED" if self.violations else "STABLE",
            "violations": [asdict(v) for v in self.violations[-10:]],
        }


class WindowsPermissionGuard:
    """Memory 27: Windows-specific icacls implementation."""

    @staticmethod
    def lock_file(file_path: Path):
        if platform.system() != "Windows":
            return
        try:
            cmd = [
                "icacls",
                str(file_path),
                "/deny",
                f"{os.getlogin()}:W",
                "/inheritance:r",
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            logging.info(f"File locked (Windows ACL): {file_path}")
        except Exception as exc:  # pragma: no cover
            logging.error(f"Failed to lock file {file_path}: {exc}")

    @staticmethod
    def unlock_file(file_path: Path):
        if platform.system() != "Windows":
            return
        try:
            cmd = [
                "icacls",
                str(file_path),
                "/grant",
                f"{os.getlogin()}:F",
                "/inheritance:e",
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            logging.info(f"File unlocked (Windows ACL): {file_path}")
        except Exception as exc:  # pragma: no cover
            logging.error(f"Failed to unlock file {file_path}: {exc}")


class ArchitecturalVerifier:
    """Detects drift: hallucinated endpoints and DALS separation violations."""

    def __init__(self, repo_root: Path):
        self.root = Path(repo_root)
        self.readme = self.root / "README.md"

    def verify_no_rest_endpoints(self) -> bool:
        if not self.readme.exists():
            return True
        content = self.readme.read_text().lower()
        claims_no_rest = "no rest" in content or "current surface (no rest)" in content
        if not claims_no_rest:
            return True

        rest_patterns = ["@app.get", "@app.post", "@app.route", "from flask import", "from fastapi import", "uvicorn.run", "app.run("]
        violations = []
        for py_file in self.root.rglob("*.py"):
            try:
                code = py_file.read_text()
                for pattern in rest_patterns:
                    if pattern in code:
                        violations.append((str(py_file), pattern))
            except Exception:
                continue
        if violations:
            logging.error(f"ARCHITECTURAL DRIFT: REST detected but README claims none: {violations}")
            return False
        return True

    def verify_dals_separation(self) -> bool:
        cali_path = self.root
        violations = []
        for py_file in cali_path.rglob("*.py"):
            if ".venv" in py_file.parts or "scripts" in py_file.parts:
                continue
            try:
                code = py_file.read_text()
                if "subprocess" in code or "os.system" in code:
                    for i, line in enumerate(code.split("\n")):
                        if "subprocess" in line and not line.strip().startswith("#"):
                            violations.append(f"{py_file}:{i+1} - subprocess found")
            except Exception:
                continue
        if violations:
            logging.error(f"MEMORY 29 VIOLATION: {violations}")
            return False
        return True


class ContinuousMonitor:
    """Background thread for continuous integrity monitoring."""

    def __init__(self, integrity_monitor: FileIntegrityMonitor, check_interval: int = 30):
        self.monitor = integrity_monitor
        self.interval = check_interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logging.info("Continuous security monitoring started")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _monitor_loop(self):
        while not self._stop_event.is_set():
            try:
                is_valid, violations = self.monitor.verify_integrity()
                if not is_valid:
                    logging.critical("INTEGRITY VIOLATION DETECTED")
                    for v in violations:
                        logging.critical(f"  {v.file_path} modified by {v.process_name}")
                time.sleep(self.interval)
            except Exception as exc:  # pragma: no cover
                logging.error(f"Monitor error: {exc}")


class GitHookInstaller:
    """Memory 27: Git hooks without chmod (Windows batch files)."""

    @staticmethod
    def install_pre_commit_hook(repo_path: Path):
        git_hooks = repo_path / ".git" / "hooks"
        if not git_hooks.exists():
            logging.warning("Not a git repo or .git not found")
            return

        if platform.system() == "Windows":
            hook_file = git_hooks / "pre-commit.bat"
            hook_content = """@echo off
echo [SIG] Running security audit...
python -c "from security_audit import FileIntegrityMonitor; m = FileIntegrityMonitor('.'); valid, _ = m.verify_integrity(); exit(0 if valid else 1)"
if errorlevel 1 (
    echo [SIG] Integrity check failed. Commit aborted.
    exit /b 1
)
echo [SIG] Integrity verified.
"""
        else:
            hook_file = git_hooks / "pre-commit"
            hook_content = """#!/bin/bash
echo "[SIG] Running security audit..."
python3 -c "from security_audit import FileIntegrityMonitor; m = FileIntegrityMonitor('.'); valid, _ = m.verify_integrity(); exit(0 if valid else 1)"
if [ $? -ne 0 ]; then
    echo "[SIG] Integrity check failed. Commit aborted."
    exit 1
fi
echo "[SIG] Integrity verified."
"""

        hook_file.write_text(hook_content)
        if platform.system() != "Windows":
            os.chmod(hook_file, 0o755)
        logging.info(f"Installed pre-commit hook: {hook_file}")


def initialize_security_layer(
    cali_orb_path: Path,
    memory_matrix: Optional[ImmutableMemoryMatrix] = None,
    enforce_acl: bool = False,
) -> Tuple[FileIntegrityMonitor, ContinuousMonitor]:
    monitor = FileIntegrityMonitor(cali_orb_path, memory_matrix)

    arch_verifier = ArchitecturalVerifier(cali_orb_path)
    if not arch_verifier.verify_no_rest_endpoints():
        logging.error("README/Implementation mismatch detected!")
    if not arch_verifier.verify_dals_separation():
        logging.error("DALS Separation violation!")

    if enforce_acl and platform.system() == "Windows":
        guard = WindowsPermissionGuard()
        for filename in monitor.PROTECTED_FILES:
            file_path = cali_orb_path / filename
            if file_path.exists():
                guard.lock_file(file_path)

    GitHookInstaller.install_pre_commit_hook(cali_orb_path)

    continuous = ContinuousMonitor(monitor, check_interval=30)
    continuous.start()
    return monitor, continuous


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    base = Path(".")
    monitor, continuous = initialize_security_layer(base)
    print("Security layer initialized. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        continuous.stop()
        print(monitor.generate_report())
