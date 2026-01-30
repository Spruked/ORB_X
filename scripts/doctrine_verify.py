"""
Doctrine verification utility.
- Detects hallucinated endpoints claimed in docs but absent in code.
- Enforces Memory 29: CALI side must not execute subprocess/os calls.
"""

import ast
import re
from pathlib import Path


class DoctrineVerifier:
    def __init__(self, manifest_path: str = "README.md"):
        manifest_file = Path(manifest_path)
        self.manifest = manifest_file.read_text() if manifest_file.exists() else ""

    def verify_endpoint_consistency(self) -> bool:
        """Extract endpoints from README and compare with FastAPI-style decorators."""
        claimed = set(re.findall(r"`(POST|GET|PUT|DELETE) (/[^`]+)`", self.manifest))
        actual = set()

        for py_file in Path(".").rglob("*.py"):
            if ".venv" in py_file.parts or "__pycache__" in py_file.parts:
                continue
            try:
                tree = ast.parse(py_file.read_text())
            except Exception:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and hasattr(node.func, "attr"):
                    if node.func.attr in {"get", "post", "put", "delete"}:
                        if node.args and isinstance(node.args[0], ast.Str):
                            actual.add((node.func.attr.upper(), node.args[0].s))

        hallucinated = claimed - actual
        if hallucinated:
            print(f"🚨 DOCTRINE VIOLATION: Hallucinated endpoints in README: {hallucinated}")
            return False
        return True

    def verify_dals_separation(self) -> bool:
        """Ensure no subprocess/os calls reside on CALI side (Memory 29)."""
        violations = []
        for py_file in Path(".").rglob("*.py"):
            if ".venv" in py_file.parts or "scripts" in py_file.parts:
                continue
            text = py_file.read_text(encoding="utf-8", errors="ignore")
            if "subprocess" in text or "os.system" in text:
                violations.append(py_file)

        if violations:
            print(f"🚨 MEMORY 29 VIOLATION: execution logic found in CALI scope: {violations}")
            return False
        return True


if __name__ == "__main__":
    verifier = DoctrineVerifier()
    assert verifier.verify_endpoint_consistency()
    assert verifier.verify_dals_separation()
    print("✅ Doctrine alignment verified")
