"""
ORB_X ignition sequence: create folders, write manifest, and verify core deps.
Non-destructive and safe to rerun.
"""

import json
import os
import sys
from pathlib import Path


def setup_orb_vessel() -> None:
    print("\U0001f30c Initializing ORB_X Environment...")

    root = Path.cwd()
    folders = [
        root / "orb_x",
        root / "orb_x" / "core",
        root / "orb_x" / "ui",
        root / "orb_x" / "dals",
        root / "orb_x" / "logs",
        root / "posteriori_vault" / "converged",
    ]

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
        print(f"  [+] ensured {folder.relative_to(root)}")

    manifest = {
        "orb_name": "ORB_X_PRIMARY",
        "version": "2026.1.29",
        "doctrine_path": "./core/spine.ontology",
        "core_minds": ["caleon", "cali_x", "kaygee", "ecm"],
        "governance": {
            "consensus_radius": 0.25,
            "tension_threshold": 0.8,
            "ghost_decay": 0.1,
        },
        "visualizer_settings": {
            "animation_fps": 30,
        },
    }

    manifest_path = root / "orb_x" / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=4))
    print(f"  [+] wrote {manifest_path.relative_to(root)}")

    requirements = [
        "PySide6>=6.5.0",
        "matplotlib>=3.7.0",
        "numpy>=1.24.0",
        "qasync>=0.24.0",
        "aiohttp>=3.8.0",
    ]
    req_path = root / "requirements.txt"
    req_path.write_text("\n".join(requirements))
    print(f"  [+] wrote {req_path.relative_to(root)}")

    print("\n\U0001f50d Verifying bindings...")
    missing = []
    try:
        import PySide6  # noqa: F401
        print("  [OK] PySide6 present")
    except ImportError as e:  # pragma: no cover
        missing.append(str(e))
    try:
        import matplotlib  # noqa: F401
        print("  [OK] matplotlib present")
    except ImportError as e:  # pragma: no cover
        missing.append(str(e))
    try:
        import numpy  # noqa: F401
        print("  [OK] numpy present")
    except ImportError as e:  # pragma: no cover
        missing.append(str(e))
    try:
        import qasync  # noqa: F401
        print("  [OK] qasync present")
    except ImportError as e:  # pragma: no cover
        missing.append(str(e))

    if missing:
        print("  [!] Missing dependencies detected:")
        for m in missing:
            print(f"      - {m}")
        print("  [TIP] Run: pip install -r requirements.txt")
    else:
        print("  [OK] All critical dependencies found.")

    print("\n\U0001f680 Environment ready. You can now launch ORB_X.")


if __name__ == "__main__":
    setup_orb_vessel()
