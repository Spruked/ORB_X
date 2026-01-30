# ORB_X - Desktop Control Interface ![License](https://img.shields.io/badge/License-Proprietary-red.svg)

PySide6 desktop vessel for UCM Core-4 cognition with live 3D manifold visualization, governance enforcement, and DALS-separated maintenance.

⚠️ Contributor access is restricted. See CONTRIBUTOR_ACCESS_NOTICE.md.
💼 Commercial use requires a separate license. See LICENSE_COMMERCIAL_STUB.md.

## Architecture

```
ORB_X (PySide6 UI)
    ↓ qasync event loop
UCMCognitiveBridge (ucm_bridge.py)
    ↓ Core-4 manifold + worker dispatch
CALI (orb_main.py)
    ├── Doctrine validation
    ├── Vaults (apriori/posteriori)
    ├── Worker Swarm + Forges (stubbed)
    └── Maintenance SKG (delegates to DALS)
Visualization (space_field.py)
    ├── Convergence geometry
    ├── Integrity overlays
    └── Shadow/ghost trails
```

## Requirements

- Python 3.8+
- See `requirements.txt` (PySide6, qasync, matplotlib, numpy, lint stack)

## Quick Start

```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows
pip install -r requirements.txt
python main.py
```

## Current Surface (no REST)

- Desktop UI only via `main.py`
- No REST API implemented (planned v2.0)

## Key Files

- `main.py` — qasync entrypoint
- `orb_gui.py` — UI shell wired to `ORBXController`
- `ucm_bridge.py` — Cognitive bridge (Core-4 traversal, worker dispatch)
- `orb_main.py` — CALI cognition, doctrine, vaults, workers/forges
- `space_field.py` — 3D manifold, convergence geometry, integrity overlay
- `system_maintenance.py` — Maintenance SKG + DoctrineAuditor (delegates to DALS)
- `ecm_contract.json` — Runtime stability tests
- `scripts/stress_test.py` / `scripts/stress_test_high_tension.py` — Load tests
- `scripts/doctrine_verify.py` — Docs/Memory 29 checks

## Usage Notes

- Queries flow through `ORBXController` → `UCMCognitiveBridge` → CALI pipeline.
- Integrity overlays and shadows are visual only; SoftMax is advisory and does not move points.
- Maintenance execution must occur in DALS; CALI side performs cognition and staging only.

## Licensing / Patent Intent

© 2026 TrueMark UCM. All rights reserved. The authors intend to pursue patent protection for core orchestration, manifold visualization, and governance mechanisms embodied herein.