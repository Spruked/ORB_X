"""
Stress test: launch ORB UI and send rapid-fire queries to observe manifold tension/divergence.
Run with: python scripts/stress_test.py
"""

import asyncio
import sys

import qasync
from PySide6.QtWidgets import QApplication

from orb_gui import ORBWindow, build_cali

QUERIES = [
    "Synthesize a novel governance pattern",
    "How to process refunds with dual approval?",
    "Design a resilient worker escalation flow",
    "Map ontology axioms to UI hints",
    "Detect and resolve convergence tension in core minds",
]


async def fire_queries(window: ORBWindow, delay: float = 0.4):
    for q in QUERIES:
        await window._run_query(q)  # uses non-blocking bridge
        await asyncio.sleep(delay)


async def main():
    app = QApplication(sys.argv)
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    cali = build_cali()
    window = ORBWindow(cali)
    window.show()

    asyncio.create_task(fire_queries(window))

    app.aboutToQuit.connect(loop.stop)  # type: ignore[arg-type]
    with loop:
        await loop.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
