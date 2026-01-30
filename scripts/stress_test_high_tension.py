"""
High-tension stress test: drive UCMCognitiveBridge directly with conflicting queries.
Headless: no UI window shown, but uses the same qasync loop and visualizer logic.
Run with: python scripts/stress_test_high_tension.py
"""

import asyncio
import sys

import qasync
from PySide6.QtWidgets import QApplication

from ucm_bridge import ORBXController
from space_field import SpaceFieldVisualizer

STRESS_QUERIES = [
    "Synthesize Axiom 2 with high-velocity I/O",
    "Force dissent on Caleon regarding structural recursion",
    "Analyze chaos theory vs. DALS stability",
    "Generate a recursive loop check for ECM",
    "Validate ghost trace decay at 500% speed",
]


async def run_stress_test(controller: ORBXController):
    print("INITIALIZING HIGH-TENSION STRESS TEST...")
    tasks = []
    for idx, query in enumerate(STRESS_QUERIES, start=1):
        await asyncio.sleep(0.2)
        print(f"Injecting Query {idx}: {query}")
        tasks.append(asyncio.create_task(controller.execute_query(query)))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    print("\n--- STRESS TEST COMPLETE ---")
    for idx, res in enumerate(results, start=1):
        status = "SUCCESS" if not isinstance(res, Exception) else f"FAILED: {res}"
        print(f"Query {idx}: {status}")


async def main():
    app = QApplication(sys.argv)
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    # Headless visualizer: not shown, but provides geometry/tension detection
    visualizer = SpaceFieldVisualizer()
    controller = ORBXController(visualizer)

    asyncio.create_task(run_stress_test(controller))

    app.aboutToQuit.connect(loop.stop)  # type: ignore[arg-type]
    with loop:
        await loop.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
