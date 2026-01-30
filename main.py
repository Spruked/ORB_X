import asyncio
import sys
from pathlib import Path

from PySide6 import QtWidgets
import qasync

from orb_gui import ORBWindow, build_cali


def main():
    app = QtWidgets.QApplication(sys.argv)
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    cali = build_cali()
    window = ORBWindow(cali)
    window.show()

    # Ensure loop stops when the app closes
    app.aboutToQuit.connect(loop.stop)  # type: ignore[arg-type]

    with loop:
        loop.run_forever()


if __name__ == "__main__":
    main()
