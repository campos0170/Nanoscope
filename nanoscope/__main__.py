"""`python -m nanoscope` -> launch the GUI.

Importing the package (which happens before this module runs) executes bootstrap, so the Qt
plugin path is fixed before gui.app imports PySide6.
"""
from __future__ import annotations

import sys

from .gui.app import main

if __name__ == "__main__":
    sys.exit(main(sys.argv))
