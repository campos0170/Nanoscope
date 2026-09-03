"""GUI layer: the application shell, trace viewer, docks and panels.

ARCHITECTURE (Rule 1): every panel holds exactly ONE reference — to `Session` — and communicates
only by calling Session mutators and subscribing to Session signals. No panel imports another
panel. `main_window` is the sole exception: it constructs every panel and wires each to `Session`
(and to itself), never to one another. `tests/test_viewer_contracts.py::test_no_cross_panel_imports`
enforces this by walking the AST of every module here.
"""
