"""Isolate the lifetime test plug-in from the suite-wide ``conftest`` module.

Keeping this directory as a package makes pytest import its local configuration as
``lifetime.conftest`` instead of replacing ``sys.modules['conftest']``.  Several older
regression tests import shared builders from the suite-wide module directly, so the
replacement otherwise makes a full-suite collection depend on directory traversal order.
"""
