"""Readers. Import `base` for the registry; reader modules self-register on lazy import.

Nothing is imported here at module scope on purpose: `base.select_reader` imports the reader
modules inside a function, which is what keeps `base <- abf/hdf5/tdms/text` from being circular.
"""
