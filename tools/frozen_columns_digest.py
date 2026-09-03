"""Compare the 20 frozen EventSet columns between two trees on the owner's real recording.

METHOD, stated so it is reproducible: import nanoscope from the tree given as argv[1], open the
15 s apo ABF through the app's own reader, take channel 2 calibrated, run detect_events at its
defaults, then hash each frozen column's raw bytes INDEPENDENTLY (so a diff names the column).
"""
import sys, hashlib, json
import numpy as np
sys.path.insert(0, sys.argv[1])
import nanoscope
from nanoscope.analysis.events import SCHEMA
from nanoscope.analysis.detect import detect_events
from nanoscope.core.session import Session
from nanoscope.io.base import open_document

ABF = "/Users/abrahamcampos/500 pM Apo ferritin_1 x PBS_22 nm_0004_process _ -1000 mV_ 385-400 s time trace.abf"
s = Session()
did = s.add_document(open_document(ABF))
ch = s.calibrated(did, 2)
df, prov = detect_events(np.asarray(ch.data), ch.sample_rate_hz)
out = {"__tree__": nanoscope.__file__, "__n_rows__": int(len(df))}
for c in SCHEMA:
    if c not in df.columns:
        out[c] = "MISSING"; continue
    a = df[c].to_numpy()
    if a.dtype.kind in "OUS":
        a = np.asarray([("" if x is None else str(x)) for x in a], dtype="U64")
    out[c] = hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]
print(json.dumps(out))

# USAGE, and why it exists:
#   cd / && /opt/anaconda3/bin/python3 tools/frozen_columns_digest.py <tree>
# Run it against the working tree and against a preserved oracle tree and diff the two JSON
# lines. Every column is hashed SEPARATELY so a difference names the column that moved rather
# than telling you only that something did.
#
# It exists because three consecutive rounds of this campaign cited a single whole-frame hash
# ("c0f0f144e8e392ae…") as proof that no measured number moved, and a skeptic then could not
# reproduce that digest under any of seven conventions — the evidence chain had a broken link
# for a claim everything else rested on. A digest nobody else can recompute is not evidence.
#
# Measured 2026-09-02: current tree and ~/Desktop/nanoscope-reports/pre-wave-a-2026-09-02
# agree on all 20 columns, 115 rows, on the 15 s -1000 mV apo recording.
