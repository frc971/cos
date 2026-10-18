#!/usr/bin/env python3
"""Fit nonnegative, zero-intercept partial-ball corrections; never use validation reports."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import nnls

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('reports', nargs='+', type=Path)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--total-evidence', action='store_true', help='Fit verified count plus total area/edge evidence jointly')
a = p.parse_args()
rows = []
for file in a.reports:
    report = json.loads(file.read_text())
    bad = {(r[0], r[1]) for r in report['geometry_errors']}
    rows.extend(r for r in report['observations'] if (r['run'], r['frame']) not in bad)
X = np.array([[r['verified'], *r['total_evidence']] if a.total_evidence else r['residual'] for r in rows])
counts = np.array([r['balls'] for r in rows], float)
hard = np.array([r['verified'] for r in rows], float)
# Equal relative-count importance across sparse and dense piles.
target = counts if a.total_evidence else np.maximum(0, counts-hard)
coefficients, _ = nnls(X/counts[:, None], target/counts)
estimates = np.maximum(hard, X @ coefficients) if a.total_evidence else hard + X @ coefficients
offset = int(a.total_evidence)
result = dict(area_coefficient=float(coefficients[offset]), edge_coefficient=float(coefficients[offset+1]),
              intercept=0, observations=len(rows), runs=len({r['run'] for r in rows}),
              median_absolute_count_error_fraction=float(np.median(abs(estimates-counts)/counts)),
              sources={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in a.reports})
if a.total_evidence: result['verified_coefficient'] = float(coefficients[0])
a.output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
