#!/usr/bin/env python3
"""Measure within-frame accuracy separately from ratio drift and adjacent-frame jitter."""
import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('root', type=Path)
p.add_argument('--report', type=Path, required=True, help='Output of evaluate_gamepiece_density.py')
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
report = json.loads(a.report.read_text())
rows = {(r['run'], r['frame'], r['group']): r for r in report['observations']}
spans, adjusted_spans, jitter = [], [], []
per_run, total_reused, total_verified, visible_groups, predicted_groups = [], 0, 0, 0, 0
for manifest in sorted(a.root.glob('*/manifest.json')):
    data = json.loads(manifest.read_text()); run = manifest.parent.name
    frames = json.loads((manifest.parent/'clusters/clusters.json').read_text())
    ids = np.array([b['cluster_id'] for b in data['gamepieces']])
    membership = {}
    for fi, reference in enumerate(data['frames']):
        visible = np.array(reference['visible_fuel_pixels']) > 0
        for g in range(len(data['fuel_clusters'])):
            membership[fi, g] = tuple(np.where(visible & (ids == g))[0])
            visible_groups += bool(membership[fi, g])
        predicted_groups += len(frames[fi]['clusters'])
    run_spans, run_adjusted, run_jitter = [], [], []
    for g, h in itertools.combinations(range(len(data['fuel_clusters'])), 2):
        fs = [fi for fi in range(len(frames)) if (run, fi, g) in rows and (run, fi, h) in rows]
        if len(fs) < 2: continue
        ratios = [rows[run, f, g]['estimated']/rows[run, f, h]['estimated'] for f in fs]
        truth = [rows[run, f, g]['truth']/rows[run, f, h]['truth'] for f in fs]
        error = np.array(ratios)/truth
        run_adjusted.append(float(max(error)/min(error)))
        if len(fs) == len(frames) and all(len({membership[f, k] for f in fs}) == 1 for k in (g, h)):
            run_spans.append(float(max(ratios)/min(ratios)))
        for i in range(1, len(fs)):
            if fs[i] == fs[i-1]+1 and all(membership[fs[i], k] == membership[fs[i-1], k] for k in (g, h)):
                run_jitter.append(float(np.exp(abs(np.log(ratios[i]/ratios[i-1])))))
    spans += run_spans; adjusted_spans += run_adjusted; jitter += run_jitter
    per_run.append(dict(run=run, frames=len(frames), fixed_membership_pair_spans=run_spans,
                        reference_adjusted_pair_spans=run_adjusted, fixed_membership_adjacent_pair_changes=run_jitter))
    for frame in frames[1:]:
        for c in frame['clusters']:
            total_reused += c.get('reused_ball_count', 0); total_verified += c.get('verified_ball_count', 0)

def summary(values):
    return dict(samples=len(values), median=float(np.median(values)), p90=float(np.percentile(values, 90)), maximum=float(max(values))) if values else dict(samples=0)
result = dict(accuracy=report['summary'], fixed_membership_ratio_span=summary(spans),
              fixed_membership_adjacent_ratio_change=summary(jitter), reference_adjusted_ratio_span=summary(adjusted_spans),
              reused_fraction_after_first_frame=total_reused/total_verified if total_verified else None,
              geometry_coverage=dict(reference_visible_groups=visible_groups, predicted_groups=predicted_groups,
                                     matched_groups=len(rows), missed_groups=visible_groups-len(rows), extra_groups=predicted_groups-len(rows)), runs=per_run)
a.output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k: v for k, v in result.items() if k != 'runs'}, indent=2))
