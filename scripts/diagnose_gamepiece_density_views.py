#!/usr/bin/env python3
"""Ablate frozen density outputs; oracle positions are diagnostics, never tracker inputs."""
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('root', type=Path, help='Downloaded metadata containing ideal/ and noisy/')
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
radius = .075

def covariance(points):
    return np.cov(points.T, bias=True) if len(points) > 1 else np.zeros((2, 2))

def area(cov):
    return float(2*np.pi*np.sqrt(np.linalg.det(cov + np.eye(2)*radius**2/4)))

def match(errors, limit):
    n, m = errors.shape
    costs = np.concatenate((np.where(errors <= limit, errors, np.inf), np.full((n, n), n+1.0)), axis=1)
    return [(r, c) for r, c in zip(*linear_sum_assignment(costs)) if c < m]

def percentiles(values):
    return dict(zip(('median', 'p90', 'maximum'), np.percentile(values, [50, 90, 100]).tolist()))

result = {'reference': 'Identical image-visible ball identities across every view of both piles',
          'span': 'maximum estimated pile ratio / minimum across the run', 'conditions': {}}
for condition in ('ideal', 'noisy'):
    records, hashes, formula_errors = defaultdict(dict), {}, []
    for manifest in sorted((a.root/condition).glob('*/manifest.json')):
        data = json.loads(manifest.read_text())
        file = manifest.parent/'clusters/clusters.json'
        frames = json.loads(file.read_text())
        for source in (manifest, file):
            hashes[str(source.relative_to(a.root))] = hashlib.sha256(source.read_bytes()).hexdigest()
        balls = np.array([b['translation_m'][:2] for b in data['gamepieces']])
        ids = np.array([b['cluster_id'] for b in data['gamepieces']])
        for fi, (truth, frame) in enumerate(zip(data['frames'], frames, strict=True)):
            pose = frame['robot_pose']; angle = pose['yaw']
            R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            origin = np.array([pose['x'], pose['y']])
            visible = np.array(truth['visible_fuel_pixels']) > 0
            robot_balls = (balls-origin) @ R
            groups = [g for g in range(len(data['fuel_clusters'])) if np.any(visible & (ids == g))]
            clouds = [balls[visible & (ids == g)] for g in groups]
            centers = np.array([cloud.mean(0) for cloud in clouds])
            clusters = frame['clusters']
            positions = np.array([R @ c['ground_position_m'] + origin for c in clusters])
            pairs = match(np.linalg.norm(centers[:, None]-positions[None, :], axis=2), .9)
            hard = np.array([point for c in clusters for point in c['verified_ball_centers_m']]).reshape(-1, 2)
            snapped = hard.copy()
            errors = np.linalg.norm(robot_balls[visible, None]-hard[None, :], axis=2)
            for i, j in match(errors, .15): snapped[j] = robot_balls[visible][i]
            offsets = np.cumsum([0]+[c['verified_ball_count'] for c in clusters])
            for i, j in pairs:
                c = clusters[j]; n = c['verified_ball_count']; C = np.array(c['ground_covariance_m2'])
                cloud_C = covariance((clouds[i]-origin) @ R); N = len(clouds[i])
                snapped_C = covariance(snapped[offsets[j]:offsets[j+1]]) if n >= 3 else C
                if n >= 3:
                    hard_C = covariance(np.array(c['verified_ball_centers_m']))
                    formula_errors.append(abs(area((n*hard_C+3*C)/(n+3))-c['density_area_m2']))
                records[manifest.parent.name, groups[i]][fi] = dict(
                    views=len(frames), density=c['density'], count=c['estimated_ball_count'],
                    area=c['density_area_m2'], old_area=area(C),
                    snapped_area=area((n*snapped_C+3*C)/(n+3)) if n >= 3 else area(C),
                    complete_area=area((N*cloud_C+3*C)/(N+3)),
                    members=tuple(np.where(visible & (ids == groups[i]))[0]),
                    centroid_error=float(np.linalg.norm(centers[i]-positions[j])))
    spans, pair_count = defaultdict(list), 0
    for run in sorted({r for r, g in records}):
        for g, h in itertools.combinations(sorted(g for r, g in records if r == run), 2):
            x, y = records[run, g], records[run, h]
            fs = sorted(x.keys() & y.keys())
            if len(fs) != x[fs[0]]['views']: continue
            if any(len({row[f]['members'] for f in fs}) != 1 for row in (x, y)): continue
            pair_count += 1
            for key in ('density', 'count', 'area', 'old_area', 'snapped_area', 'complete_area'):
                ratios = [x[f][key]/y[f][key] for f in fs]
                spans[key].append(max(ratios)/min(ratios))
    if not pair_count: p.error(f'No fixed-membership pile pairs in {condition}')
    result['conditions'][condition] = dict(
        fixed_membership_pairs=pair_count, ratio_spans={k: percentiles(v) for k, v in spans.items()},
        centroid_error_m_all_observations=percentiles([v['centroid_error'] for rows in records.values() for v in rows.values()]),
        reproduced_area_max_error_m2=max(formula_errors), metadata_sha256=hashes)

# Exact projected outline, equal focal lengths, no pixel noise, perfect center depth.
# Integrate the production residual-arc weight along a fully visible sphere.
angles = np.arange(20000)*2*np.pi/20000
result['isolated_sphere_arc_normalization'] = []
for degrees in (0, 20, 35, 45):
    depth, focal = 4.0, 1000.0
    center = np.array([depth*np.tan(np.deg2rad(degrees)), 0, depth])
    axis = center/np.linalg.norm(center)
    first = np.cross(axis, [0, 1, 0]); first /= np.linalg.norm(first)
    second = np.cross(axis, first); sine = radius/np.linalg.norm(center)
    direction = np.sqrt(1-sine*sine)*axis + sine*(np.cos(angles)[:, None]*first + np.sin(angles)[:, None]*second)
    ray = direction/direction[:, 2, None]; pixel = focal*ray[:, :2]
    length = np.linalg.norm(np.roll(pixel, -1, axis=0)-pixel, axis=1)
    midpoint = (ray+np.roll(ray, -1, axis=0))/2
    evidence = float(sum(length*depth*np.linalg.norm(midpoint, axis=1)/(2*np.pi*radius*focal)))
    result['isolated_sphere_arc_normalization'].append(dict(off_axis_degrees=degrees, arc_evidence=evidence, expected=1))
a.output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k: {f: v for f, v in c.items() if f != 'metadata_sha256'} for k, c in result['conditions'].items()}, indent=2))
print(json.dumps(result['isolated_sphere_arc_normalization'], indent=2))
