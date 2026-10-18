#!/usr/bin/env python3
"""Run on dev-orin from /root; check downloaded artifacts locally with NumPy/SciPy."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('action', choices=['run', 'check'])
p.add_argument('--phase', choices=['ideal', 'noisy'], required=True)
p.add_argument('--root', type=Path, required=True)
p.add_argument('--fixtures', type=Path, default=Path('/root/cluster-validation'))
p.add_argument('--replay-from', type=Path, help='Reuse unchanged simulated RGB from a previous matrix run')
a = p.parse_args()
matrix = a.fixtures/'acceptance_suites.json'
specs = [s for s in json.loads(matrix.read_text()) if a.phase == 'ideal' or not s.get('negative')]
scripts = Path(__file__).resolve().parent
noise = 0.1 if a.phase == 'noisy' else 0.0
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
fixture_hashes = {f: digest(a.fixtures/f) for s in specs for f in [s['scenarios'], s['poses']]}
phase_file = a.root/(a.phase+'-phase.json')
if a.action == 'run':
    a.root.mkdir(parents=True, exist_ok=True)
    binaries = {name: digest(Path('tools')/name) for name in ['path_camera_sim', 'hsv_cluster_visualizer']}
    if a.replay_from:
        source = json.loads((a.replay_from/(a.phase+'-phase.json')).read_text())
        if (source['binaries']['path_camera_sim'] != binaries['path_camera_sim'] or
            source['matrix'] != digest(matrix) or source['fixtures'] != fixture_hashes):
            p.error('replay source must use the same simulator, matrix, and fixtures')
    if a.phase == 'noisy':
        ideal = json.loads((a.root/'ideal-report.json').read_text())
        if (not ideal['passed'] or ideal['binaries'] != binaries or ideal['matrix'] != digest(matrix)
            or any(ideal['fixtures'].get(f) != h for f, h in fixture_hashes.items())):
            p.error('noisy phase requires a passing ideal report for these same C++ binaries')
    phase_file.unlink(missing_ok=True)
    for s in specs:
        if a.replay_from:
            name = s['name']+'-'+a.phase
            subprocess.run([sys.executable, str(scripts/'replay_gamepiece_validation.py'),
                            str(a.replay_from/name), str(a.root/name)], check=True)
            continue
        cmd = [sys.executable, str(scripts/'run_gamepiece_validation.py'),
               '--root', str(a.root/(s['name']+'-'+a.phase)), '--scenarios', str(a.fixtures/s['scenarios']),
               '--poses', str(a.fixtures/s['poses']), '--seeds', ','.join(map(str, s['seeds'])), '--noise', str(noise)]
        if s.get('only'): cmd += ['--only', ','.join(s['only'])]
        subprocess.run(cmd, check=True)
    if any(digest(Path('tools')/name) != h for name, h in binaries.items()):
        raise RuntimeError('C++ binaries changed during acceptance')
    phase_file.write_text(json.dumps({'binaries': binaries, 'matrix': digest(matrix), 'fixtures': fixture_hashes}))
else:
    metadata = json.loads(phase_file.read_text())
    errors, reports, observed_counts, full_six_frames = [], {}, set(), 0
    if metadata['matrix'] != digest(matrix): errors.append('acceptance matrix changed')
    if metadata['fixtures'] != fixture_hashes: errors.append('acceptance fixtures changed')
    reports_dir = a.root/'reports'; reports_dir.mkdir(exist_ok=True)
    for s in specs:
        name = s['name']+'-'+a.phase; suite = a.root/name
        scenarios = json.loads((a.fixtures/s['scenarios']).read_text())
        if s.get('only'): scenarios = {n: scenarios[n] for n in s['only']}
        expected = {f'{n}-{seed}' for n in scenarios for seed in s['seeds']}
        if {m.parent.name for m in suite.glob('*/manifest.json')} != expected: errors.append(name+': wrong run coverage')
        frames = len(json.loads((a.fixtures/s['poses']).read_text())['poses'])
        for case, clusters in scenarios.items():
            if len(clusters) > 6: errors.append(name+': exceeds six clusters')
            for seed in s['seeds']:
                run = suite/f'{case}-{seed}'; truth = json.loads((run/'manifest.json').read_text())
                obs = json.loads((run/'clusters/clusters.json').read_text())
                if (len(truth['frames']) != frames or len(obs) != frames or truth['fuel_clusters'] != clusters
                    or truth['fuel_seed'] != seed or truth['fuel_noise'] != noise): errors.append(str(run)+': fixture mismatch')
                for f in truth['frames']:
                    counts = [sum(v >= 8 and piece['cluster_id'] == g for piece, v in zip(truth['gamepieces'], f['visible_fuel_pixels'])) for g in range(len(clusters))]
                    if s.get('full') and any(n < 4 for n in counts): errors.append(str(run)+': nominal pile not visible')
                    if len(clusters) == 6 and all(n >= 4 for n in counts) and all(f['visible_cluster_centroids']): full_six_frames += 1
        cmd = [sys.executable, str(scripts/'evaluate_gamepiece_validation.py'), str(suite), '--output', str(reports_dir/(name+'.json'))]
        if s.get('centroid_gate'): cmd += ['--allow-partial', '--centroid-gate']
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode not in (0, 1): raise RuntimeError(result.stderr)
        report = json.loads(result.stdout); reports[name] = report
        if not report['passed']: errors.append(name+': failed metrics')
        for r in report['runs']:
            observed_counts.update(pair[0] for pair in r['counts'])
        if s.get('centroid_gate'):
            subprocess.run([sys.executable, str(scripts/'evaluate_gamepiece_validation.py'), str(suite),
                            '--output', str(reports_dir/(name+'-strict.json'))], stdout=subprocess.DEVNULL, check=False)
        print(name, report['passed'], flush=True)
    if not {4, 5, 6} <= observed_counts or full_six_frames == 0: errors.append('missing complex full-view coverage')
    result = {'passed': not errors, 'phase': a.phase, 'binaries': metadata['binaries'],
              'matrix': metadata['matrix'], 'fixtures': fixture_hashes,
              'full_six_centroid_frames': full_six_frames, 'errors': errors,
              'runs': sum(len(r['runs']) for r in reports.values()),
              'frames': sum(x['frames'] for r in reports.values() for x in r['runs']),
              'suites': {n: {'passed': r['passed'], 'strict_passed': r['strict_passed']} for n, r in reports.items()}}
    (a.root/(a.phase+'-report.json')).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
