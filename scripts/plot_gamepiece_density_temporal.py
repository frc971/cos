#!/usr/bin/env python3
"""Create a standalone Plotly review of fixed, preselected pile pairs and step response."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('metadata', type=Path, help='Downloaded accepted/ metadata')
p.add_argument('--results', type=Path, default=Path('validation/temporal-density'))
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
fig = make_subplots(rows=3, cols=1, vertical_spacing=.12, subplot_titles=(
    'Same visible balls: forward motion (pile groups 1 / 4)',
    'Wider angular path: visibility can change (pile groups 1 / 4)',
    'Scene changes: 25 → 9 → 25 balls'))
styles = [('baseline', 'Previous method', '#D55E00'), ('stateless', 'Normalized, no history', '#CC79A7'),
          ('filter', 'Filter only', '#56B4E9'), ('predict', 'Persistent model', '#009E73')]

def observations(file):
    return {(r['run'], r['frame'], r['group']): r for r in json.loads(file.read_text())['observations']}

def reference(mode, run, g=1, h=4):
    m = json.loads((a.metadata/mode/run/'manifest.json').read_text())
    positions = np.array([b['translation_m'][:2] for b in m['gamepieces']])
    ids = np.array([b['cluster_id'] for b in m['gamepieces']])
    values = []
    for frame in m['frames']:
        visible = np.array(frame['visible_fuel_pixels']) > 0
        densities = []
        for group in (g, h):
            points = positions[visible & (ids == group)]
            if not len(points): break
            C = np.cov(points.T, bias=True) if len(points) > 1 else np.zeros((2, 2))
            densities.append(len(points)/(2*np.pi*np.sqrt(np.linalg.det(C+np.eye(2)*.075**2/4))))
        values.append(densities[0]/densities[1] if len(densities) == 2 else None)
    return m, values

for row, mode, run in [(1, 'ideal', 'six_dense_steps-1009'), (2, 'angular', 'six_dense_steps-5003')]:
    manifest, expected = reference(mode, run)
    xs = [f['robot_pose']['translation_m'][0] for f in manifest['frames']] if row == 1 else list(range(len(expected)))
    for key, label, color in styles:
        if key == 'baseline':
            file = Path('validation/density/ideal-validation.json') if row == 1 else a.results/'angular-baseline-accuracy.json'
        else:
            suffix = '' if key == 'predict' else '-'+key
            file = a.results/(mode+suffix+'-accuracy.json')
        data = observations(file); values = []
        for fi in range(len(xs)):
            g, h = data.get((run, fi, 1)), data.get((run, fi, 4))
            values.append(g['estimated']/h['estimated'] if g and h else None)
        fig.add_trace(go.Scatter(x=xs, y=values, mode='lines+markers', name=label,
                                line=dict(color=color, width=2), legendgroup=key, showlegend=row == 1,
                                connectgaps=False), row=row, col=1)
    fig.add_trace(go.Scatter(x=xs, y=expected, mode='lines', name='Visible reference',
                            line=dict(color='#222222', dash='dash', width=2), showlegend=row == 1), row=row, col=1)
    fig.update_yaxes(title_text='Density ratio', row=row, col=1)
    fig.update_xaxes(title_text='Camera field X (m)' if row == 1 else 'Frame (camera yaw +25.8° → −25.8°)', row=row, col=1)

case = next(t for t in ET.parse(a.results/'unit-results.xml').iter('testcase') if t.attrib['name'] == 'DensityHistoryRespondsToRemovalAndAddition')
values = {int(x.attrib['name'][5:]): float(x.attrib['value']) for x in case.iter('property') if x.attrib['name'].startswith('step_')}
assert sorted(values) == list(range(20))
fig.add_trace(go.Scatter(x=list(range(20)), y=[values[i] for i in range(20)], mode='lines+markers',
                        name='Persistent model', line=dict(color='#009E73', width=2), showlegend=False), row=3, col=1)
fig.add_trace(go.Scatter(x=list(range(20)), y=[25]*8+[9]*6+[25]*6, mode='lines', name='Actual count',
                        line=dict(color='#222222', dash='dash', width=2, shape='hv'), showlegend=False), row=3, col=1)
for x in (7.5, 13.5): fig.add_vline(x=x, line_dash='dot', line_color='#888888', row=3, col=1)
fig.update_xaxes(title_text='Processed frame', row=3, col=1)
fig.update_yaxes(title_text='Ball count', row=3, col=1)
fig.update_layout(title='Density continuity: frozen C++ outputs and independent reference', height=1150,
                  template='plotly_white', margin=dict(t=140, b=120),
                  legend=dict(orientation='h', y=1.09), hovermode='x unified')
fig.add_annotation(text='Forward pair was selected using the previous method’s median span. Angular pair uses the same group IDs.<br>'
                        'Gaps indicate unavailable pile ratios. The angular reference changes when visibility changes. All curves are causal outputs.',
                   xref='paper', yref='paper', x=0, y=-.09, showarrow=False, align='left', font=dict(size=11))
html = fig.to_html(include_plotlyjs=True, full_html=True)
a.output.write_text('\n'.join(line.rstrip() for line in html.splitlines())+'\n')
print(a.output)
