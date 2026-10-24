#!/usr/bin/env python3
"""Build the E1 encoder-comparison summary table from report JSONs (card C-32).
Usage: python eval/e1cmp_summary.py <evidence_dir> [--e1b-dir docs/evidence/2026-10-12_E1_learning_curve] [--readme]
Series: E1b seed 0 (e1b_ep<N>_report.json in --e1b-dir or <evidence_dir>; epochs 41-45 = post-recipe extension), E1b seed 1
(e1b_s1_ep<N>), E1(C'), E1(D'), and every E1(<arm>) series found as e1_<short>_s0_ep<N>_report.json in <evidence_dir>.
Columns per encoder: dev PDMS (sub-scores in a second table); paired deltas from the D5 JSONs when present.
--readme rewrites the block between <!-- RESULTS:BEGIN --> and <!-- RESULTS:END --> in <evidence_dir>/README.md."""
import json, glob, os, re, sys, statistics as st
ev = sys.argv[1]
e1b_dir = sys.argv[sys.argv.index('--e1b-dir') + 1] if '--e1b-dir' in sys.argv else 'docs/evidence/2026-10-12_E1_learning_curve'
def first(*paths):
    for p in paths:
        if os.path.exists(p): return p
    return paths[-1]
ENC = [('E1b (e50) seed 0', 'e1b', lambda n: first(f'{e1b_dir}/e1b_ep{n}_report.json', f'{ev}/e1b_ep{n}_report.json')),
       ('E1b (e50) seed 1', 'e1b_s1', lambda n: f'{ev}/e1b_s1_ep{n}_report.json'),
       ("E1(C') comma plain", 'e1_Cp_s0', lambda n: f'{ev}/e1_Cp_s0_ep{n}_report.json'),
       ("E1(D') comma + motion", 'e1_Dp_s0', lambda n: f'{ev}/e1_Dp_s0_ep{n}_report.json')]
ARM_NAMES = {'Dpshuf': "E1(D'-shuf) shuffled-CAN control", 'D': 'E1(D) mix + motion', 'C': 'E1(C) mix plain', 'A': 'E1(A) public plain',
             'B': 'E1(B) public + motion', 'Dshuf': 'E1(D-shuf) shuffled-CAN control', 'Dpfields': "E1(D'-fields) + comma CAN fields"}
for short in sorted({m for g in glob.glob(f'{ev}/e1_*_s0_ep*_report.json') for m in re.findall(r'/e1_([A-Za-z]+)_s0_ep\d+_report', g)} - {'Cp', 'Dp'}):
    ENC.append((ARM_NAMES.get(short, f'E1({short})'), f'e1_{short}_s0', (lambda s: lambda n: f'{ev}/e1_{s}_s0_ep{n}_report.json')(short)))
SUB = [('NC', 'no_at_fault_collisions'), ('DAC', 'drivable_area_compliance'), ('EP', 'ego_progress'),
       ('TTC', 'time_to_collision_within_bound'), ('C', 'comfort'), ('DDC', 'driving_direction_compliance')]
def load(f):
    return json.load(open(f)) if os.path.exists(f) else None
epochs = sorted({n for _, _, f in ENC for n in range(1, 61) if load(f(n))})
def delta(a, b):
    f = f'{ev}/report_{a}_vs_{b}.json'
    if not os.path.exists(f): return None
    r = [x for x in json.load(open(f))['diff'] if x['bucket'] == 'all'][0]
    return r['delta_PDMS'], r['ci_low'], r['ci_high']
fmt = lambda d: '–' if d is None else f'{d[0]:+.2f} [{d[1]:.2f}, {d[2]:.2f}]'
out = []
out.append('| epoch | ' + ' | '.join(e[0] for e in ENC) + " | Δ s1−s0 (paired) | Δ C'−E1b (paired) | Δ D'−E1b (paired) | Δ D'−official | Δ C'−official |")
out.append('|---:|' + '---:|' * (len(ENC) + 5))
for n in epochs:
    cells = []
    for _, _, f in ENC:
        r = load(f(n)); cells.append(f"{r['score']:.2f}" if r else '–')
    out.append(f'| {n} | ' + ' | '.join(cells) + f" | {fmt(delta(f'e1b_ep{n}', f'e1b_s1_ep{n}'))} | {fmt(delta(f'e1b_ep{n}', f'e1_Cp_s0_ep{n}'))} | {fmt(delta(f'e1b_ep{n}', f'e1_Dp_s0_ep{n}'))} | {fmt(delta('e0_official', f'e1_Dp_s0_ep{n}'))} | {fmt(delta('e0_official', f'e1_Cp_s0_ep{n}'))} |")
out.append('')
out.append("Late exports: epochs 36–40 (protocol C-29). E1b seed 0 was trained before the per-epoch export existed (exports 5/10/…/40), so its late "
           "window is the post-recipe extension 41–45 (same constant lr 1e-4, decided 2026-10-16 ④); treat it as a plateau sample, not as epochs 36–40.")
out.append('')
out.append('| encoder | epochs evaluated | late-5 mean ± SD | dev-selected best (epoch) | all evaluated epochs ≥25: mean ± SD |')
out.append('|---|---|---:|---:|---:|')
for name, key, f in ENC:
    vals = {n: load(f(n))['score'] for n in epochs if load(f(n))}
    window = range(41, 46) if key == 'e1b' else range(36, 41)
    late = [vals[n] for n in window if n in vals]
    ge25 = [vals[n] for n in vals if n >= 25]
    best = max(vals.items(), key=lambda kv: kv[1]) if vals else None
    sd = lambda v: f'{st.mean(v):.2f} ± {st.pstdev(v) if len(v) < 2 else st.stdev(v):.2f}' if v else '–'
    if best:
        out.append(f"| {name} | {', '.join(map(str, sorted(vals)))} | {sd(late)} (n={len(late)}{'; 41–45' if key == 'e1b' else ''}) | {best[1]:.2f} ({best[0]}) | {sd(ge25)} (n={len(ge25)}) |")
    else:
        out.append(f'| {name} | – | – | – | – |')
out.append('')
out.append('Sub-scores (dev, %):')
out.append('')
out.append('| run | epoch | PDMS | ' + ' | '.join(s for s, _ in SUB) + ' |')
out.append('|---|---:|---:|' + '---:|' * len(SUB))
for name, _, f in ENC:
    for n in epochs:
        r = load(f(n))
        if r: out.append(f"| {name} | {n} | {r['score']:.2f} | " + ' | '.join(f"{r[c]:.2f}" for _, c in SUB) + ' |')
text = '\n'.join(out)
if '--readme' in sys.argv:
    rp = os.path.join(ev, 'README.md'); r = open(rp, encoding='utf-8').read()
    b, e = '<!-- RESULTS:BEGIN -->', '<!-- RESULTS:END -->'
    assert b in r and e in r, 'README markers missing'
    r = r[: r.index(b) + len(b)] + '\n' + text + '\n' + r[r.index(e):]
    open(rp, 'w', encoding='utf-8').write(r); print(f'README results block refreshed: {rp}')
else:
    print(text)
