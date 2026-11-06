"""Audit of the compact encoding on converted archives (convert.py --compact): per obj_float column and per token type, how
often it is used, its spread, and whether it looks like a quantity, an id / hash, or a bit field -- to curate the encoding
before training (dead columns, values that need an embedding or bits, heavy tails).

    python tools/compact_audit.py DATASET_DIR [DATASET_DIR ...]"""
import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cr.imitation import read                      # noqa: E402
from cr.compact import capture as CC               # noqa: E402
from cr.compact import encode_compact as E          # noqa: E402

files = [f for d in sys.argv[1:] for f in sorted(Path(d).glob('**/*.npz'))]
cols = [[] for _ in range(E.F)]
rows = 0
tok_types = collections.Counter()
tok_vals = collections.defaultdict(lambda: [[] for _ in range(E.VALUES_W)])
for f in files:
    try:
        d = read(f, 0)
    except Exception:
        continue
    if 'x_cobj_float' not in d:
        continue
    of = d['x_cobj_float'].astype(np.float32) * CC.OF_SCALE        # back to raw values
    keep = np.random.default_rng(0).random(len(of)) < 0.2            # a fifth of the rows is plenty
    of = of[keep]
    rows += len(of)
    for c in range(E.F):
        cols[c].append(of[:, c])
    ti, tf = d['x_ctok_int'], d['x_ctok_float'].astype(np.float32)
    for t, scale in CC.TF_SCALE.items():              # back to raw values, per token type
        tf[ti[..., 1] == t] *= scale
    m = ti[..., 0] >= 0
    for t in np.unique(ti[..., 1][m]):
        sel = m & (ti[..., 1] == t)
        tok_types[int(t)] += int(sel.sum())
        v = tf[sel]
        for k in range(E.VALUES_W):
            tok_vals[int(t)][k].append(v[:, k])
print(f'{len(files)} archives, {rows} object rows sampled')


def describe(v):
    nz = v[v != 0]
    if not len(nz):
        return 'never non-zero', None
    frac = len(nz) / len(v)
    distinct = len(np.unique(nz))
    big = np.abs(nz).max()
    q = np.percentile(np.abs(nz), [50, 99])
    kind = 'flag' if distinct <= 2 else 'small set' if distinct <= 12 else 'quantity'
    if big > 2 ** 20 and distinct > 50:
        kind = 'ID/HASH?'
    if q[1] > 0 and big / max(q[1], 1e-9) > 50:
        kind += ' heavy-tail'
    return f'used {frac:6.1%} distinct {distinct:6d} |v| median {q[0]:10.1f} p99 {q[1]:12.1f} max {big:14.1f}  {kind}', kind


groups = collections.Counter()
print('\n-- obj_float columns (raw values) --')
for c, name in enumerate(E.FLOAT_SLOTS):
    v = np.concatenate(cols[c]) if cols[c] else np.zeros(0)
    text, kind = describe(v)
    groups[(name.split('[')[0].split('.bit')[0] if not name.startswith('raw[') else 'raw', kind)] += 1
    if not name.startswith('us.tags.bit') and not name.startswith('raw['):
        print(f'{name:24} {text}')
print('\n-- tags bits / raw words: counts by kind --')
for (g, kind), n in sorted(groups.items(), key=lambda x: str(x)):
    if g in ('us.tags', 'raw'):
        print(f'{g:10} {str(kind):30} {n}')
raw_ids = [E.FLOAT_SLOTS[c] for c in range(E.F) if E.FLOAT_SLOTS[c].startswith('raw[') and cols[c]
           and describe(np.concatenate(cols[c]))[1] not in (None,)]
print('\n-- tokens --')
for t, n in sorted(tok_types.items()):
    print(f'type {t:2} tokens {n:8}')
    for k in range(E.VALUES_W):
        v = np.concatenate(tok_vals[t][k])
        text, kind = describe(v)
        if kind is not None:
            print(f'     value {k:2} {text}')
