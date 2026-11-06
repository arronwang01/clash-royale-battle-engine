"""Compact per-object encoding for inputs v2 (2026-11-06; the same encoder as clapha-inputs code/crx/encode_compact.py).

docs/INPUTS_V2_COMPACT.md says why and how to use it. In short: one slot per path ever seen (v2_build_layout.py, x2 for the
validity flags) made an object 2,750 numbers wide, when real objects carry a median of 24-27 non-zero numbers (54 at most in
four full pro battles). Here every entities[] input of input_spec.py goes to

  obj_int    [N, 8]    side+1, kind, behaviour state, card, data (vocabulary indices: embed them), and the target /
                       shooter / homing target as indices of other objects in the same frame (-1 none)
  obj_float  [N, 283]  one fixed block, the same meaning in every object (FLOAT_SLOTS names each column); divide by SCALES
  tok_int    [T, 4]    the variable-length parts as tokens: owner object index, token type, key (vocabulary index), the
                       referenced object (-1 none)
  tok_float  [T, 20]   each token's values

check() rebuilds every input from the encoding and compares it with the frame, field by field.
"""
from __future__ import annotations

import gzip
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from cr.inputs_v2 import input_spec as M  # noqa: E402

DATA = HERE
BUILD = '160635001'

# ---------------------------------------------------------------------------------------------------------- vocabularies
# Every key a token can carry is a row of the game's own data (static_attributes, variables, context keys): the vocabulary
# is the whole table, not the keys seen in some battles, so a live battle cannot bring an unknown one. 0 = none, 1 = not in
# the table (counted by check()).


def _vocab(keys):
    return {int(k): i + 2 for i, k in enumerate(sorted(int(k) for k in keys))}


_keys = json.load(open(os.path.join(DATA, f'compact_vocab-{BUILD}.json')))
VOCAB = {name: _vocab(_keys[name]) for name in ('data', 'card', 'action', 'buff', 'variable', 'context')}


def vidx(table: str, key) -> int:
    if key is None or key in (0, -1):
        return 0
    return VOCAB[table].get(int(key) & 0xFFFFFFFF, VOCAB[table].get(int(key), 1))


# ---------------------------------------------------------------------------------------------------------- the layout
# obj_float columns: (input field, width). Scalars width 1; fixed groups their width.
CORE = [('x', 1), ('y', 1), ('tx', 1), ('ty', 1), ('hp', 1), ('max_hp', 1), ('level', 1), ('deploy_remaining', 1),
        ('has_attack', 1), ('has_move', 1), ('atk_stage', 1), ('atk_timeline', 1), ('atk_load', 1), ('charge', 1),
        ('us.hidden', 1), ('us.immune', 1), ('us.airborne', 1), ('us.invisible', 1), ('us.invisible_again_ms', 1),
        ('us.shield', 2), ('us.pushed', 1), ('us.push_ms', 1), ('us.hit_count', 1), ('us.dash_ms', 1), ('us.hop', 1),
        ('us.shot', 3), ('us.special_ms', 1), ('us.spawn_ms', 1), ('us.jump_ms', 1), ('us.surface', 2), ('us.z', 1),
        ('us.daggers', 1), ('us.a10', 2), ('us.waypoint', 2), ('us.facing', 2), ('us.hlt', 6), ('us.bcomp', 2),
        ('us.dx', 1), ('us.dy', 1), ('us.aim', 2), ('us.start', 2), ('us.area_ms', 1), ('us.area_life', 1),
        ('us.prj_d8', 1)]
TAG_BITS = 64                      # us.tags: two 32-bit words -> 64 bits (crx/data/game_tags-160635001.json names them)
# raw words by object kind, sharing one block (a troop never has a projectile's words): troop / princess tower: ent + atk +
# mov; projectile: prj; spell area: area_raw; King tower: king + atk
RAW_GROUPS = {'unit': ['us.ent', 'us.atk', 'us.mov'], 'projectile': ['us.prj'], 'area': ['us.area_raw'],
              'king': ['us.king', 'us.atk']}
RAW_W = 80                         # raw words, each as two exact halves: low 16 bits (0..65535) and the rest (v >> 16):
                                   # raw words hold flag bits (a King tower word 16,777,217 = 1 << 24 | 1, 11-06), which a
                                   # float32 would round
FLOAT_SLOTS = [f'{f}[{i}]' if w > 1 else f for f, w in CORE for i in range(w)] + \
              [f'us.tags.bit{i}' for i in range(TAG_BITS)] + \
              [f'raw[{i}].{h}' for i in range(RAW_W) for h in ('lo16', 'hi')]
F = len(FLOAT_SLOTS)

INT_SLOTS = ['side', 'kind', 'behavior_state_raw', 'card_id', 'data_id', 'target', 'us.source', 'us.aim_object']

# token types: (type id, input field, key vocabulary or None, how the record is laid out)
TOKENS = {
    'us.vars': (1, 'variable'),        # [variable id, value]
    'us.acts': (2, 'action'),          # action id per running action (index i); ahead / astate rows of the same index
    'us.ahead': (2, 'action'),         #   join it: one token per running action
    'us.astate': (2, 'action'),
    'us.sched': (3, 'action'),         # [action id, ticks until it starts]
    'us.alists': (4, 'action'),        # [action id, [[ids ...] per list]] -> one token per element
    'us.buffs': (5, 'buff'),           # [kind, left, elapsed, side, speed, hit speed, applied by]; braw row i joins it
    'us.braw': (5, 'buff'),
    'us.links': (6, None),             # [other unit, kind, flags, timer]
    'us.unlinking': (7, None),         # [other unit, kind]
    'us.hit_by': (8, None),            # [object, count]
    'us.ctx': (9, 'context'),          # [key, value]
    'us.area_units': (10, None),       # object ids
    'us.prj_units': (11, None),
    'us.atk_lists': (12, None),        # three object-id lists: list index as a value
    'us.prj_sub': (13, None),          # [4 words], [ids], [ids]
    'us.mov_lists': (14, None),        # two parallel lists
    'us.hl': (15, None),               # two lists
    'us.buffs2': (16, None),
}
VALUES_W = 20

# where every entities[] input goes (check() proves it)
COVERAGE = {f: 'obj_float' for f, _ in CORE} | {'us.tags': 'obj_float (bits)'} | \
           {f: 'obj_float (raw block)' for g in RAW_GROUPS.values() for f in g} | \
           {f: 'obj_int' for f in INT_SLOTS} | {f: 'tokens' for f in TOKENS} | {'category': 'the object index itself'}


def _get(e: dict, field: str):
    return (e.get('us') or {}).get(field[3:]) if field.startswith('us.') else e.get(field)


def _raw_group(e: dict) -> str | None:
    us = e.get('us') or {}
    if 'king' in us:
        return 'king'
    if 'prj' in us:
        return 'projectile'
    if 'area_raw' in us:
        return 'area'
    if 'ent' in us or 'atk' in us or 'mov' in us:
        return 'unit'
    return None


def encode_frame(frame: dict) -> dict:
    ents = frame.get('entities') or []
    # references: object ids (category) in most fields, the object's address in `target` (the reader writes the attack
    # component's target pointer) -- both resolve to the object's index in this frame
    index = {e.get('category'): i for i, e in enumerate(ents)}
    index.update({e.get('address'): i for i, e in enumerate(ents) if e.get('address')})   # raw reader frames only
    ref = lambda oid: index.get(oid, -1) if oid not in (None, 0, -1, '0x0') else -1      # noqa: E731
    n = len(ents)
    obj_int = np.full((n, len(INT_SLOTS)), 0, dtype=np.int64)
    obj_float = np.zeros((n, F), dtype=np.float32)
    toks_i, toks_f = [], []

    def token(owner, ttype, key, refi, values):
        v = [float(x) for x in values][:VALUES_W]
        if len(values) > VALUES_W:
            raise ValueError(f'token type {ttype} with {len(values)} values > VALUES_W')
        toks_i.append((owner, ttype, key, refi))
        toks_f.append(v + [0.0] * (VALUES_W - len(v)))

    for i, e in enumerate(ents):
        obj_int[i] = [int(e.get('side') or 0) + 1, int(e.get('kind') or 0), int(e.get('behavior_state_raw') or 0),
                      vidx('card', e.get('card_id')), vidx('data', e.get('data_id')), ref(e.get('target')),
                      ref(_get(e, 'us.source')), ref(_get(e, 'us.aim_object'))]
        col = 0
        for f, w in CORE:
            v = _get(e, f)
            vals = v if isinstance(v, list) else [v]
            for k in range(w):
                x = vals[k] if k < len(vals) and vals[k] is not None else 0
                obj_float[i, col + k] = float(x)
            col += w
        tags = _get(e, 'us.tags') or [0, 0]
        word = (int(tags[0]) & 0xFFFFFFFF) | ((int(tags[1]) & 0xFFFFFFFF) << 32) if len(tags) >= 2 else 0
        obj_float[i, col:col + TAG_BITS] = [(word >> b) & 1 for b in range(TAG_BITS)]
        col += TAG_BITS
        group = _raw_group(e)
        raw = [x for f in RAW_GROUPS.get(group, []) for x in (_get(e, f) or [])]
        if len(raw) > RAW_W:
            raise ValueError(f'{len(raw)} raw words > RAW_W')
        for k, v in enumerate(raw):
            v = int(v)
            obj_float[i, col + 2 * k] = v & 0xFFFF
            obj_float[i, col + 2 * k + 1] = v >> 16
        # tokens
        us = e.get('us') or {}
        for vid, val in us.get('vars') or []:
            token(i, 1, vidx('variable', vid), -1, [val])
        acts, ahead, astate = us.get('acts') or [], us.get('ahead') or [], us.get('astate') or []
        for k, aid in enumerate(acts):
            words = (ahead[k][1:] if k < len(ahead) else []) + (astate[k][1] if k < len(astate) else [])
            token(i, 2, vidx('action', aid), -1, [k] + words)
        for aid, ticks in us.get('sched') or []:
            token(i, 3, vidx('action', aid), -1, [ticks])
        for aid, lists in us.get('alists') or []:
            for li, xs in enumerate(lists):
                for x in xs:
                    token(i, 4, vidx('action', aid), ref(x), [li, x])
        braw = us.get('braw') or []
        for k, b in enumerate(us.get('buffs') or []):
            token(i, 5, vidx('buff', b[0]), ref(b[6]) if len(b) > 6 else -1, [k] + b[1:] + (braw[k] if k < len(braw) else []))
        for other, *rest in us.get('links') or []:
            token(i, 6, 0, ref(other), [other] + rest)
        for other, *rest in us.get('unlinking') or []:
            token(i, 7, 0, ref(other), [other] + rest)
        for obj, cnt in us.get('hit_by') or []:
            token(i, 8, 0, ref(obj), [obj, cnt])
        for key, val in us.get('ctx') or []:
            token(i, 9, vidx('context', key), -1, [val])
        for x in us.get('area_units') or []:
            token(i, 10, 0, ref(x), [x])
        for x in us.get('prj_units') or []:
            token(i, 11, 0, ref(x), [x])
        for li, xs in enumerate(us.get('atk_lists') or []):
            for x in xs:
                token(i, 12, 0, ref(x), [li, x])
        for li, xs in enumerate(us.get('prj_sub') or []):
            if li == 0:
                token(i, 13, 0, -1, [li] + list(xs))
            else:
                for x in xs:
                    token(i, 13, 0, ref(x), [li, x])
        for name, ttype in (('mov_lists', 14), ('hl', 15)):
            for li, xs in enumerate(us.get(name) or []):
                for k, x in enumerate(xs):
                    token(i, ttype, 0, -1, [li, k, x])
        for k, x in enumerate(us.get('buffs2') or []):
            token(i, 16, 0, -1, [k] + (x if isinstance(x, list) else [x]))
    return {'obj_int': obj_int, 'obj_float': obj_float, 'object_addresses': [e.get('address') for e in ents],
            'tok_int': np.array(toks_i, dtype=np.int64).reshape(-1, 4),
            'tok_float': np.array(toks_f, dtype=np.float32).reshape(-1, VALUES_W),
            'object_ids': [e.get('category') for e in ents]}


# ---------------------------------------------------------------------------------------------------------- the proof
def decode_object(enc: dict, i: int, frame_ents: list) -> dict:
    """the inputs of object i rebuilt from the encoding alone (with the object ids, which are the index)"""
    ids, addrs = enc['object_ids'], enc['object_addresses']
    oid = lambda j: ids[j] if j >= 0 else None          # noqa: E731
    oaddr = lambda j: addrs[j] if j >= 0 else None      # noqa: E731
    inv = {t: {v: k for k, v in VOCAB[t].items()} for t in VOCAB}
    out = {}
    row = enc['obj_float'][i]
    col = 0
    for f, w in CORE:
        out[f] = [int(round(x)) for x in row[col:col + w]] if w > 1 else int(round(row[col]))
        col += w
    bits = row[col:col + TAG_BITS]
    word = sum(int(b) << k for k, b in enumerate(bits))
    out['us.tags'] = [word & 0xFFFFFFFF, word >> 32]
    col += TAG_BITS
    out['raw'] = [int(round(row[col + 2 * k + 1])) * 65536 + int(round(row[col + 2 * k])) for k in range(RAW_W)]
    oi = enc['obj_int'][i]
    out.update({'side': int(oi[0]) - 1, 'kind': int(oi[1]), 'behavior_state_raw': int(oi[2]),
                'card_id': inv['card'].get(int(oi[3])), 'data_id': inv['data'].get(int(oi[4])),
                'target': (oaddr if isinstance(frame_ents[i].get('target'), str) else oid)(int(oi[5])),
                'us.source': oid(int(oi[6])), 'us.aim_object': oid(int(oi[7]))})
    toks = [(tuple(map(int, ti)), tf) for ti, tf in zip(enc['tok_int'], enc['tok_float']) if ti[0] == i]
    out['tokens'] = [(t[1], t[2], t[3], [int(round(x)) for x in tf]) for t, tf in toks]
    return out


def _norm_scalar(v):
    if v is None or v is False:
        return 0
    if v is True:
        return 1
    return int(v)


def check(paths: list[str], every: int = 1) -> int:
    """every entities[] input of the frames rebuilt from the encoding and compared with the frame: scalars and fixed groups
    exactly; ids through the vocabularies; references through the object index (a reference to an object not in this frame
    counts as lost); raw words in order; every list record present in the tokens with all its values"""
    spec = {k.split('.', 1)[1] for k, v in M.SPEC.items() if k.startswith('entities[].') and v[0] == M.IN}
    missing = spec - set(COVERAGE)
    if missing:
        print('NOT COVERED:', sorted(missing))
        return 1
    stats = {'frames': 0, 'objects': 0, 'tokens': 0, 'values': 0, 'differ': 0, 'unknown_keys': 0, 'dangling_refs': 0,
             'numbers_per_object': []}
    examples = []
    for path in paths:
        opener = gzip.open if path.endswith('.gz') else open
        for n, line in enumerate(opener(path, 'rt')):
            if not line.startswith('{') or n % every:
                continue
            frame = json.loads(line)
            if not frame.get('entities'):
                continue
            enc = encode_frame(frame)
            stats['frames'] += 1
            stats['tokens'] += len(enc['tok_int'])
            stats['unknown_keys'] += int((enc['tok_int'][:, 2] == 1).sum()) + int((enc['obj_int'][:, 3:5] == 1).sum())
            for i, e in enumerate(frame['entities']):
                stats['objects'] += 1
                nz = int((enc['obj_float'][i] != 0).sum()) + sum(1 for t in enc['tok_int'] if t[0] == i) * 2
                stats['numbers_per_object'].append(nz)
                d = decode_object(enc, i, frame['entities'])
                us = e.get('us') or {}

                def cmp(name, got, want):
                    stats['values'] += 1
                    if got != want:
                        stats['differ'] += 1
                        if len(examples) < 12:
                            examples.append((name, got, want))
                for f, w in CORE:
                    v = _get(e, f)
                    want = ([_norm_scalar(v[k]) if isinstance(v, list) and k < len(v) else 0 for k in range(w)] if w > 1
                            else _norm_scalar(v[0] if isinstance(v, list) else v))
                    cmp(f, d[f], want)
                tags = us.get('tags') or [0, 0]
                cmp('us.tags', d['us.tags'], [int(tags[0]) & 0xFFFFFFFF, int(tags[1]) & 0xFFFFFFFF] if len(tags) >= 2 else [0, 0])
                raw = [x for f in RAW_GROUPS.get(_raw_group(e), []) for x in (_get(e, f) or [])]
                cmp('raw', d['raw'][:len(raw)], raw)
                cmp('side', d['side'], int(e.get('side') or 0))
                for f in ('kind', 'behavior_state_raw'):
                    cmp(f, d[f], int(e.get(f) or 0))
                for f in ('card_id', 'data_id'):
                    v = e.get(f)
                    if v not in (None, 0, -1):
                        cmp(f, d[f], int(v) & 0xFFFFFFFF if d[f] is not None and d[f] > 2 ** 31 else int(v))
                for f in ('target', 'us.source', 'us.aim_object'):
                    v = _get(e, f)
                    if v not in (None, 0, -1, '0x0'):
                        if d[f] is None:
                            stats['dangling_refs'] += 1
                        else:
                            cmp(f, d[f], v)
                # every list record is in a token with all its values, in order
                vals = [t[3] for t in d['tokens']]
                def has(ttype, values):
                    stats['values'] += 1
                    ok = any(t[0] == ttype and t[3][:len(values)] == values for t in d['tokens'])
                    if not ok:
                        stats['differ'] += 1
                        if len(examples) < 12:
                            examples.append((f'token {ttype}', None, values))
                for vid, val in us.get('vars') or []:
                    has(1, [val])
                for k, (aid, ticks) in enumerate(us.get('sched') or []):
                    has(3, [ticks])
                braw = us.get('braw') or []
                for k, b in enumerate(us.get('buffs') or []):
                    has(5, [k] + b[1:] + (braw[k] if k < len(braw) else []))
                for aid, lists in us.get('alists') or []:
                    for li, xs in enumerate(lists):
                        for x in xs:
                            has(4, [li, x])
                for other, *rest in us.get('unlinking') or []:
                    has(7, [other] + rest)
                for li, xs in enumerate(us.get('atk_lists') or []):
                    for x in xs:
                        has(12, [li, x])
                for li, xs in enumerate(us.get('prj_sub') or []):
                    if li == 0:
                        has(13, [0] + list(xs))
                    else:
                        for x in xs:
                            has(13, [li, x])
                for name, ttype in (('mov_lists', 14), ('hl', 15)):
                    for li, xs in enumerate(us.get(name) or []):
                        for k, x in enumerate(xs):
                            has(ttype, [li, k, x])
                for k, x in enumerate(us.get('buffs2') or []):
                    has(16, [k] + (x if isinstance(x, list) else [x]))
                for other, *rest in us.get('links') or []:
                    has(6, [other] + rest)
                for obj, cnt in us.get('hit_by') or []:
                    has(8, [obj, cnt])
                for key, val in us.get('ctx') or []:
                    has(9, [val])
                for x in us.get('area_units') or []:
                    has(10, [x])
                for x in us.get('prj_units') or []:
                    has(11, [x])
                for k, aid in enumerate(us.get('acts') or []):
                    a = us.get('ahead') or []
                    s = us.get('astate') or []
                    has(2, [k] + (a[k][1:] if k < len(a) else []) + (s[k][1] if k < len(s) else []))
    per = sorted(stats.pop('numbers_per_object'))
    q = lambda p: per[int(p * (len(per) - 1))] if per else 0      # noqa: E731
    print(json.dumps({**stats, 'obj_float_width': F, 'obj_int_width': len(INT_SLOTS), 'token_values': VALUES_W,
                      'nonzero_numbers_per_object': {'median': q(.5), 'p99': q(.99), 'max': per[-1] if per else 0},
                      'tokens_per_frame': round(stats['tokens'] / max(1, stats['frames']), 1)}, indent=1))
    for x in examples:
        print('  differ:', x)
    ok = stats['differ'] == 0 and stats['unknown_keys'] == 0      # float32 rounding shows up as a difference
    print('VERDICT', 'PASS: every input rebuilt from the encoding' if ok else 'FAIL')
    return 0 if ok else 1


# ---------------------------------------------------------------------------------------------------------- scaling
SCALES_PATH = os.path.join(DATA, f'compact_scales-{BUILD}.json')


def build_scales(paths: list[str], every: int = 5) -> dict:
    """per obj_float column and per token value position: the 99th percentile of |value| over the frames (1 where always 0);
    a model divides by it. Written to data/compact_scales-BUILD.json"""
    cols, toks = [[] for _ in range(F)], [[] for _ in range(VALUES_W)]
    for path in paths:
        opener = gzip.open if path.endswith('.gz') else open
        for n, line in enumerate(opener(path, 'rt')):
            if not line.startswith('{') or n % every:
                continue
            enc = encode_frame(json.loads(line))
            for c in range(F):
                cols[c].extend(np.abs(enc['obj_float'][:, c]).tolist())
            for c in range(VALUES_W):
                toks[c].extend(np.abs(enc['tok_float'][:, c]).tolist())
    q = lambda xs: float(max(1.0, np.percentile(xs, 99))) if xs else 1.0      # noqa: E731
    out = {'build': BUILD, 'note': 'p99 of |value| per column over the frames listed; divide by it',
           'frames_from': [os.path.basename(p) for p in paths],
           'obj_float': dict(zip(FLOAT_SLOTS, [q(x) for x in cols])), 'tok_float': [q(x) for x in toks]}
    json.dump(out, open(SCALES_PATH, 'w'), indent=0)
    return out


SCALES = json.load(open(SCALES_PATH)) if os.path.exists(SCALES_PATH) else None


if __name__ == '__main__':
    if sys.argv[1:2] == ['--scales']:
        out = build_scales(sys.argv[2:])
        print('scales written to', SCALES_PATH, 'from', len(out['frames_from']), 'files')
        sys.exit(0)
    sys.exit(check(sys.argv[1:]))


def scaled(enc: dict) -> dict:
    """obj_float and tok_float divided by SCALES (p99 of |value| per column, from pro battles), clipped to +-8"""
    of = np.asarray([SCALES['obj_float'][name] for name in FLOAT_SLOTS], np.float32)
    tf = np.asarray(SCALES['tok_float'], np.float32)
    return {**enc, 'obj_float': np.clip(enc['obj_float'] / of, -8, 8), 'tok_float': np.clip(enc['tok_float'] / tf, -8, 8)}
