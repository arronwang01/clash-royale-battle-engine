"""The compact object encoding captured during conversion (2026-11-06): at each recorded decision the reader (the same C reader
the console runs live, built into the engine process) reads the frame, and encode_compact turns its objects into ids, one
fixed block of numbers and tokens. Stored sparse (only non-zero values: ~25 an object) in decision['compact'], and turned
into per-owner arrays aligned with observation.py's objects by features() at archive time.

Only for build 160635001 (the reader's layout). Enabled by convert.py --compact.
"""
import ctypes
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from cr.compact import encode_compact as E

HERE = Path(__file__).resolve().parent
LIBG_SHIFT = 0xC8DC0            # the x86_64 engine's class addresses vs the phone build's (vtmap-160635001.json)


def build_reader(out_dir):
    """the reader as a library for this process: Linux reads /proc/self/mem, the Mac its own task memory"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if sys.platform == 'darwin':
        lib = out_dir / 'sampler_x86.dylib'
        cmd = ['clang', '-arch', 'x86_64', '-O1', '-w', '-I', str(HERE), '-include', str(HERE / 'shim_mac.h'), '-dynamiclib',
               '-o', str(lib), str(HERE / 'live_sampler_tbi.c')]
    else:
        lib = out_dir / 'sampler.so'
        cmd = ['cc', '-O1', '-w', '-fPIC', '-shared', '-I', str(HERE), '-include', str(HERE / 'shim_linux.h'),
               '-o', str(lib), str(HERE / 'live_sampler_tbi.c')]
    if not lib.exists() or lib.stat().st_mtime < (HERE / 'live_sampler_tbi.c').stat().st_mtime:
        subprocess.run(cmd, check=True)
    return lib


class Capture:
    def __init__(self, env, work_dir):
        if int(env.B.build) != 160635001:
            raise ValueError(f'the reader reads build 160635001, not {env.B.build}')
        os.environ['CLAPHA_LIBG_BASE'] = hex(env.g.addr(0))
        os.environ['CLAPHA_LIBG_SHIFT'] = hex(LIBG_SHIFT)
        self.lib = ctypes.CDLL(str(build_reader(work_dir)))
        self.lib.sampler_main.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        args = [b'x', str(os.getpid()).encode(), b'50', hex(env.B.glob('manager_global')).encode(),
                hex(env.B.field('manager.state')).encode(), b'--unified', b'1']
        self.argv = (ctypes.c_char_p * len(args))(*args)
        self.argc = len(args)
        self.tmp = os.path.join(tempfile.mkdtemp(prefix='compact-'), 'frame.json')
        self.frames = 0
        self.retries = 0

    def frame(self):
        fd = os.open(self.tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        saved = os.dup(1)
        os.dup2(fd, 1)
        try:
            self.lib.sampler_main(self.argc, self.argv)
        finally:
            os.dup2(saved, 1)
            os.close(fd)
            os.close(saved)
        with open(self.tmp) as f:
            line = next((line for line in f if line.startswith('{')), None)
        return json.loads(line) if line else None

    def record(self, expected_tick=None):
        """this decision's sparse compact record, or None when the reader saw no battle"""
        frame = self.frame()
        for _ in range(2):
            # a new battle at the previous one's address: the reader's kept player cache is stale for one read
            # ('players_unresolved'), which also resets it -- read again (2026-11-06: the first decision of 2 in 18 battles)
            if frame and frame.get('battle_active'):
                break
            self.retries += 1
            frame = self.frame()
        if not frame or not frame.get('battle_active'):
            return None
        if expected_tick is not None and frame.get('game_tick') != expected_tick:
            raise RuntimeError(f"reader tick {frame.get('game_tick')} != engine tick {expected_tick}")
        self.frames += 1
        return sparse(E.encode_frame(frame))


def sparse(enc):
    """only the non-zero numbers: per object (column, value) pairs; tokens as they are (they are few)"""
    of = enc['obj_float']
    rows, cols = np.nonzero(of)
    return {'ids': [int(x) for x in enc['object_ids']], 'int': enc['obj_int'].astype(np.int32).tolist(),
            'f_row': rows.astype(np.int16).tolist(), 'f_col': cols.astype(np.int16).tolist(),
            'f_val': of[rows, cols].astype(np.float64).tolist(),
            'tok_int': enc['tok_int'].astype(np.int32).tolist(), 'tok_float': enc['tok_float'].astype(np.float64).tolist()}


# ---------------------------------------------------------------- per owner, aligned with observation.py's object slots
# columns that are board coordinates (x, y pairs) or vectors, turned with the observer as observation.rotate turns positions
PAIRS = [('x', 'y'), ('tx', 'ty'), ('us.surface[0]', 'us.surface[1]'), ('us.aim[0]', 'us.aim[1]'),
         ('us.start[0]', 'us.start[1]'), ('us.waypoint[0]', 'us.waypoint[1]')]
VECTORS = [('us.dx', 'us.dy'), ('us.facing[0]', 'us.facing[1]')]
COL = {name: i for i, name in enumerate(E.FLOAT_SLOTS)}
OF_SCALE = np.asarray([E.SCALES['obj_float'][n] for n in E.FLOAT_SLOTS], np.float32) if E.SCALES else np.ones(E.F, np.float32)
TF_SCALE = np.asarray(E.SCALES['tok_float'], np.float32) if E.SCALES else np.ones(E.VALUES_W, np.float32)


def owner_arrays(record, object_ids, owner, rotate, max_tokens):
    """one decision's compact record -> arrays for `owner`, row i = observation.py's object slot i (object_ids[i]):
    c_int [slots, 8] (side 1 = owner's, 2 = the other's, as obj_side), c_float [slots, F] scaled, tokens [max_tokens, 4] +
    [max_tokens, V] with the owner and the referenced object as slot indices (-1 none). Returns also how many tokens did not
    fit."""
    slots = len(object_ids)
    c_int = np.zeros((slots, len(E.INT_SLOTS)), np.int32)
    c_float = np.zeros((slots, E.F), np.float32)
    t_int = np.full((max_tokens, 4), -1, np.int32)
    t_float = np.zeros((max_tokens, E.VALUES_W), np.float32)
    if record is None:
        return c_int, c_float, t_int, t_float, 0
    where = {oid: i for i, oid in enumerate(record['ids'])}
    slot_of = {oid: s for s, oid in enumerate(object_ids)}
    src_to_slot = {where[oid]: s for oid, s in slot_of.items() if oid in where}
    dense = np.zeros((len(record['ids']), E.F), np.float32)
    if record['f_row']:
        dense[record['f_row'], record['f_col']] = record['f_val']
    ints = np.asarray(record['int'], np.int32).reshape(-1, len(E.INT_SLOTS))
    for src, s in src_to_slot.items():
        row = dense[src].copy()
        for a, b in PAIRS:
            if row[COL[a]] or row[COL[b]]:
                row[COL[a]], row[COL[b]] = rotate(row[COL[a]], row[COL[b]], owner)
        if owner == 1:
            for a, b in VECTORS:
                row[COL[a]], row[COL[b]] = -row[COL[a]], -row[COL[b]]
        c_float[s] = row / OF_SCALE
        i = ints[src].copy()
        i[0] = 1 if i[0] - 1 == owner else 2                           # side + 1 -> 1 owner's / 2 the other's
        for k in (5, 6, 7):                                           # references -> slots
            i[k] = src_to_slot.get(int(i[k]), -1) if i[k] >= 0 else -1
        c_int[s] = i
    n = 0
    dropped = 0
    for ti, tf in zip(record['tok_int'], record['tok_float']):
        s = src_to_slot.get(ti[0])
        if s is None:
            continue
        if n >= max_tokens:
            dropped += 1
            continue
        t_int[n] = [s, ti[1], ti[2], src_to_slot.get(ti[3], -1) if ti[3] >= 0 else -1]
        t_float[n] = np.asarray(tf, np.float32) / TF_SCALE
        n += 1
    return c_int, np.clip(c_float, -8, 8), t_int, np.clip(t_float, -8, 8), dropped
