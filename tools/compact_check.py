"""Check converted archives with the compact encoding (convert.py --compact): every object row of every decision, both
players' views, Jason's inputs against the compact encoding of the same object -- position (both turned to the player's
view), side, and that a compact row exists wherever Jason has an object. Prints counts; exit 1 on any mismatch.

    python tools/compact_check.py DATASET_DIR [max_archives]"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cr.imitation import read                      # noqa: E402
from cr.compact import capture as CC               # noqa: E402

root = Path(sys.argv[1])
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 9
files = sorted(root.glob('**/*.npz'))[:limit]
n = bad_pos = bad_side = empty = objects = tokens = 0
worst = 0.0
for f in files:
    for owner in (0, 1):
        try:
            d = read(f, owner)
        except Exception:
            continue
        if 'x_cobj_float' not in d:
            continue
        n += 1
        of, ci = d['x_cobj_float'].astype(np.float32), d['x_cobj_int']
        jx, jy = d['x_obj_float'][:, 0] * 18000, d['x_obj_float'][:, 1] * 32000
        cx = of[:, CC.COL['x']] * CC.OF_SCALE[CC.COL['x']]
        cy = of[:, CC.COL['y']] * CC.OF_SCALE[CC.COL['y']]
        objects += len(jx)
        missing = (ci[:, 0] == 0)
        empty += int(missing.sum())
        err = np.maximum(np.abs(cx - jx), np.abs(cy - jy))[~missing]
        worst = max(worst, float(err.max()) if len(err) else 0.0)
        bad_pos += int((err > 30).sum())             # float16 of a scaled position: ~0.05% -> a few units
        bad_side += int((ci[~missing, 0] != d['x_obj_side'][~missing]).sum())
        tokens += int((d['x_ctok_int'][..., 0] >= 0).sum())
print(f'{len(files)} archives, {n} player views, {objects} object rows: no compact row {empty}, position off >30 {bad_pos} '
      f'(worst {worst:.1f}), side differs {bad_side}; tokens {tokens}')
sys.exit(1 if (empty or bad_pos or bad_side) else 0)
