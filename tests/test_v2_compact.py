"""The compact object encoding: nothing lost, mistakes caught, live frames never rejected, and the model piece trains."""
import copy
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from cr.inputs_v2 import input_spec, v2_compact as V
from cr.inputs_v2.compact_model import CompactEntityEncoder, pad_batch

FRAMES = Path(__file__).parent / 'data' / 'compact_frames.jsonl.gz'   # 24 frames of a converted pro battle (000YLG22U2GC)


def frames():
    return [json.loads(line) for line in gzip.open(FRAMES, 'rt')]


def test_every_entity_input_has_a_place():
    spec = {k.split('.', 1)[1] for k, v in input_spec.SPEC.items() if k.startswith('entities[].') and v[0] == input_spec.IN}
    assert spec <= set(V.COVERAGE), sorted(spec - set(V.COVERAGE))


def test_every_input_rebuilt_from_the_encoding():
    assert V.check([str(FRAMES)]) == 0


def test_a_changed_value_is_caught(monkeypatch):
    original = V.encode_frame

    def wrong(frame):
        enc = original(frame)
        if len(enc['obj_float']) > 5:
            enc['obj_float'][5, V.FLOAT_SLOTS.index('hp')] += 1
        return enc
    monkeypatch.setattr(V, 'encode_frame', wrong)
    assert V.check([str(FRAMES)]) == 1


def test_a_live_frame_with_more_entries_than_training_is_encoded():
    """the frozen path layout raised 'V2 field exceeds frozen layout' for a list longer than any in training (a 15th running
    action, an 11th hit-by record): a live crash. Here a longer list is just more tokens"""
    frame = copy.deepcopy(max(frames(), key=lambda f: len(f['entities'])))
    unit = next(e for e in frame['entities'] if e.get('us', {}).get('hlt'))
    us = unit['us']
    us['hit_by'] = [[unit['category'], k] for k in range(40)]
    us['vars'] = [[1084569640, k] for k in range(30)]
    enc = V.encode_frame(frame)
    assert len(enc['tok_int']) >= 70


def test_ids_are_never_floats():
    enc = V.encode_frame(frames()[10])
    keys = {int(k) for t in ('action', 'buff', 'variable', 'context') for k in V.VOCAB[t]}
    assert not set(np.abs(enc['obj_float']).astype(np.int64).ravel()) & keys - {0, 1}


def test_model_piece_forward_backward():
    encs = [V.scaled(V.encode_frame(f)) for f in frames()[:6]]
    oi, of, ti, tf = pad_batch(encs)
    model = CompactEntityEncoder(384)
    out = model(oi, of, ti, tf)
    assert out.shape == (6, oi.shape[1], 384) and torch.isfinite(out).all()
    out.square().mean().backward()
    assert model.key.weight.grad.abs().sum() > 0 and model.floats.weight.grad.abs().sum() > 0
