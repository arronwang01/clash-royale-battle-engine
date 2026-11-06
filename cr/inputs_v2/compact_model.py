"""The model side of the compact encoding (v2_compact.py): one D-wide vector per object, to add to the object token where
policy.py adds `self.v2_entity(f['v2_entity'])` today.

    enc = CompactEntityEncoder(D)
    objects = objects + enc(obj_int, obj_float, tok_int, tok_float)

Shapes (padded batches; B = batch x time):
    obj_int   [B, O, 8]    int64 (v2_compact.INT_SLOTS)
    obj_float [B, O, 283]  float32, already divided by v2_compact.SCALES (v2_compact.scaled)
    tok_int   [B, K, 4]    int64: owner object index (-1 = padding), token type, key, referenced object (-1 none)
    tok_float [B, K, 20]   float32, scaled
Every id goes through an embedding (never a float: ids have no order), every token is added into its owner object, and an
object's reference to another object (target, shooter, homing target, a token's referenced object) brings in that object's
card / data embedding, so "attacking the Hog Rider" is seen without a separate attention pass.
"""
import torch
from torch import nn

from cr.inputs_v2 import v2_compact as V

TOKEN_TYPES = 17
KEY_TABLES = ('action', 'buff', 'variable', 'context')
# token type -> the vocabulary its key comes from (v2_compact.TOKENS)
TYPE_TABLE = {1: 'variable', 2: 'action', 3: 'action', 4: 'action', 5: 'buff', 9: 'context'}


class CompactEntityEncoder(nn.Module):
    def __init__(self, width, embed=64):
        super().__init__()
        sizes = {name: len(V.VOCAB[name]) + 2 for name in ('card', 'data', *KEY_TABLES)}
        self.side = nn.Embedding(4, embed, padding_idx=0)
        self.kind = nn.Embedding(32, embed)
        self.state = nn.Embedding(32, embed)
        self.card = nn.Embedding(sizes['card'], embed, padding_idx=0)
        self.data = nn.Embedding(sizes['data'], embed, padding_idx=0)
        # one key table, each vocabulary at its own offset
        self.offsets = {}
        total = 0
        for name in KEY_TABLES:
            self.offsets[name] = total
            total += sizes[name]
        offset = torch.zeros(TOKEN_TYPES, dtype=torch.long)
        for ttype, name in TYPE_TABLE.items():
            offset[ttype] = self.offsets[name]
        self.register_buffer('key_offset', offset, persistent=False)
        self.register_buffer('key_used', torch.tensor([t in TYPE_TABLE for t in range(TOKEN_TYPES)]), persistent=False)
        self.key = nn.Embedding(total, embed)
        self.ttype = nn.Embedding(TOKEN_TYPES, embed)
        self.floats = nn.Linear(V.F, width)
        self.ids = nn.Linear(5 * embed + 3 * embed, width)
        self.token = nn.Sequential(nn.Linear(V.VALUES_W + 3 * embed, width), nn.GELU(), nn.Linear(width, width))

    def forward(self, obj_int, obj_float, tok_int, tok_float):
        B, O, _ = obj_int.shape
        what = self.card(obj_int[..., 3]) + self.data(obj_int[..., 4])          # [B, O, e]: what each object is

        def referenced(index):                                                   # what the referenced object is
            ok = (index >= 0) & (index < O)
            got = torch.gather(what, 1, index.clamp(0, O - 1).unsqueeze(-1).expand(-1, -1, what.shape[-1]))
            return got * ok.unsqueeze(-1)

        ids = torch.cat([self.side(obj_int[..., 0].clamp(0, 3)), self.kind(obj_int[..., 1].clamp(0, 31)),
                         self.state(obj_int[..., 2].clamp(0, 31)), self.card(obj_int[..., 3]), self.data(obj_int[..., 4]),
                         referenced(obj_int[..., 5]), referenced(obj_int[..., 6]), referenced(obj_int[..., 7])], -1)
        out = self.ids(ids) + self.floats(obj_float)
        if tok_int.shape[1]:
            owner, ttype, key, ref = tok_int.unbind(-1)
            ttype = ttype.clamp(0, TOKEN_TYPES - 1)
            key = (key + self.key_offset[ttype]) * self.key_used[ttype]
            t = self.token(torch.cat([tok_float, self.ttype(ttype), self.key(key), referenced(ref)], -1))
            valid = (owner >= 0) & (owner < O)
            t = t * valid.unsqueeze(-1)
            out = out.scatter_add(1, owner.clamp(0, O - 1).unsqueeze(-1).expand(-1, -1, t.shape[-1]), t)
        return out


def pad_batch(encodings):
    """a list of v2_compact encodings (one per frame, scaled) -> padded tensors for CompactEntityEncoder"""
    O = max(1, max(len(e['obj_int']) for e in encodings))
    K = max(1, max(len(e['tok_int']) for e in encodings))
    B = len(encodings)
    oi = torch.zeros(B, O, len(V.INT_SLOTS), dtype=torch.long)
    of = torch.zeros(B, O, V.F)
    ti = torch.full((B, K, 4), -1, dtype=torch.long)
    tf = torch.zeros(B, K, V.VALUES_W)
    for b, e in enumerate(encodings):
        n, k = len(e['obj_int']), len(e['tok_int'])
        oi[b, :n] = torch.as_tensor(e['obj_int'])
        of[b, :n] = torch.as_tensor(e['obj_float'])
        if k:
            ti[b, :k] = torch.as_tensor(e['tok_int'])
            tf[b, :k] = torch.as_tensor(e['tok_float'])
    return oi, of, ti, tf
