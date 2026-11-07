# Inputs v2: a compact object encoding (2,750 -> 283 + tokens)

**2026-11-07: v2 only ADDS inputs.** Every input of the existing pipeline stays exactly as it is; only a proven bug may
change one. In 1535f9d on `feat/inputs-v2-causal-history`:
- **Must undo:** `pending_inputs` replacing `own_flight_*` / `opp_flight_*`. Own plays then show only 6 ticks before
  landing instead of from the tap (~22), the opponent's 6 instead of `OPP_LEADS` (5-19, median 13), `playable()` counts
  in-flight elixir as still there for ~16 ticks after each tap, and live play (Jason's flights) differs from training.
  `own_flight_slots=2` also changes a parameter's shape, so a Jason checkpoint cannot load as is. Keep the archive's flights.
- **Keep only if equal to Jason's converter:** `x_ability` / `x_ability_card`, `hand_float[4:6]` and the objects'
  `ability_button`, filled from v2 (they look like fills for older archives). Compare on one battle with `cr/observation.py`.
（中文：v2 只追加输入，原有输入一律不改。必须撤回：用 v2 6-tick 队列覆盖 own/opp flight。ability / 手牌形态 / ability_button
的填充：和 Jason 转换器在一局上一致就保留，否则用 Jason 的。）

**中文摘要：** v2 每个实体 2750 维不是信息量大，是展开方式造成的：每个“出现过的路径”（含列表下标，如 `us.acts[3]`、`us.hit_by[9][0]`）占一列，再乘 2（值 + 有效位）。实测真实实体非零数中位只有 24–29 个，最多 54。另外两个问题：所有 ID（动作、buff、变量、对象 id）都被 signed-log 当成浮点数（两个不同动作 ID 变成 21.33 和 21.26，几乎一样）；冻结布局在实盘遇到训练没见过的列表长度会直接报错（`V2 field exceeds frozen layout: us.acts[3]`，实测）。本 PR 加入紧凑编码：每实体 8 个 ID（走 embedding）+ 283 维固定槽（所有实体同一含义）+ 变长部分做成 token，并附模型模块和测试；所有输入都能从编码精确还原（4 局完整对局 1110 万个值 0 差异）。接入方法见下面第 4 节。

## 1. Where 2,750 came from

`tools/v2_build_layout.py` gives every path `flatten()` ever yields inside an object its own column, and `NumericLayout`
doubles it (value + validity bit). Paths include list positions, so every running action, every hit-by record, every effect
and every raw word at every position it ever reached in the dataset is a separate column, and so is every list's
`.length`. On the 24 frames in `tests/data` alone the layout is already 712 wide.

Real objects carry far less. Measured on four full pro battles (one from today's reader): an object has a **median of 24-29
non-zero numbers, at most 54**. The rest of the 2,750 columns are zeros for that object.

`x_v2_entity` is also stored as `[objects, 2, width]` (two group members), so a single object costs 2 x 2,750 floats.

## 2. Two problems besides size (both measured on this branch's code)

1. **Ids are encoded as floats.** `NumericLayout.encode` stores `sign * log1p(|v|)` for every number, ids included. Action
   ids 1838885781 and 1703061916 (two different card actions) become 21.33 and 21.26; buff, variable, context and object ids
   the same. A model cannot tell them apart; ids need an embedding.
2. **A live frame can crash the encoder.** The layout is frozen to the paths seen in training, and `encode` raises on any
   other: a unit with one more running action than any in training gives
   `ValueError: V2 field exceeds frozen layout: us.acts[3]` (reproduced with the layout of the test frames). Live play meets
   list lengths training never had (more effects, more hit-by records), so the bot would stop mid-battle.

## 3. The compact encoding (`cr/inputs_v2/v2_compact.py`)

The same encoder as `clapha-inputs/code/crx/encode_compact.py`. Per frame:

| array | shape | contents |
|---|---|---|
| `obj_int` | [N, 8] | side+1, kind, behaviour state, card, data (vocabulary indices), and target / shooter / homing target as indices of objects in the same frame (-1 none) |
| `obj_float` | [N, 283] | one fixed block with the same meaning in every object (`FLOAT_SLOTS` names each column): position, heading, hp, level, timers, states, attack phase, the 64 game tags as bits, facing / aim / shield / health words, and 80 raw words as exact 16-bit halves whose meaning the object's kind gives |
| `tok_int` | [T, 4] | the variable-length parts as tokens: owner object, type (1 variable, 2 running action, 3 scheduled action, 4 action-list element, 5 effect, 6 link, 7 unlinking, 8 hit by, 9 context value, 10-16 id lists), key (vocabulary index), referenced object |
| `tok_float` | [T, 20] | the token's values |

- **Keys come from the game's own tables** (`compact_vocab-160635001.json`: every unit / building / projectile / area,
  card, action, effect, variable and context key of build 160635001), not from the keys some battles happened to have: a
  live frame cannot bring an unknown one, and a longer list is just more tokens.
- **Static card data is not copied into objects.** It belongs to the card / data id: an embedding, or the card table
  compressed once per id.
- **Raw words** hold flag bits (a King tower word is `1 << 24 | 1`, which float32 rounds), so each is stored as two exact
  16-bit halves.
- **Scaling:** `v2_compact.scaled(enc)` divides by `compact_scales-160635001.json` (p99 of |value| per column, from six
  pro battles) and clips to +-8.

**Proof that nothing is lost:** `v2_compact.check(paths)` rebuilds every `entities[]` input of `input_spec.py` from the
encoding and compares it with the frame, field by field (scalars, groups, ids through the vocabularies, references through
the object index, raw words, every list record in its token). On four full pro battles: **11.1 million values, 0 differ**.
A single value changed by 1 in the encoding, in the floats or in a token, fails it.

Size at the busiest frame measured: about 28 KB against about 319 KB at 2,750 floats an object (x2 again for
`[objects, 2, width]`). Encoding takes about 0.6 ms a frame in Python.

## 4. Wiring it in (what this PR does not change yet)

This PR adds the encoder, the model piece and tests; it does not rewire the dataset and training scripts, so nothing
existing changes behaviour. To switch:

1. **Dataset** (`v2_dataset.encode_episode`): next to the existing object inputs (in place of `entity_layout.encode(entity)`, the 2,750-wide v2 layout), call
   `v2_compact.scaled(v2_compact.encode_frame({'entities': canonical_entities}))` per frame and store `obj_int`, `obj_float`
   per object (same order as the other object arrays) and the frame's tokens with their owner indices. Targets: raw reader
   frames carry an address, which `encode_frame` resolves through the objects' `address`; frames where targets are object ids
   work as they are.
2. **Model** (`policy.py`, where `self.v2_entity(f['v2_entity'])` is added): `CompactEntityEncoder(D)` and
   `objects = objects + compact(obj_int, obj_float, tok_int, tok_float)`. `compact_model.pad_batch` shows the padded shapes.
   Every id goes through an embedding; each token is added into its owner object; a reference brings in the referenced
   object's card / data embedding.
3. **Compression groups** (`compression.py`): a merged group keeps its first member's ids and the mean of the members'
   `obj_float`, as for `obj_float` today; the members' tokens all go to the group's index.
4. **Check your own data:** `python -m cr.inputs_v2.v2_compact FRAMES.jsonl.gz` prints PASS when every input of those frames
   is in the encoding.
5. Re-make the training samples. **Keep every existing input** (the v1 47 object features, hand, cycle, elixir, own /
   opponent plays in flight, abilities, legal masks) exactly as `cr/observation.py` makes them: the compact block is added
   to each object, it replaces nothing. Values it shares with v1 (position, hp) are simply there twice.

## 5. Tests (`tests/test_v2_compact.py`, 24 frames of a converted public pro battle in `tests/data`)

- every `entities[]` input of `input_spec.py` has a place;
- every input rebuilt from the encoding (PASS);
- a value changed by 1 in the encoding is caught;
- a frame with 40 hit-by records and 30 variables (more than any training data) encodes;
- no id ends up in a float column;
- the model piece runs forward and backward with gradients in the embeddings and the projections.

The rest of the suite: 218 passed; the five `test_history_continuation` failures happen without this PR too (a temporary
directory path in that test, on this machine).
