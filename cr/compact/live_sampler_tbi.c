#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

enum {
  BATTLE_TICK = 0x60,
  /* the battle's RNG seed, chosen by the server: in 160635001 at +0xa0 ("rng" in the frame; +0x9c, RNG.md's seed word of
     an older build, reads 0 -- checked in-process with seed 1234567891, 2026-11-02). With both decks, the account ids and
     every play it rebuilds a recorded battle in the engine (groundwork check 4) */
  BATTLE_SEED = 0x9C, BATTLE_RNG = 0xA0,
  PLAYER_TABLE = 0xE0,
  PLAYER_HAND_VECTOR = 0x210,
  PLAYER_HAND_CAPACITY = 0x218,
  PLAYER_HAND_SIZE = 0x21C,
  PLAYER_CYCLE_VECTOR = 0x220,
  PLAYER_CYCLE_CAPACITY = 0x228,
  PLAYER_CYCLE_SIZE = 0x22C,
  PLAYER_DECK_COUNT = 0x230,
  PLAYER_ELIXIR_RAW = 0x2F8,
  PLAYER_REFILL_MS = 0x208,      /* ms until the next card can be dealt; 1000 at a deal, -50 a tick (engine refill_probe) */
  MAX_DISCOVERY_NODES = 20000,
  MAX_DISCOVERY_DEPTH = 5,
  MAX_PATH_LENGTH = MAX_DISCOVERY_DEPTH + 2,
  MAX_ENTITIES = 2048,
};

typedef struct {
  uint64_t address;
  int32_t elixir_raw;
  int32_t refill_timer;
  /* +0x318 cards played, +0x340 elixir spent and +0x344 elixir wasted at full (x10000), +0x288 / +0x290 / +0x298 the first troop /
     spell / building played (-1 before): crx/census/player_words_probe.py, 2026-11-03 */
  int32_t cards_played, elixir_spent, elixir_wasted, first_cards[3];
  int32_t player_raw[6];   /* player +0x120 / +0x1FC / +0x30 / +0xB8 / +0x268 / +0x280: 15 -> 16 and 0 -> 1 together, once (pro-replay census 2026-11-04; raw until named) */
  int32_t slot_words[8];   /* player +0x2C8: one word per deck slot, 0 -> 1 / 2 as the slot's card is played (raw until named) */
  int32_t next_deck_index;
  int32_t hand[4];
  int32_t hand_size;
  int32_t cycle[8];
  int32_t cycle_size;
  int32_t deck_count;
  int32_t deck_card_ids[8];
  int32_t deck_form_flags[8];
  int32_t deck_level_index[8];   /* slot +0x18: displayed level - the rarity's first level (common 1, rare 3, epic 6,
                                    legendary 9, champion 11), as the match setup's "l"; clapha-engine 2026-10-12 */
  int32_t deck_visible;
  int32_t deal_id;   /* player +0x24: the id the deal seeds its shuffle with (RNG.md: v + this word) */
  int32_t account_hi, account_lo;
  /* the battle's own deck, the one the game deals from (logic +0x88 + 8 * side -> slots vector +0x20; slot: card data +0x10 ->
     id +0x40, level +0x18, form +0x1C): the profile deck above (deck_card_ids) may hold the same cards in another order
     (check 4's first live battle, 2026-11-03: every hand differed from tick 0) */
  int32_t battle_deck_count;
  int32_t battle_deck[8][3];   /* the player's account (logic +0x30 + 8 * side -> +0, +4): whose commands are whose */
  /* Per-deck-slot evolution progress (player+0x2e8 native vector) and the two hero/champion
     ability controllers (player+0x3a0, +0x3a8). Offsets are FirstLight's (Null's 15.535.13
     probe) and were confirmed on this build by src/runtime_probe.c on the device. */
  int32_t evo_count;
  int32_t evo_progress[8];
  int32_t ability_present[2];
  int32_t ability_charges[2], ability_button[2], ability_cooldown[2], ability_configured[2];
  uint32_t ability_character[2];
  int32_t ability_used[2], ability_effect_delay[2];
  /* the units the controller drives (controller +0xA0 vector): a Clone's copy of a champion has none */
  int32_t ability_unit_count[2];
  uint64_t ability_units[2][4];
} PlayerFrame;

typedef struct {
  uint64_t address;
  int depth;
  uint16_t path[MAX_PATH_LENGTH];
} DiscoveryNode;

typedef struct {
  uint64_t battle, player_state, last_discovery_us;
  uint16_t path[MAX_PATH_LENGTH];
  int path_length;
} PlayerCache;

#define MAX_BUFFS 8
#define MAX_VARS 16
#define MAX_ACTS 32
/* the troop object's words the census found changing and no field read (crx/census/entity_words_probe.py, 2026-11-03): read raw
   for troops and princess towers (not the King towers: their object is the player block) until each is named */
/* more changing words no field read, raw until named (census open list, 2026-11-03): the attack component, the movement
   component (its stored path is left out: it follows from position, target and the map; its next node is the waypoint)
   and the projectile object */
static const uint16_t ATK_RAW[] = {0x18, 0x1c, 0x7c, 0x80, 0x90, 0x94, 0xa0, 0xa4, 0xc4, 0xd4, 0x34, 0x38, 0x3c, 0x40, 0x50, 0x68, 0x6c, 0x70, 0xa8};   /* the last 9: words the census counted read inside a span (fresh break test 2026-11-04) */
static const uint16_t MOV_RAW[] = {0x38, 0x21c, 0x1d4, 0x1d8, 0x1dc, 0x1e4, 0x1e8, 0x1f0, 0x1f4, 0x1f8, 0x1fc, 0x204, 0x208, 0x20c, 0x210};   /* the 13 after 0x21c: inside the observe.py span, never output (fresh break test G / H) */
static const uint16_t PRJ_RAW[] = {0x68, 0x6c, 0x90, 0x94, 0x98, 0x9c, 0xa0, 0xa4, 0xa8, 0xb4, 0xc0, 0xd0, 0xe0, 0x128, 0x134, 0x138, 0x13c, 0x140, 0x144, 0x148, 0x14c, 0x160, 0x164, 0x174, 0x178, 0x17c, 0x188, 0x198, 0x1b0, 0x1b4, 0x1b8, 0x1bc, 0x1c4, 0x1c8, 0x1d8, 0x1dc, 0x200, 0xbc};
#define ATK_RAW_N ((int)(sizeof(ATK_RAW) / 2))
#define MOV_RAW_N ((int)(sizeof(MOV_RAW) / 2))
#define PRJ_RAW_N ((int)(sizeof(PRJ_RAW) / 2))
/* the King towers' own changing words (their object is also the player block): facing, attack timers, hit counters, ... */
static const uint16_t KING_RAW[] = {0xbc, 0xf8, 0xfc, 0x114, 0x118, 0x1b4, 0x1d8, 0x310, 0x32c};
#define KING_RAW_N ((int)(sizeof(KING_RAW) / 2))
static const uint16_t ENT_RAW[] = {0x74, 0x90, 0x94, 0x98, 0xa0, 0xa4, 0xa8, 0xd0, 0xf8, 0xfc, 0x110, 0x114, 0x118, 0x124, 0x128, 0x12c, 0x130, 0x148, 0x14c, 0x154, 0x158, 0x160, 0x170, 0x174, 0x178, 0x190, 0x194, 0x1b4, 0x1d8, 0x1e0, 0x1e4, 0x1e8, 0x1ec, 0x1f4, 0x1fc, 0x200, 0x204, 0xb8, 0xbc, 0x1a0, 0x30};
#define ENT_RAW_N ((int)(sizeof(ENT_RAW) / sizeof(ENT_RAW[0])))
/* Byte masks (2026-11-04, check 4 on a live ladder battle, 160635009; as clapha-engine crx/observe.py RAW_MASK): 1-2-byte flags
   whose neighbouring padding bytes hold old memory on a live client (the engine zeroes them); only the flag bytes are output */
static int32_t raw_mask(char kind, uint16_t off, int32_t v) {
  uint32_t m = 0xFFFFFFFFu;
  switch (kind) {
    case 'e': if (off == 0xa4 || off == 0x110 || off == 0x174 || off == 0x1e4 || off == 0x200) m = 0xFF; break;
    case 'a': if (off == 0x40 || off == 0x18) m = 0xFFFF; break;
    case 'k': if (off == 0x32c) m = 0xFF; break;
    case 'm': if (off == 0x21c) m = 0xFF; break;
    case 'p': if (off == 0x9c || off == 0xa4 || off == 0x14c) m = 0xFF; else if (off == 0x188) m = 0xFF00; else if (off == 0x1dc) m = 0xFFFF; break;
    case 'r': if (off == 0xa4 || off == 0xf8 || off == 0x9c) m = 0xFF; break;
  }
  return (int32_t)((uint32_t)v & m);
}
#include "action_fields_160635001.h"   /* per action kind: its state words (clapha-engine crx/census/action_table_export.py) */
#define DUCHESS_BURST_ID 1852043032u   /* DaggerDuchess_OnStartingAction (ActionBurstAttack) */
#define MUSKETEER_SNIPE_ID 1738620055u /* Musketeer_EV1_snipe_targeting (ActionMusketeerSnipe) */
#define MAX_QUEUE 256
#define MAX_LINKS 64
typedef struct {
  uint64_t address;
  int32_t category, kind, side, x, y, card_id, level, hp, max_hp, behavior;
  int32_t target_x, target_y;   /* entity +0x84 / +0x88, as observe.py's targetX / targetY (groundwork check 3 first read them) */
  /* Troop runtime state, per FirstLight's probe layout (phase_runtime_layout.h), with the
     component vtables located on this build by src/comp_probe.c. -1 / 0 = component absent. */
  int32_t deploy_remaining, has_attack, atk_stage, atk_timeline, atk_load, has_move, charge;
  uint64_t target;
  /* The object's own data record (+0x48 -> +0x40, FirstLight's kObjectDataOffset): the unit a
     spawner produced (a Battle Ram's Barbarians, a Tombstone's Skeletons) or the ProjectileData
     / AreaEffectData of a spell in flight -- the id FirstLight's archetype catalog is keyed by. */
  uint32_t data_id;
  /* Unit states (clapha-engine docs/engine/UNIT_STATES.md; the same reads as its Reader.unit_states(), offsets below).
     us = 1 when they were read for this object. */
  int us;
  int32_t hidden, immune, airborne, invisible, shield, shield_max, pushed, push_ms, hit_count, dash_ms, spawn_ms;
  /* movement +0x208 / +0x20c: the next path node it walks to (a vector on some pushes); +0x30 / +0x34: facing, length ~256
     (crx/census/push_vector_probe.py, 2026-11-03) */
  int32_t waypoint_x, waypoint_y, facing_x, facing_y;
  uint64_t tags;   /* game tags: entity +0xB0, bit = row of game_tags.csv (clapha-engine crx/data/game_tags-<build>.json) */
  /* the card sweep's mechanics: height, jump time left, where an underground unit comes up, Royal Ghost's time to invisible
     again; Golden Knight's hop, a returning shot out (Executioner's axe, Fisherman's hook), the Fisherman's countdown */
  int32_t height, jump_ms, surface_x, surface_y, invisible_again, hop, shot_out, shot_elapsed, shot_total, special_ms;
  /* towers: Dagger Duchess's daggers left, -1 otherwise (the burst action's charges: [[tower +0x260] +0] +0x60 in
     160402012, clapha-engine crx/tower_probe.py) */
  int32_t daggers;
  /* the evolved Musketeer's snipe shots left (a10_64: slot-10 action +0x64, ActionMusketeerSnipe AmmoCount), -1 otherwise;
     a10_60 stays -1 (a tower's daggers: `daggers`) */
  int32_t a10_60, a10_64;
  /* card variables (ActionSetVariable: Skeleton King's souls, the evolved Giant's rage / sequence / hits, ...):
     key = Java hashCode(name) & 0x7FFFFFFF, named by clapha-engine crx/data/variables-<build>.json */
  int32_t var_count;
  struct { uint32_t key; int32_t value; } vars[MAX_VARS];
  /* card actions (the action runner, entity +0x28; clapha-engine observe.py _runner, from the game's code 2026-11-03):
     running = list +0x50, count +0x5c, each action's data row +0x18 -> id +0x40; schedule = the table counted at +0x1c,
     data rows +0x10, int32 ticks until start +0x20. Entries past a count are leftovers and never read */
  int32_t act_count, sched_count;
  uint32_t acts[MAX_ACTS];
  uint64_t act_addr[MAX_ACTS];
  struct { uint32_t id; int32_t ticks; } sched[MAX_ACTS];
  int32_t ent_raw_ok, ent_raw[48];
  int32_t braw[32][10], b2_n, b2[16][14], bcomp_ok, bcomp[2];   /* effects: raw words per entry, the second list (+0x50), component +0x38 / +0x3C */
  int32_t area_raw_ok, area_raw[12], area_n, area_units[32];   /* spell area: raw words, unit ids it has caught / hit */
  int32_t hlt_raw_ok, hlt_raw[6], hit_n, hit_by[32][2], hl_n[2], hl[2][16];   /* health component: raw words, who hit it, two lists */
  int32_t atk_list_n[3], atk_list[3][16];   /* attack component: unit-id lists +0x88 / +0x98 / +0xC8 (counts +0x94 / +0xA4 / +0xD4) */
  int32_t mov_list_n[2], mov_list[2][16];   /* movement component: two parallel int32 lists (+0x10 / +0x20, counts +0x1C / +0x2C) */
  int32_t atk_raw_ok, atk_raw[24], mov_raw_ok, mov_raw[16], prj_raw_ok, prj_raw[40];
  int32_t prj_d8;   /* projectile +0xD8 -> +0x14, raw (-1 without the block) */
  int32_t prj_n, prj_units[32], prj_sub_ok, prj_sub[4], prj_sub_n[2], prj_sub_list[2][32];   /* projectile: unit ids hit, its +0x168 sub-object */          /* ENT_RAW words; ent_raw_ok 0 when not read */
  int32_t ctx_count;
  struct { uint32_t key; int32_t value; } ctx[MAX_ACTS];   /* action contexts (observe.py _contexts), sorted */
  int16_t act_cls[MAX_ACTS];               /* index into ACTION_CLASSES, -1: a kind the table has no state for */
  int32_t act_head[MAX_ACTS][2];   /* each running action's +0x8 / +0x10 (a flag 0 -> 1 in most kinds; BurstAttack +0x8 1024 / 0) */
  int32_t alist_pool[96]; uint8_t alist_len[MAX_ACTS][ACTION_LIST_MAX]; int alist_used;   /* actions' int32 lists (ACTION_LISTS) */
  int64_t act_vals[MAX_ACTS][ACTION_FIELD_MAX];   /* raw: int32 words, or an untagged unit address for a target */
  int32_t buff_count;
  struct { uint32_t id; int32_t remaining, elapsed, side, speed, hit; uint64_t source; } buffs[MAX_BUFFS];
  /* projectiles: shooter id, height, last movement, aim point / aim object, launch point; areas: time left, life */
  int32_t source_id, z, prev_x, prev_y, aim_x, aim_y, start_x, start_y, area_ms, area_life;
  uint64_t aim_object;
} EntityFrame;

/* Component vtables as offsets into libg for build 160402012 (libg sha256 aec6cc5e...),
   found by comp_probe on live troops: the attack component is the one whose +0x10 points at
   another live object 96% of the time and whose +0x24 / +0x28 count up / down in ms; the
   movement component carries classic-charge progress at +0x1e0 (-1 = unavailable). */
#define UNTAG(p) ((p) & 0x00FFFFFFFFFFFFFFULL)
#define ATTACK_COMPONENT_VT 0x1a548b0ULL
#define MOVEMENT_COMPONENT_VT 0x1a54a18ULL
#define AREA_VT 0x1a4e9a0ULL   /* the spell-area class: ARM64 160635001 (x86 0x1b17760; clapha-engine crx/census/server_client/vtmap-160635001.json) */
/* an ability command has no card data at +0x38 (other fields sit there): its vtable, ARM64 and x86_64 (clapha-engine
   crx/census/server_client vtmap); a play is 0x1a28748 / 0x1af1508 */
#define PLAY_ABILITY_VT_ARM 0x1a28bc8ULL
#define PLAY_ABILITY_VT_X86 0x1af1988ULL
static uint64_t g_libg;
/* CLAPHA_LIBG_SHIFT: where the built-in libg offsets (component vtables) sit in a rebuilt libg of the same code, e.g.
   0x1000 in Null's Royale 16.635.1 (every class one page on); 0 for the official game */
static uint64_t g_vt_shift;

/* Unit-state offsets: clapha-engine's crx bindings for 160402020 (the same game version, 16.402, on x86_64; the layouts this
   reader already used match it, e.g. hand +0x210 / cycle +0x220, which moved from August). The three checks follow the
   engine's own functions (character vtable +0xF8 hidden, +0x100 burrowed, +0x108 air; character_damage_immune), whose field
   versions equal the function calls on every unit of 24 random battles in clapha-engine (checks_from_fields).
   Confirmed on this build (libg aec6cc5e, 2026-10-12) from its own ARM64 code, dumped read-only from the running game:
   hidden (libg+0xF84D2C), burrowed (+0xF84CC0), air (+0xF81FF4) and the immune checks use exactly these fields and data
   getters, and the status masks read from the live process are bit 24 (hidden, libg+0x1A65800), 44 (air, +0x1A65878) and
   39 (ground, +0x1A65860). The other offsets (effects, shield, push, attack, projectiles) follow the same layout; check
   them on a recorded battle with tools/live_unit_states_verify.py. */
enum {
  E_STATUS = 0xB0, E_FOLD_TIMER = 0x138, E_IMMUNE_REF = 0x180, E_IMMUNE_TIMER = 0x1A4, E_AIR_A = 0x74, E_AIR_B = 0x8C,
  E_STATE = 0x11C, E_SPAWN_TIMER = 0x144, E_HEIGHT = 0x8C, E_INVISIBLE_AGAIN = 0x1AC,
  A_HOP = 0x78, A_SHOT = 0x58, A_SHOT_ELAPSED = 0x60, A_SHOT_TOTAL = 0x64, A_SPECIAL = 0x44,
  M_SURFACE_X = 0x1CC, M_SURFACE_Y = 0x1D0, M_JUMP_LEFT = 0x218,
  E_PRJ_SOURCE = 0xF8, E_PRJ_Z = 0x8C, E_PRJ_PREV_X = 0x84, E_PRJ_PREV_Y = 0x88, E_PRJ_AIM_OBJECT = 0x108,
  E_PRJ_AIM_X = 0x120, E_PRJ_AIM_Y = 0x124, E_PRJ_START_X = 0x12C, E_PRJ_START_Y = 0x130, E_AREA_LEFT = 0x100,
  D_FOLDS_A = 0x399, D_FOLDS_B = 0x39A, D_HIDE_TIME = 0x3E4, D_BURROW_TARGETABLE = 0x5D8, D_DASH_IMMUNITY = 0x598,
  T_ACTIONS = 0x28, T_ACTION_COUNT = 0x30, BURST_SLOT = 10, BURST_CHARGES = 0x60, SNIPE_AMMO = 0x64,
  E_VARS_FIRST = 0xD8, E_VARS_SIZE = 0xE0, VN_KEY = 0x10, VN_VALUE = 0x14,
  D_FLYING = 0x334, D_SPAWN = 0x238, D_AREA_LIFE = 0x148,
  A_HIT_COUNT = 0x4C, A_DASH = 0x48, M_CHARGE = 0x1E0, M_PUSH_LEFT = 0x1EC, M_PUSHED = 0x214,
  H_SHIELD = 0x2C, H_SHIELD_MAX = 0x30,
  B_ENTRIES = 0x18, B_COUNT = 0x24, B_INVISIBLE = 0x34,
  BE_LEFT = 0x08, BE_TOTAL = 0x0C, BE_DATA = 0x18, BE_SOURCE = 0x38, BE_SIDE = 0x48,
  BD_ID = 0x40, BD_HIT = 0xE0, BD_SPEED = 0xE4,
  C_USED = 0xB0, C_EFFECT_DELAY = 0xAC, C_UNITS = 0xA0,
};
#define STATUS_HIDDEN_BIT 24
#define STATUS_AIR_BIT 44
#define STATUS_GROUND_BIT 39
#define ENTITY_READ 0x1B0   /* covers +0x1AC */

/* static game data, read once per data record: the character fields of the three checks, a buff's id and percentages */
typedef struct { uint64_t data; uint8_t folds, keeps_targetable; int32_t hide_time, dash_immunity, flying; uint64_t spawn; int32_t life; } CharData;
typedef struct { uint64_t data; uint32_t id; int32_t hit, speed; } BuffData;
static CharData g_chars[256];
static BuffData g_buffs[256];

static const CharData *char_data(int fd, uint64_t data);
static const BuffData *buff_data(int fd, uint64_t data);

typedef struct {
  uint64_t root, context, battle, player_state, hp_state, registry, collection, data;
  int32_t replay_tick, native_count, entity_count, filtered_count;
  int discovery_nodes, discovery_budget_exhausted;
  const char *failure;
  EntityFrame entities[MAX_ENTITIES];
  /* the newest tick received from the server (battle +0x64) and the command queue (battle +0x38: data +0x08, count +0x14):
     every command of both sides from its arrival until it executes. All a battle's plays are here for 2-21 ticks at least,
     so a recording rebuilds the battle (groundwork check 4). Fields as a replay's "cmd" (checked against every command of a
     pro replay, 2026-11-02): +0x0c t2, +0x10 t, +0x14 idHi, +0x18 idLo; a play +0x28 px, +0x2c py, +0x30 sid, +0x38 card
     data (id at +0x40) = sel.os, +0x48 sel.pd; an ability +0x28 cgid. The type is the entry's vtable as a libg offset */
  int32_t server_tick, queue_count;
  int32_t location, arena;     /* logic +0xB8 / +0xD0 -> row id +0x40: the match setup a recording needs */
  int32_t game_mode;           /* [[logic +0x1B0] +0x18] -> row id +0x40 (Ladder 72000006, Tournament 72000009: card levels differ) */
  struct { uint64_t vt; int32_t t, t2, hi, lo, a, b, sid, card, pd; } queue[MAX_QUEUE];
  /* the game-object manager's unit link table (collection +0xC0; clapha-engine observe.py links()): rows unit, other, kind,
     flags, timer ms (columns +0x38 +0x48 +0x58 +0x98 +0xA8, count +0x44) and removals queued (unit, other, kind: +0x68
     +0x78 +0x88, count +0x74). Evolved Goblin Cage pull, Boss Bandit dash, hero Mega Minion / Magic Archer ability, Rune Giant */
  int32_t link_n, unlink_n, links[MAX_LINKS][5], unlinks[MAX_LINKS][3];
  int32_t clock_ok, clock[20], logic_raw_ok, logic_raw[9];   /* 8 logic words, then the battle context's +0x1F4 */
  int32_t card_count_n[2], card_counts[2][256][2];   /* per player: cards played so far, in first-play order, and how often (profile battle +0xD8 / +0xE0 -> +0xC8) */   /* battle clock (logic +0x190, words +0x10..+0x5C); the logic's own words */
} ChainFrame;

static int read_exact(int fd, uint64_t address, void *output, size_t size) {
  uint8_t *cursor = (uint8_t *)output;
  size_t done = 0;
  while (done < size) {
    /* Strip the Android heap pointer tag (top byte, e.g. 0xb4) before reading. */
    ssize_t value = pread(fd, cursor + done, size - done,
                          (off_t)((address & 0x00FFFFFFFFFFFFFFULL) + done));
    if (value <= 0) return 0;
    done += (size_t)value;
  }
  return 1;
}

static const CharData *char_data(int fd, uint64_t data) {
  CharData *slot = &g_chars[(data >> 4) & 255];
  if (slot->data == data) return slot;
  uint8_t a = 0, b = 0, keep = 0;
  CharData fresh = {.data = data};
  if (!read_exact(fd, data + D_FOLDS_A, &a, 1) || !read_exact(fd, data + D_FOLDS_B, &b, 1) ||
      !read_exact(fd, data + D_HIDE_TIME, &fresh.hide_time, 4) || !read_exact(fd, data + D_BURROW_TARGETABLE, &keep, 1) ||
      !read_exact(fd, data + D_DASH_IMMUNITY, &fresh.dash_immunity, 4) || !read_exact(fd, data + D_FLYING, &fresh.flying, 4))
    return NULL;
  read_exact(fd, data + D_SPAWN, &fresh.spawn, 8);
  fresh.folds = a || b;
  fresh.keeps_targetable = keep != 0;
  *slot = fresh;
  return slot;
}

/* the unit's card actions (EntityFrame.acts / sched): entity +0x28 -> runner; see EntityFrame */
static void read_runner(int fd, const uint8_t *raw, EntityFrame *item) {
  uint64_t runner = 0;
  uint8_t r[0x60];
  memcpy(&runner, raw + T_ACTIONS, 8);
  item->act_count = item->sched_count = 0;
  item->alist_used = 0;
  if (UNTAG(runner) < 0x100000 || !read_exact(fd, UNTAG(runner), r, sizeof(r))) return;
  uint64_t list = 0, rows = 0, ticks = 0;
  int32_t n = 0, m = 0;
  memcpy(&list, r + 0x50, 8); memcpy(&n, r + 0x5C, 4);
  memcpy(&rows, r + 0x10, 8); memcpy(&m, r + 0x1C, 4); memcpy(&ticks, r + 0x20, 8);
  for (int k = 0; k < n && n <= 64 && item->act_count < MAX_ACTS && UNTAG(list) > 0x100000; ++k) {
    uint64_t action = 0, data = 0;
    uint32_t id = 0;
    if (read_exact(fd, UNTAG(list) + 8 * (uint64_t)k, &action, 8) && UNTAG(action) > 0x100000 &&
        read_exact(fd, UNTAG(action) + 0x18, &data, 8) && UNTAG(data) > 0x100000 &&
        read_exact(fd, UNTAG(data) + 0x40, &id, 4)) {
      int k2 = item->act_count, cls = -1, lo = 0, hi = ACTION_BY_ID_COUNT - 1;
      while (lo <= hi) {                   /* the action's kind, by its id (ACTION_BY_ID is sorted) */
        int mid = (lo + hi) / 2;
        if (ACTION_BY_ID[mid].id == id) { cls = ACTION_BY_ID[mid].cls; break; }
        if (ACTION_BY_ID[mid].id < id) lo = mid + 1; else hi = mid - 1;
      }
      item->act_cls[k2] = (int16_t)cls;
      for (int f = 0; cls >= 0 && f < ACTION_CLASSES[cls].count; ++f) {
        uint64_t w = 0;
        int32_t v = 0;
        if (ACTION_CLASSES[cls].kind[f] == 7) {
          item->act_vals[k2][f] = read_exact(fd, UNTAG(action) + ACTION_CLASSES[cls].off[f], &w, 8) ? (int64_t)UNTAG(w) : 0;
        } else {
          item->act_vals[k2][f] = read_exact(fd, UNTAG(action) + ACTION_CLASSES[cls].off[f], &v, 4) ? v : 0;
          if (ACTION_FLAGS[cls].flag[f]) item->act_vals[k2][f] &= 0xFF;   /* a 1-byte flag: its padding holds old memory live */
        }
      }
      { int32_t hw[4] = {0}; read_exact(fd, UNTAG(action) + 0x8, hw, sizeof(hw)); item->act_head[k2][0] = hw[0]; item->act_head[k2][1] = hw[2] & 0xFF; }   /* +0x10: a flag byte (raw_mask) */
      for (int j = 0; j < ACTION_LIST_MAX; ++j) item->alist_len[k2][j] = 0;
      for (int j = 0; cls >= 0 && j < ACTION_LISTS[cls].count; ++j) {   /* its int32 lists: to the count, 16 each, 96 a unit */
        uint8_t hd[16]; uint64_t p = 0; int32_t cnt = 0, m = 0;
        if (read_exact(fd, UNTAG(action) + ACTION_LISTS[cls].off[j], hd, sizeof(hd))) {
          memcpy(&p, hd, 8); memcpy(&cnt, hd + 12, 4);
          m = (UNTAG(p) > 0x100000 && cnt > 0) ? (cnt < 16 ? cnt : 16) : 0;
          if (m > 96 - item->alist_used) m = 96 - item->alist_used;
          if (m > 0 && !read_exact(fd, UNTAG(p), item->alist_pool + item->alist_used, (size_t)m * 4)) m = 0;
        }
        item->alist_len[k2][j] = (uint8_t)m;
        item->alist_used += m;
      }
      item->act_addr[k2] = UNTAG(action);
      item->acts[item->act_count++] = id;
    }
  }
  for (int k = 0; k < m && m <= 64 && item->sched_count < MAX_ACTS && UNTAG(rows) > 0x100000 && UNTAG(ticks) > 0x100000; ++k) {
    uint64_t data = 0;
    uint32_t id = 0;
    int32_t t = 0;
    if (read_exact(fd, UNTAG(rows) + 8 * (uint64_t)k, &data, 8) && UNTAG(data) > 0x100000 &&
        read_exact(fd, UNTAG(data) + 0x40, &id, 4) && read_exact(fd, UNTAG(ticks) + 4 * (uint64_t)k, &t, 4)) {
      item->sched[item->sched_count].id = id;
      item->sched[item->sched_count++].ticks = t;
    }
  }
}

/* the unit's action contexts: action +0x28 -> context; +0x8 -> block; [begin, end) at +0x20 / +0x28 of 16-byte entries
   (uint32 key hash, value pointer -> int32 at +0x4); each context once; sorted like observe.py _contexts */
static void read_contexts(int fd, EntityFrame *item) {
  uint64_t done[MAX_ACTS];
  int nd = 0;
  item->ctx_count = 0;
  for (int k = 0; k < item->act_count; ++k) {
    uint64_t c = 0, blk = 0, be[2] = {0, 0};
    if (!read_exact(fd, item->act_addr[k] + 0x28, &c, 8) || !c || (UNTAG(c) & 7)) continue;
    c = UNTAG(c);
    int dup = 0;
    for (int d = 0; d < nd; ++d) dup |= done[d] == c;
    if (dup) continue;
    done[nd++] = c;
    if (!read_exact(fd, c + 0x8, &blk, 8) || !blk || (UNTAG(blk) & 7) || !read_exact(fd, UNTAG(blk) + 0x20, be, 16)) continue;
    uint64_t beg = UNTAG(be[0]), end = UNTAG(be[1]);
    if (!beg || (beg & 7) || end <= beg || end - beg > 16 * 64 || (end - beg) % 16) continue;
    for (uint64_t e = beg; e < end && item->ctx_count < MAX_ACTS; e += 16) {
      uint32_t key = 0; uint64_t v = 0; int32_t val = 0;
      if (!read_exact(fd, e, &key, 4) || !read_exact(fd, e + 8, &v, 8)) continue;
      if (v && !(UNTAG(v) & 7)) read_exact(fd, UNTAG(v) + 4, &val, 4);
      item->ctx[item->ctx_count].key = key;
      item->ctx[item->ctx_count++].value = val;
    }
  }
  for (int i = 1; i < item->ctx_count; ++i)       /* sort by (key, value) */
    for (int j = i; j > 0 && (item->ctx[j - 1].key > item->ctx[j].key ||
                              (item->ctx[j - 1].key == item->ctx[j].key && item->ctx[j - 1].value > item->ctx[j].value)); --j) {
      __typeof__(item->ctx[0]) t = item->ctx[j]; item->ctx[j] = item->ctx[j - 1]; item->ctx[j - 1] = t;
    }
}

static const BuffData *buff_data(int fd, uint64_t data) {
  BuffData *slot = &g_buffs[(data >> 4) & 255];
  if (slot->data == data) return slot;
  BuffData fresh = {.data = data};
  if (!read_exact(fd, data + BD_ID, &fresh.id, 4) || !read_exact(fd, data + BD_HIT, &fresh.hit, 4) ||
      !read_exact(fd, data + BD_SPEED, &fresh.speed, 4))
    return NULL;
  *slot = fresh;
  return slot;
}

/* hidden / immune / airborne as the engine decides them (see the offsets above) */
static void unit_checks(int fd, const uint8_t *raw, uint64_t data, EntityFrame *item) {
  const CharData *d = data ? char_data(fd, data) : NULL;
  if (!d) return;
  int32_t state = 0, fold = 0, immune_timer = 0, air_a = 0, air_b = 0;
  uint64_t status = 0, immune_ref = 0;
  memcpy(&state, raw + E_STATE, 4);
  memcpy(&status, raw + E_STATUS, 8);
  item->tags = status;
  memcpy(&fold, raw + E_FOLD_TIMER, 4);
  memcpy(&immune_ref, raw + E_IMMUNE_REF, 8);
  memcpy(&immune_timer, raw + E_IMMUNE_TIMER, 4);
  memcpy(&air_a, raw + E_AIR_A, 4);
  memcpy(&air_b, raw + E_AIR_B, 4);
  if (state == 11 || state == 6 || (state == 7 && !d->keeps_targetable)) item->hidden = 1;
  else if (d->folds) item->hidden = fold == d->hide_time;
  else item->hidden = (int)(status >> STATUS_HIDDEN_BIT & 1);
  item->immune = immune_ref != 0 || (state == 3 && d->dash_immunity > 0) || immune_timer > 0;
  int air = (int)(status >> STATUS_AIR_BIT & 1), ground = (int)(status >> STATUS_GROUND_BIT & 1);
  item->airborne = air && ground ? air_a + air_b > 0 : air || (!ground && d->flying > 0);
  if (d->spawn) memcpy(&item->spawn_ms, raw + E_SPAWN_TIMER, 4);
  memcpy(&item->height, raw + E_HEIGHT, 4);
  memcpy(&item->invisible_again, raw + E_INVISIBLE_AGAIN, 4);
  if (item->invisible_again < 0) item->invisible_again = 0;
  item->us = 1;
}

static uint64_t monotonic_us(void) {
  struct timespec value;
  clock_gettime(CLOCK_MONOTONIC, &value);
  return (uint64_t)value.tv_sec * 1000000ULL +
         (uint64_t)value.tv_nsec / 1000ULL;
}

static uint64_t find_libg_base(int pid) {
  char maps_path[64];
  char line[1024];
  uint64_t best = UINT64_MAX;
  snprintf(maps_path, sizeof(maps_path), "/proc/%d/maps", pid);
  FILE *maps = fopen(maps_path, "r");
  if (!maps) return 0;
  while (fgets(line, sizeof(line), maps)) {
    unsigned long long start = 0;
    unsigned long long offset = 0;
    char permissions[8] = {0};
    if (!strstr(line, "/libg.so")) continue;
    if (sscanf(line, "%llx-%*llx %7s %llx", &start, permissions, &offset) !=
        3)
      continue;
    if ((uint64_t)start >= (uint64_t)offset &&
        (uint64_t)start - (uint64_t)offset < best)
      best = (uint64_t)start - (uint64_t)offset;
  }
  fclose(maps);
  return best == UINT64_MAX ? 0 : best;
}

static int valid_index(int32_t value, int32_t deck_count, int allow_empty) {
  if (allow_empty && value == -1) return 1;
  return value >= 0 && value < deck_count;
}

static int read_player(int fd, uint64_t player, PlayerFrame *out) {
  uint64_t hand_vector = 0, cycle_vector = 0;
  int32_t hand_capacity = -1, cycle_capacity = -1;
  memset(out, 0, sizeof(*out));
  out->address = player;
  out->deck_count = 8;
  out->next_deck_index = -1;
  for (int index = 0; index < 4; ++index) out->hand[index] = -1;
  if (!player ||
      !read_exact(fd, player + PLAYER_ELIXIR_RAW, &out->elixir_raw, 4) ||
      !read_exact(fd, player + PLAYER_REFILL_MS, &out->refill_timer, 4) ||
      !read_exact(fd, player + 0x318, &out->cards_played, 4) ||
      !read_exact(fd, player + 0x340, &out->elixir_spent, 4) ||
      !read_exact(fd, player + 0x344, &out->elixir_wasted, 4) ||
      !read_exact(fd, player + 0x288, out->first_cards, 12) ||
      !read_exact(fd, player + 0x2C8, out->slot_words, 32) ||
      !read_exact(fd, player + 0x120, &out->player_raw[0], 4) ||
      !read_exact(fd, player + 0x1FC, &out->player_raw[1], 4) ||
      !read_exact(fd, player + 0x30, &out->player_raw[2], 4) ||
      !read_exact(fd, player + 0xB8, &out->player_raw[3], 4) ||
      !read_exact(fd, player + 0x268, &out->player_raw[4], 4) ||
      !read_exact(fd, player + 0x280, &out->player_raw[5], 4) ||
      !read_exact(fd, player + PLAYER_HAND_VECTOR, &hand_vector, 8) ||
      !read_exact(fd, player + PLAYER_HAND_CAPACITY, &hand_capacity, 4) ||
      !read_exact(fd, player + PLAYER_HAND_SIZE, &out->hand_size, 4) ||
      !read_exact(fd, player + PLAYER_CYCLE_VECTOR, &cycle_vector, 8) ||
      !read_exact(fd, player + PLAYER_CYCLE_CAPACITY, &cycle_capacity, 4) ||
      !read_exact(fd, player + PLAYER_CYCLE_SIZE, &out->cycle_size, 4))
    return 0;
  if (out->elixir_raw < 0 || out->elixir_raw > 100000 ||
      hand_capacity < 4 || hand_capacity > 8 || out->hand_size != 4 ||
      cycle_capacity < 0 || cycle_capacity > 8 ||
      out->cycle_size < 0 || out->cycle_size > cycle_capacity ||
      (out->hand_size && !hand_vector) ||
      (out->cycle_size && !cycle_vector))
    return 0;
  if (out->hand_size &&
      !read_exact(fd, hand_vector, out->hand,
                  (size_t)out->hand_size * sizeof(out->hand[0])))
    return 0;
  if (out->cycle_size &&
      !read_exact(fd, cycle_vector, out->cycle,
                  (size_t)out->cycle_size * sizeof(out->cycle[0])))
    return 0;
  for (int index = 0; index < out->hand_size; ++index)
    if (!valid_index(out->hand[index], out->deck_count, 1)) return 0;
  for (int index = 0; index < out->cycle_size; ++index)
    if (!valid_index(out->cycle[index], out->deck_count, 0)) return 0;
  if (out->cycle_size > 0) out->next_deck_index = out->cycle[0];
  return 1;
}

static int read_visible_deck(int fd, PlayerFrame *player) {
  int visible = player->next_deck_index >= 0 || player->cycle_size > 0;   /* own side: the cycle stays visible while the hand
                                                                              is being dealt (fresh break test I: the deck vanished
                                                                              in 342 of 1600 frames); the live opponent has all hidden */
  for (int index = 0; index < 4; ++index)
    if (player->hand[index] >= 0) visible = 1;
  if (!visible) return 1;
  uint64_t context = 0, root = 0, avatar = 0, owner = 0, entries = 0;
  int32_t side = -1, count = -1, identity = -1;
  int32_t account_hi = 0, account_lo = 0, entry_count = -1;
  if (!read_exact(fd, player->address + 0x10, &context, 8) || !context ||
      !read_exact(fd, context + 0x98, &root, 8) || !root ||
      !read_exact(fd, player->address + 0x78, &side, 4) ||
      side < 0 || side > 5 ||
      !read_exact(fd, root + 0x30 + (uint64_t)side * 8, &avatar, 8) ||
      !avatar || !read_exact(fd, avatar, &account_hi, 4) ||
      !read_exact(fd, avatar + 4, &account_lo, 4) ||
      !read_exact(fd, root + 0x60, &count, 4) || count < 1 || count > 6)
    return 0;
  for (int index = 0; index < count; ++index) {
    uint64_t candidate = 0;
    int32_t hi = 0, lo = 0;
    if (!read_exact(fd, root + 0x30 + (uint64_t)index * 8,
                    &candidate, 8) || !candidate ||
        !read_exact(fd, candidate, &hi, 4) ||
        !read_exact(fd, candidate + 4, &lo, 4))
      continue;
    if (hi == account_hi && lo == account_lo) {
      identity = index;
      break;
    }
  }
  if (identity < 0 ||
      !read_exact(fd, root + 0x88 + (uint64_t)identity * 8, &owner, 8) ||
      !owner || !read_exact(fd, owner + 0x20, &entries, 8) || !entries ||
      !read_exact(fd, owner + 0x2c, &entry_count, 4) || entry_count != 8)
    return 0;
  for (int index = 0; index < 8; ++index) {
    uint64_t entry = 0, data = 0;
    if (!read_exact(fd, entries + (uint64_t)index * 8, &entry, 8) || !entry ||
        !read_exact(fd, entry + 0x10, &data, 8) || !data ||
        !read_exact(fd, data + 0x40, &player->deck_card_ids[index], 4) ||
        !read_exact(fd, entry + 0x1c, &player->deck_form_flags[index], 4))
      return 0;
    int32_t level_index = -1;
    player->deck_level_index[index] = read_exact(fd, entry + 0x18, &level_index, 4) && level_index >= 0 &&
                                      level_index < 16 ? level_index : -1;
    int32_t card_id = player->deck_card_ids[index];
    int32_t form = player->deck_form_flags[index];
    if (card_id < 25000000 || card_id > 29999999 || form < 0 || form > 2)
      return 0;
  }
  player->deck_visible = 1;
  return 1;
}

/* Never fails the frame: a battle with no heroes or a transient read simply reports none. */
static void read_player_runtime(int fd, PlayerFrame *player) {
  player->evo_count = 0;
  for (int i = 0; i < 2; ++i) player->ability_present[i] = 0;
  if (!player->address) return;
  uint64_t data = 0;
  int32_t capacity = 0, count = 0;
  if (read_exact(fd, player->address + 0x2E8, &data, 8) &&
      read_exact(fd, player->address + 0x2F0, &capacity, 4) &&
      read_exact(fd, player->address + 0x2F4, &count, 4) && data && count > 0 && count <= 8 &&
      capacity >= count && capacity <= 64 &&
      read_exact(fd, data, player->evo_progress, (size_t)count * sizeof(int32_t)))
    player->evo_count = count;
  for (int i = 0; i < 2; ++i) {
    uint64_t controller = 0, back = 0, character = 0;
    if (!read_exact(fd, player->address + 0x3A0 + (uint64_t)i * 8, &controller, 8) || !controller)
      continue;
    /* the controller's back-pointer is how the probe identified it; keep it as the check */
    if (!read_exact(fd, controller + 0x20, &back, 8) ||
        (back & 0x00FFFFFFFFFFFFFFULL) != (player->address & 0x00FFFFFFFFFFFFFFULL))
      continue;
    read_exact(fd, controller + C_USED, &player->ability_used[i], 4);
    uint64_t units = 0, unit_data = 0;
    int32_t unit_count = 0;
    player->ability_unit_count[i] = 0;
    if (read_exact(fd, controller + C_UNITS, &units, 8) && units && read_exact(fd, UNTAG(units), &unit_data, 8) &&
        read_exact(fd, UNTAG(units) + 12, &unit_count, 4) && unit_count > 0 && unit_count <= 64 && unit_data) {
      if (unit_count > 4) unit_count = 4;
      if (read_exact(fd, UNTAG(unit_data), player->ability_units[i], (size_t)unit_count * 8)) {
        for (int k = 0; k < unit_count; ++k) player->ability_units[i][k] = UNTAG(player->ability_units[i][k]);
        player->ability_unit_count[i] = unit_count;
      }
    }
    read_exact(fd, controller + C_EFFECT_DELAY, &player->ability_effect_delay[i], 4);
    if (!read_exact(fd, controller + 0x78, &player->ability_cooldown[i], 4) ||
        !read_exact(fd, controller + 0x7C, &player->ability_configured[i], 4) ||
        !read_exact(fd, controller + 0x80, &player->ability_charges[i], 4) ||
        !read_exact(fd, controller + 0x98, &player->ability_button[i], 4))
      continue;
    player->ability_character[i] = 0;
    if (read_exact(fd, controller + 0x90, &character, 8) && character)
      read_exact(fd, character + 0x40, &player->ability_character[i], 4);
    player->ability_present[i] = 1;
  }
}

static int read_player_pair(int fd, uint64_t player_state,
                            PlayerFrame players[2]) {
  uint64_t addresses[2] = {0, 0};
  return player_state &&
         read_exact(fd, player_state + PLAYER_TABLE, addresses,
                    sizeof(addresses)) &&
         read_player(fd, addresses[0], &players[0]) &&
         read_player(fd, addresses[1], &players[1]) &&
         read_visible_deck(fd, &players[0]) &&
         read_visible_deck(fd, &players[1]) &&
         (read_player_runtime(fd, &players[0]), read_player_runtime(fd, &players[1]), 1);
}

static int seen_insert(uint64_t *table, size_t capacity, uint64_t value) {
  size_t index = (size_t)((value >> 3) * 11400714819323198485ull) &
                 (capacity - 1);
  for (size_t attempt = 0; attempt < capacity; ++attempt) {
    if (table[index] == value) return 0;
    if (table[index] == 0) {
      table[index] = value;
      return 1;
    }
    index = (index + 1) & (capacity - 1);
  }
  return 0;
}

static uint64_t discover_player_state(int fd, uint64_t battle,
                                      PlayerFrame players[2], PlayerCache *cache,
                                      ChainFrame *frame) {
  enum { SEEN_CAPACITY = 32768 };
  DiscoveryNode *nodes = calloc(MAX_DISCOVERY_NODES, sizeof(*nodes));
  uint64_t *seen = calloc(SEEN_CAPACITY, sizeof(*seen));
  if (!nodes || !seen) {
    free(nodes);
    free(seen);
    return 0;
  }
  int count = 1;
  uint64_t started = monotonic_us();
  nodes[0].address = battle;
  seen_insert(seen, SEEN_CAPACITY, battle);
  for (int cursor = 0; cursor < count; ++cursor) {
    if ((cursor & 63) == 0 && monotonic_us() - started > 20000) {
      frame->discovery_budget_exhausted = 1;
      break;
    }
    frame->discovery_nodes = cursor + 1;
    DiscoveryNode node = nodes[cursor];
    if (read_player_pair(fd, node.address, players)) {
      cache->path_length = node.depth;
      memcpy(cache->path, node.path, sizeof(cache->path));
      free(nodes);
      free(seen);
      return node.address;
    }
    uint64_t context = 0, indirect_state = 0;
    if (read_exact(fd, node.address + 0x10, &context, 8) && context &&
        read_exact(fd, context + 0x98, &indirect_state, 8) &&
        read_player_pair(fd, indirect_state, players)) {
      cache->path_length = node.depth + 2;
      memcpy(cache->path, node.path, sizeof(cache->path));
      cache->path[node.depth] = 0x10;
      cache->path[node.depth + 1] = 0x98;
      free(nodes);
      free(seen);
      return indirect_state;
    }
    if (node.depth >= MAX_DISCOVERY_DEPTH) continue;
    uint8_t raw[0x600];
    if (!read_exact(fd, node.address, raw, sizeof(raw))) continue;
    for (size_t offset = 0;
         offset + sizeof(uint64_t) <= sizeof(raw) &&
         count < MAX_DISCOVERY_NODES;
         offset += sizeof(uint64_t)) {
      uint64_t child = 0, probe = 0;
      memcpy(&child, raw + offset, sizeof(child));
      if (child < 0x10000ull || (child & 7) ||
          !seen_insert(seen, SEEN_CAPACITY, child) ||
          !read_exact(fd, child, &probe, sizeof(probe)))
        continue;
      nodes[count].address = child;
      nodes[count].depth = node.depth + 1;
      memcpy(nodes[count].path, node.path, sizeof(node.path));
      nodes[count].path[node.depth] = (uint16_t)offset;
      ++count;
    }
  }
  free(nodes);
  free(seen);
  return 0;
}

static int read_entities(int fd, ChainFrame *frame) {
  uint64_t addresses[MAX_ENTITIES];
  if (!read_exact(fd, frame->battle + 0xA8, &frame->hp_state, 8) || !frame->hp_state ||
      !read_exact(fd, frame->hp_state + 0x08, &frame->registry, 8) || !frame->registry ||
      !read_exact(fd, frame->registry + 0x40, &frame->collection, 8) || !frame->collection ||
      !read_exact(fd, frame->collection + 0x08, &frame->data, 8) ||
      !read_exact(fd, frame->collection + 0x14, &frame->native_count, 4) ||
      frame->native_count < 0 || frame->native_count > MAX_ENTITIES ||
      (frame->native_count && (!frame->data || !read_exact(fd, frame->data, addresses,
          (size_t)frame->native_count * sizeof(addresses[0]))))) return 0;
  frame->link_n = frame->unlink_n = 0;
  {
    uint64_t table = 0, col[8];
    int32_t n = 0, q = 0;
    if (read_exact(fd, frame->collection + 0xC0, &table, 8) && table && read_exact(fd, table + 0x44, &n, 4) &&
        read_exact(fd, table + 0x74, &q, 4)) {
      int ok = 1;
      for (int k = 0; k < 8; ++k) ok = ok && read_exact(fd, table + 0x38 + 16 * k, &col[k], 8);
      static const int ROW[5] = {0, 1, 2, 6, 7}, QUEUE[3] = {3, 4, 5};
      if (ok && n > 0 && n <= MAX_LINKS) {
        int good = 1;
        for (int c = 0; c < 5 && good; ++c)
          for (int i = 0; i < n && good; ++i) good = read_exact(fd, col[ROW[c]] + 4 * (uint64_t)i, &frame->links[i][c], 4);
        if (good) frame->link_n = n;
      }
      if (ok && q > 0 && q <= MAX_LINKS) {
        int good = 1;
        for (int c = 0; c < 3 && good; ++c)
          for (int i = 0; i < q && good; ++i) good = read_exact(fd, col[QUEUE[c]] + 4 * (uint64_t)i, &frame->unlinks[i][c], 4);
        if (good) frame->unlink_n = q;
      }
    }
  }
  for (int i = 0; i < frame->native_count; ++i) {
    uint8_t raw[ENTITY_READ];
    /* the unit-state fields reach +0x1A8; smaller objects (projectiles, areas) may end before that, so fall back to the
       original 0x160 there */
    int wide = addresses[i] && read_exact(fd, addresses[i], raw, sizeof(raw));
    if (!wide && (!addresses[i] || !read_exact(fd, addresses[i], raw, 0x160))) return 0;
    EntityFrame item = {.address = addresses[i], .hp = -1, .max_hp = -1,
                        .atk_stage = -1, .atk_timeline = -1, .atk_load = -1, .charge = -1,
                        .hit_count = -1, .dash_ms = -1, .spawn_ms = -1, .area_ms = -1, .area_life = -1,
                        .surface_x = -1, .surface_y = -1};
    memcpy(&item.category, raw + 0x08, 4);
    memcpy(&item.kind, raw + 0x30, 4);
    memcpy(&item.side, raw + 0x78, 4);
    memcpy(&item.x, raw + 0x7C, 4);
    memcpy(&item.y, raw + 0x80, 4);
    memcpy(&item.target_x, raw + 0x84, 4);
    memcpy(&item.target_y, raw + 0x88, 4);
    memcpy(&item.card_id, raw + 0xAC, 4);
    memcpy(&item.level, raw + 0x120, 4);
    memcpy(&item.behavior, raw + 0x11C, 4);
    uint64_t data_record = 0;
    memcpy(&data_record, raw + 0x48, 8);
    if (data_record) read_exact(fd, data_record + 0x40, &item.data_id, 4);
    /* What is kept (2026-10-12, checked offline by clapha-engine crx/filter_probe.py on 40 random battles over every
       card, evolution and hero, against crx's unit_states): projectiles (kind 0, object id 4xxxxxx), spell / damage
       areas (kind 0, 3xxxxxx) and timed objects (5xxxxxx outside the unit kinds: death bombs, Bomb Tower's bomb, Rage
       bottle, Graveyard's container), whatever their data id: evolutions, heroes and newer cards carry hashed data ids
       (not 10xxxxxx / 22xxxxxx), and before this the filter dropped them (Mighty Miner's bomb, Graveyard, Goblin Curse,
       Earthquake, Vines, Goblinstein, hero projectiles, ...). Projectiles may start or end a little outside the arena
       (Mega Knight's jump starts at y -2500, the Log ends past y 32000). Everything else keeps the unit filter. */
    int is_projectile = item.kind == 0 && item.category >= 4000000 && item.category < 5000000;
    int is_area = item.kind == 0 && item.category >= 3000000 && item.category < 4000000;
    int effect = is_projectile || is_area;
    int timed = item.category >= 5000000 && item.category < 6000000 && (item.kind < 10 || item.kind > 20);
    int unit = !effect && !timed;
    if (unit && (item.category < 5000000 || item.category >= 6000000 ||
        item.kind < 10 || item.kind > 20) ) {
      frame->filtered_count++;
      continue;
    }
    /* wide for effects: an Elite Archer arrow flies on past the arena (x -5604 was dropped at 5000, groundwork check 3) */
    int margin = unit ? 0 : 40000;
    if (item.side < 0 || item.side > 1 ||
        item.x < -margin || item.x > 18000 + margin || item.y < -margin || item.y > 32000 + margin ||
        /* (2026-11-06) a unit's level can pass 16: the hero Mini P.E.K.K.A read 17 for the last 17 decisions of a converted
           ladder battle (09YPPQG0J2QV) and was dropped from the frame while crx's reader had it; 32 still rejects garbage */
        (unit && (item.level < 0 || item.level > 32)) ||
        (item.card_id != -1 && !(!unit && item.card_id == 0) &&
         (item.card_id < 20000000 || item.card_id >= 1000000000) &&
         /* evolution unit forms carry 13xxxxxx, below the card-id range */
         (item.card_id < 13000000 || item.card_id >= 14000000))) {
      frame->filtered_count++;
      continue;
    }
    if (effect) {
      /* a projectile or area effect: no level, hitpoints, deploy timer, state or components of a unit (+0x11C and +0x120
         may lie past their end: pointer bits, found by groundwork check 4) */
      item.level = 0;
      item.behavior = 0;
      item.deploy_remaining = 0;
      if (is_projectile && wide) {                    /* a projectile */
        memcpy(&item.source_id, raw + E_PRJ_SOURCE, 4);
        memcpy(&item.z, raw + E_PRJ_Z, 4);
        memcpy(&item.prev_x, raw + E_PRJ_PREV_X, 4);
        memcpy(&item.prev_y, raw + E_PRJ_PREV_Y, 4);
        memcpy(&item.aim_object, raw + E_PRJ_AIM_OBJECT, 8);
        item.aim_object = UNTAG(item.aim_object);
        memcpy(&item.aim_x, raw + E_PRJ_AIM_X, 4);
        memcpy(&item.aim_y, raw + E_PRJ_AIM_Y, 4);
        memcpy(&item.start_x, raw + E_PRJ_START_X, 4);
        memcpy(&item.start_y, raw + E_PRJ_START_Y, 4);
        item.us = 1;
        read_runner(fd, raw, &item); read_contexts(fd, &item);   /* its card actions (a rage trail, ...) */
        { uint8_t pb[0x1a8];
          if (read_exact(fd, item.address + 0x60, pb, sizeof(pb))) {
            for (int k = 0; k < PRJ_RAW_N; ++k) { memcpy(&item.prj_raw[k], pb + (PRJ_RAW[k] - 0x60), 4); item.prj_raw[k] = raw_mask('p', PRJ_RAW[k], item.prj_raw[k]); }
            item.prj_raw_ok = 1;
            uint64_t p = 0, sub = 0; int32_t n = 0;
            memcpy(&p, pb, 8); memcpy(&n, pb + 0xC, 4);              /* +0x60 unit ids hit, count +0x6C (observe.py prjUnits) */
            if (p && n > 0 && n <= 32 && read_exact(fd, p, item.prj_units, (size_t)n * 4)) item.prj_n = n;
            { uint64_t d8 = 0; memcpy(&d8, pb + (0xD8 - 0x60), 8); item.prj_d8 = -1;
              if (d8) read_exact(fd, d8 + 0x14, &item.prj_d8, 4); }
            memcpy(&sub, pb + (0x168 - 0x60), 8);                       /* +0x168 sub-object (observe.py prjSub) */
            uint8_t sb[0x30];
            if (sub && read_exact(fd, sub, sb, sizeof(sb))) {
              for (int k = 0; k < 4; ++k) memcpy(&item.prj_sub[k], sb + 4 * k, 4);
              item.prj_sub[3] &= 0xFFFFFF;   /* flag bytes; the top byte is padding (live: old memory, 0xb4) */
              item.prj_sub_ok = 1;
              for (int j = 0; j < 2; ++j) {
                uint64_t q = 0; int32_t m = 0;
                memcpy(&q, sb + 0x10 + 16 * j, 8); memcpy(&m, sb + 0x1C + 16 * j, 4);
                if (q && m > 0 && m <= 32 && read_exact(fd, q, item.prj_sub_list[j], (size_t)m * 4)) item.prj_sub_n[j] = m;
              }
            }
          }
        }
      } else if (is_area) {                           /* a spell / damage area */
        memcpy(&item.area_ms, raw + E_AREA_LEFT, 4);
        if (data_record) read_exact(fd, UNTAG(data_record) + D_AREA_LIFE, &item.area_life, 4);
        item.us = 1;
        read_runner(fd, raw, &item); read_contexts(fd, &item);   /* its card actions (Graveyard's skeleton schedule, ...) */
        { /* the spell-area class only (clapha-engine observe.py AREA_RAW / areaUnits) */
          uint64_t vt = 0, p = 0; int32_t n = 0;
          static const uint16_t AREA[12] = {0x94, 0x98, 0x9c, 0xa0, 0xa4, 0xa8, 0xf8, 0xfc, 0x114, 0x138, 0xb0, 0xb8};   /* +0x138: 1500 -> 0 (pro-replay census 2026-11-04) */
          memcpy(&vt, raw, 8);
          if (UNTAG(vt) == g_libg + g_vt_shift + AREA_VT) {
            for (int k = 0; k < 12; ++k) { memcpy(&item.area_raw[k], raw + AREA[k], 4); item.area_raw[k] = raw_mask('r', AREA[k], item.area_raw[k]); }
            item.area_raw_ok = 1;
            memcpy(&p, raw + 0x60, 8); memcpy(&n, raw + 0x6C, 4);
            if (p && n > 0 && n <= 32 && read_exact(fd, p, item.area_units, (size_t)n * 4)) item.area_n = n;
          }
        }
      }
      frame->entities[frame->entity_count++] = item;
      continue;
    }
    item.level++;
    memcpy(&item.deploy_remaining, raw + 0x15C, 4);
    item.daggers = item.a10_60 = item.a10_64 = -1;
    if (wide) { read_runner(fd, raw, &item); read_contexts(fd, &item); }
    if (wide && (item.category == 5000000 || item.category == 5000003)) {
      uint8_t kb[0x2c0];
      if (read_exact(fd, item.address + 0x70, kb, sizeof(kb))) {
        for (int k = 0; k < KING_RAW_N; ++k) { memcpy(&item.ent_raw[k], kb + (KING_RAW[k] - 0x70), 4); item.ent_raw[k] = raw_mask('k', KING_RAW[k], item.ent_raw[k]); }
        item.ent_raw_ok = 2;
      }
    }
    if (wide && item.category >= 5000000 && item.category != 5000000 && item.category != 5000003) {
      uint8_t ext[0x198];                    /* +0x70 .. +0x208 */
      if (read_exact(fd, item.address + 0x70, ext, sizeof(ext))) {
        for (int k = 0; k < ENT_RAW_N; ++k)   /* words below +0x70 (+0x30: the level as shown, fresh break test) are read on their own */
          { if (ENT_RAW[k] >= 0x70) memcpy(&item.ent_raw[k], ext + (ENT_RAW[k] - 0x70), 4); else read_exact(fd, item.address + ENT_RAW[k], &item.ent_raw[k], 4); item.ent_raw[k] = raw_mask('e', ENT_RAW[k], item.ent_raw[k]); }
        item.ent_raw_ok = 1;
      }
    }
    for (int k = 0; k < item.act_count; ++k)
      if (item.acts[k] == MUSKETEER_SNIPE_ID) {
        int32_t ammo = -1;
        if (read_exact(fd, item.act_addr[k] + SNIPE_AMMO, &ammo, 4) && ammo >= 0 && ammo <= 3) item.a10_64 = ammo;
      }
    if (wide) {
      /* card variables: the entity's unordered_map (+0xC8; first node +0xD8, size +0xE0; node next +0x0, key +0x10,
         value +0x14), clapha-engine observe.py _variables (found by the field census, 2026-11-01) */
      int32_t size = 0;
      uint64_t node = 0;
      memcpy(&size, raw + E_VARS_SIZE, 4);
      memcpy(&node, raw + E_VARS_FIRST, 8);
      for (int k = 0; k < size && k < MAX_VARS && UNTAG(node) > 0x100000; ++k) {
        uint8_t n[0x18];
        if (!read_exact(fd, UNTAG(node), n, sizeof(n))) break;
        memcpy(&item.vars[k].key, n + VN_KEY, 4);
        memcpy(&item.vars[k].value, n + VN_VALUE, 4);
        item.var_count = k + 1;
        memcpy(&node, n, 8);
      }
    }
    if (item.category >= 5000000 && item.category < 5000006 && item.kind == 13) {   /* a princess tower */
      /* Dagger Duchess's daggers: the burst action, found among the tower's running actions by its data id (until 2026-11-03
         the first running action was taken, right only while it stayed first), charges at +0x60 */
      if (!wide) read_runner(fd, raw, &item);
      for (int k = 0; k < item.act_count; ++k)
        if (item.acts[k] == DUCHESS_BURST_ID) {
          int32_t charges = -1;
          if (read_exact(fd, item.act_addr[k] + BURST_CHARGES, &charges, 4) && charges >= 0 && charges <= 8) item.daggers = charges;
        }
    }
    if (wide && data_record) unit_checks(fd, raw, UNTAG(data_record), &item);
    uint64_t components = 0, hp = 0;
    memcpy(&components, raw + 0x18, 8);
    int32_t component_count = 0;
    memcpy(&component_count, raw + 0x24, 4);
    if (components && g_libg && component_count > 0 && component_count <= 16) {
      uint64_t list[16];
      if (read_exact(fd, components, list, (size_t)component_count * 8)) {
        for (int c = 0; c < component_count; ++c) {
          uint8_t comp[0x80];
          if (!list[c] || !read_exact(fd, list[c], comp, sizeof(comp))) continue;
          uint64_t vt = 0;
          memcpy(&vt, comp, 8);
          vt = UNTAG(vt);
          if (vt == g_libg + g_vt_shift + ATTACK_COMPONENT_VT) {
            { uint8_t ab[0xE0];
              if (read_exact(fd, list[c], ab, sizeof(ab))) {
                for (int k = 0; k < ATK_RAW_N; ++k) { memcpy(&item.atk_raw[k], ab + ATK_RAW[k], 4); item.atk_raw[k] = raw_mask('a', ATK_RAW[k], item.atk_raw[k]); }
                item.atk_raw_ok = 1;
                static const uint16_t AL[3] = {0x88, 0x98, 0xC8};   /* observe.py atkLists */
                for (int j = 0; j < 3; ++j) {
                  uint64_t q = 0; int32_t m = 0;
                  memcpy(&q, ab + AL[j], 8); memcpy(&m, ab + AL[j] + 12, 4);
                  if (m > 16) m = 16;
                  if (q && m > 0 && read_exact(fd, q, item.atk_list[j], (size_t)m * 4)) item.atk_list_n[j] = m;
                }
              } }
            item.has_attack = 1;
            memcpy(&item.target, comp + 0x10, 8);
            item.target = UNTAG(item.target);
            memcpy(&item.atk_stage, comp + 0x20, 4);
            memcpy(&item.atk_timeline, comp + 0x24, 4);
            memcpy(&item.atk_load, comp + 0x28, 4);
            memcpy(&item.hit_count, comp + A_HIT_COUNT, 4);
            memcpy(&item.dash_ms, comp + A_DASH, 4);
            uint64_t shot = 0;
            memcpy(&item.hop, comp + A_HOP, 4);
            memcpy(&shot, comp + A_SHOT, 8);
            memcpy(&item.shot_elapsed, comp + A_SHOT_ELAPSED, 4);
            memcpy(&item.shot_total, comp + A_SHOT_TOTAL, 4);
            memcpy(&item.special_ms, comp + A_SPECIAL, 4);
            item.shot_out = shot != 0 && item.shot_total > 0;
            if (item.special_ms < 0) item.special_ms = 0;
          } else if (vt == g_libg + g_vt_shift + MOVEMENT_COMPONENT_VT) {
            for (int k = 0; k < MOV_RAW_N; ++k) if (read_exact(fd, list[c] + MOV_RAW[k], &item.mov_raw[k], 4)) { item.mov_raw[k] = raw_mask('m', MOV_RAW[k], item.mov_raw[k]); item.mov_raw_ok = 1; }
            for (int j = 0; j < 2; ++j) {   /* observe.py movLists */
              uint64_t q = 0; int32_t m = 0;
              memcpy(&q, comp + 0x10 + 16 * j, 8); memcpy(&m, comp + 0x1C + 16 * j, 4);
              if (m > 16) m = 16;
              if (q && m > 0 && read_exact(fd, q, item.mov_list[j], (size_t)m * 4)) item.mov_list_n[j] = m;
            }
            item.has_move = 1;
            uint8_t move[0x50];
            int32_t surface[2] = {-1, -1};
            if (read_exact(fd, list[c] + M_SURFACE_X, surface, sizeof(surface))) {
              item.surface_x = surface[0];
              item.surface_y = surface[1];
            }
            if (read_exact(fd, list[c] + M_CHARGE, move, sizeof(move))) {
              memcpy(&item.jump_ms, move + (M_JUMP_LEFT - M_CHARGE), 4);
              if (item.jump_ms < 0) item.jump_ms = 0;
              memcpy(&item.charge, move, 4);
              memcpy(&item.push_ms, move + (M_PUSH_LEFT - M_CHARGE), 4);   /* +0x1EC: the push SPEED (world units per tick, -25 a tick), not time; key push_ms kept */
              memcpy(&item.pushed, move + (M_PUSHED - M_CHARGE), 4);
              if (item.push_ms < 0) item.push_ms = 0;
              memcpy(&item.waypoint_x, move + (0x208 - M_CHARGE), 4);
              memcpy(&item.waypoint_y, move + (0x20C - M_CHARGE), 4);
            }
            int32_t facing[2] = {0, 0};
            if (read_exact(fd, list[c] + 0x30, facing, sizeof(facing))) {
              item.facing_x = facing[0];
              item.facing_y = facing[1];
            }
          }
        }
      }
    }
    if (components && read_exact(fd, components + 0x10, &hp, 8) && hp) {
      int32_t pair[2];
      /* hp may exceed max: the evolved P.E.K.K.A's heal overheals (2392 / 2350, 2026-11-02, groundwork check 3); requiring
         hp <= max read such a unit as having no hitpoints, so the console took it for a timed object and dropped its whole
         unit state while it stayed overhealed */
      if (read_exact(fd, hp + 0x10, pair, sizeof(pair)) && pair[0] >= 0 && pair[1] > 0 &&
          pair[1] <= 10000000 && pair[0] <= 4 * pair[1]) {   /* 1,209,998: the Tombstone champion's dummies */
        item.hp = pair[0]; item.max_hp = pair[1];
      }
      int32_t guard[2];
      if (read_exact(fd, hp + H_SHIELD, guard, sizeof(guard))) { item.shield = guard[0]; item.shield_max = guard[1]; }
      { /* more health component words (clapha-engine observe.py HLT_RAW / _health_lists) */
        static const uint16_t HLT[6] = {0x18, 0x20, 0x24, 0x28, 0x34, 0x38};
        uint8_t hb[0x78];
        if (read_exact(fd, hp, hb, sizeof(hb))) {
          for (int k = 0; k < 6; ++k) memcpy(&item.hlt_raw[k], hb + HLT[k], 4);
          item.hlt_raw_ok = 1;
          uint64_t p = 0; int32_t n = 0;
          memcpy(&p, hb + 0x48, 8); memcpy(&n, hb + 0x54, 4);
          if (p && n > 0 && n <= 32) {
            uint64_t recs[32];
            if (read_exact(fd, p, recs, (size_t)n * 8)) {
              for (int i = 0; i < n; ++i) {
                item.hit_by[i][0] = item.hit_by[i][1] = -1;
                if (recs[i]) read_exact(fd, recs[i], item.hit_by[i], 8);
              }
              item.hit_n = n;
            }
          }
          for (int j = 0; j < 2; ++j) {
            uint64_t q = 0; int32_t m = 0;
            memcpy(&q, hb + 0x58 + 16 * j, 8); memcpy(&m, hb + 0x64 + 16 * j, 4);
            if (q && m > 0 && m <= 16 && read_exact(fd, q, item.hl[j], (size_t)m * 4)) item.hl_n[j] = m;
          }
        }
      }
    }
    /* the effect list: component 3, as in clapha-engine (Reader.buffs) */
    uint64_t buffs = 0;
    if (item.us && components && component_count > 3 && read_exact(fd, components + 0x18, &buffs, 8) && buffs) {
      uint8_t head[0x60];
      uint64_t entries = 0;
      int32_t count = 0, invisible = 0;
      if (read_exact(fd, UNTAG(buffs), head, sizeof(head))) {
        memcpy(&entries, head + B_ENTRIES, 8);
        memcpy(&count, head + B_COUNT, 4);
        memcpy(&invisible, head + B_INVISIBLE, 4);
        item.invisible = invisible > 0;
        static const uint16_t BRAW[10] = {0x10, 0x28, 0x50, 0x58, 0x60, 0x64, 0x68, 0x70, 0x74, 0x78};
        memcpy(&item.bcomp[0], head + 0x38, 4); memcpy(&item.bcomp[1], head + 0x3C, 4); item.bcomp_ok = 1;
        { /* the second effect list (+0x50, count +0x5C): [globalId, remaining, total, side, raw...] (observe.py _buff_extra) */
          uint64_t e2 = 0, l2[16]; int32_t n2 = 0;
          memcpy(&e2, head + 0x50, 8); memcpy(&n2, head + 0x5C, 4);
          if (n2 > 16) n2 = 16;
          if (e2 && n2 > 0 && read_exact(fd, UNTAG(e2), l2, (size_t)n2 * 8))
            for (int k = 0; k < n2; ++k) {
              uint8_t en[0x80]; uint64_t as = 0;
              if (!l2[k] || !read_exact(fd, UNTAG(l2[k]), en, sizeof(en))) continue;
              memcpy(&as, en + BE_DATA, 8);
              const BuffData *bd2 = as ? buff_data(fd, UNTAG(as)) : NULL;
              if (!bd2) continue;
              int32_t *row = item.b2[item.b2_n++];
              row[0] = (int32_t)bd2->id; memcpy(&row[1], en + BE_LEFT, 4); memcpy(&row[2], en + BE_TOTAL, 4); memcpy(&row[3], en + BE_SIDE, 4);
              for (int j = 0; j < 10; ++j) memcpy(&row[4 + j], en + BRAW[j], 4);
            }
        }
        uint64_t list[32];
        if (count > 32) count = 32;
        if (entries && count > 0 && read_exact(fd, UNTAG(entries), list, (size_t)count * 8)) {
          for (int k = 0; k < count && item.buff_count < MAX_BUFFS; ++k) {
            uint8_t entry[0x80];
            uint64_t asset = 0;
            if (!list[k] || !read_exact(fd, UNTAG(list[k]), entry, sizeof(entry))) continue;
            memcpy(&asset, entry + BE_DATA, 8);
            const BuffData *bd = asset ? buff_data(fd, UNTAG(asset)) : NULL;
            if (!bd) continue;
            int32_t left = 0, total = 0;
            memcpy(&left, entry + BE_LEFT, 4);
            memcpy(&total, entry + BE_TOTAL, 4);
            item.buffs[item.buff_count].id = bd->id;
            item.buffs[item.buff_count].remaining = left;
            item.buffs[item.buff_count].elapsed = total - left;
            memcpy(&item.buffs[item.buff_count].side, entry + BE_SIDE, 4);
            memcpy(&item.buffs[item.buff_count].source, entry + BE_SOURCE, 8);
            item.buffs[item.buff_count].source = UNTAG(item.buffs[item.buff_count].source);
            item.buffs[item.buff_count].speed = bd->speed;
            item.buffs[item.buff_count].hit = bd->hit;
            for (int j = 0; j < 10; ++j) memcpy(&item.braw[item.buff_count][j], entry + BRAW[j], 4);
            item.buff_count++;
          }
        }
      }
    }
    frame->entities[frame->entity_count++] = item;
  }
  return 1;
}

/* an object's id (category) from its address, for buff sources and homing targets; -1 when it is not a listed object */
static int32_t object_id(const ChainFrame *frame, uint64_t address) {
  for (int i = 0; address && i < frame->entity_count; ++i)
    if (UNTAG(frame->entities[i].address) == address) return frame->entities[i].category;
  return -1;
}

/* the card actions of a unit, area or projectile: running ids, decoded state, contexts, schedule */
static void emit_actions(const EntityFrame *e, const ChainFrame *frame) {
  printf("\"acts\":[");
  for (int k = 0; k < e->act_count; ++k) printf("%s%u", k ? "," : "", e->acts[k]);
  printf("],\"astate\":[");
  for (int k = 0; k < e->act_count; ++k) {
    int cls = e->act_cls[k];
    printf("%s[%u,[", k ? "," : "", e->acts[k]);
    for (int f = 0; cls >= 0 && f < ACTION_CLASSES[cls].count; ++f) {
      if (ACTION_CLASSES[cls].kind[f] == 7)
        printf("%s%d", f ? "," : "", e->act_vals[k][f] ? object_id(frame, (uint64_t)e->act_vals[k][f]) : -1);
      else
        printf("%s%lld", f ? "," : "", (long long)e->act_vals[k][f]);
    }
    printf("]]");
  }
  printf("],\"ctx\":[");
  for (int k = 0; k < e->ctx_count; ++k) printf("%s[%u,%d]", k ? "," : "", e->ctx[k].key, e->ctx[k].value);
  printf("],\"sched\":[");
  for (int k = 0; k < e->sched_count; ++k) printf("%s[%u,%d]", k ? "," : "", e->sched[k].id, e->sched[k].ticks);
  printf("],\"ahead\":[");
  for (int k = 0; k < e->act_count; ++k) printf("%s[%u,%d,%d]", k ? "," : "", e->acts[k], e->act_head[k][0], e->act_head[k][1]);
  printf("],\"alists\":[");
  for (int k = 0, at = 0, m = 0; k < e->act_count; ++k) {
    int cls = e->act_cls[k];
    if (cls < 0 || !ACTION_LISTS[cls].count) continue;
    printf("%s[%u,[", m++ ? "," : "", e->acts[k]);
    for (int j = 0; j < ACTION_LISTS[cls].count; ++j) {
      printf("%s[", j ? "," : "");
      for (int i = 0; i < e->alist_len[k][j]; ++i) printf("%s%d", i ? "," : "", e->alist_pool[at + i]);
      at += e->alist_len[k][j];
      printf("]");
    }
    printf("]]");
  }
  printf("]");
}

/* "us": the unit states (clapha-engine Reader.unit_states names); only what the object has */
static void emit_unit_states(const EntityFrame *e, const ChainFrame *frame) {
  if (e->kind == 0 && e->category >= 4000000 && e->category < 5000000) {   /* by object id: data ids may be hashed */
    printf(",\"us\":{\"source\":%d,\"z\":%d,\"dx\":%d,\"dy\":%d,\"aim\":[%d,%d],\"aim_object\":%d,\"start\":[%d,%d],",
           e->source_id, e->z, e->x - e->prev_x, e->y - e->prev_y, e->aim_x, e->aim_y, object_id(frame, e->aim_object),
           e->start_x, e->start_y);
    emit_actions(e, frame);
    printf(",\"prj\":[");
    for (int k = 0; e->prj_raw_ok && k < PRJ_RAW_N; ++k) printf("%s%d", k ? "," : "", e->prj_raw[k]);
    printf("],\"prj_d8\":%d,\"prj_units\":[", e->prj_raw_ok ? e->prj_d8 : -1);
    for (int i = 0; i < e->prj_n; ++i) printf("%s%d", i ? "," : "", e->prj_units[i]);
    printf("],\"prj_sub\":[");
    if (e->prj_sub_ok) {
      printf("[%d,%d,%d,%d]", e->prj_sub[0], e->prj_sub[1], e->prj_sub[2], e->prj_sub[3]);
      for (int j = 0; j < 2; ++j) {
        printf(",[");
        for (int i = 0; i < e->prj_sub_n[j]; ++i) printf("%s%d", i ? "," : "", e->prj_sub_list[j][i]);
        printf("]");
      }
    }
    printf("]}");
    return;
  }
  if (e->kind == 0) {
    printf(",\"us\":{\"area_ms\":%d,\"area_life\":%d,", e->area_ms, e->area_life);
    emit_actions(e, frame);
    if (e->area_raw_ok) {
      printf(",\"area_raw\":[");
      for (int k = 0; k < 12; ++k) printf("%s%d", k ? "," : "", e->area_raw[k]);
      printf("],\"area_units\":[");
      for (int i = 0; i < e->area_n; ++i) printf("%s%d", i ? "," : "", e->area_units[i]);
      printf("]");
    }
    printf("}");
    return;
  }
  printf(",\"us\":{\"tags\":[%u,%u],\"hidden\":%d,\"immune\":%d,\"airborne\":%d,\"invisible\":%d,\"shield\":[%d,%d],"
         "\"pushed\":%d,\"push_ms\":%d,\"hit_count\":%d,\"dash_ms\":%d,\"spawn_ms\":%d,\"z\":%d,\"jump_ms\":%d,"
         "\"surface\":[%d,%d],\"invisible_again_ms\":%d,\"hop\":%d,\"shot\":[%d,%d,%d],\"special_ms\":%d,"
         "\"daggers\":%d,\"a10\":[%d,%d],\"waypoint\":[%d,%d],\"facing\":[%d,%d],\"vars\":[",
         (unsigned)(e->tags & 0xFFFFFFFFu), (unsigned)(e->tags >> 32), e->hidden, e->immune, e->airborne, e->invisible, e->shield, e->shield_max, e->pushed == 1, e->push_ms,
         e->hit_count, e->dash_ms, e->spawn_ms, e->height, e->jump_ms, e->surface_x, e->surface_y, e->invisible_again,
         e->hop, e->shot_out, e->shot_elapsed, e->shot_total, e->special_ms, e->daggers, e->a10_60, e->a10_64,
         e->waypoint_x, e->waypoint_y, e->facing_x, e->facing_y);
  for (int k = 0; k < e->var_count; ++k)
    printf("%s[%u,%d]", k ? "," : "", e->vars[k].key, e->vars[k].value);
  printf("],");
  emit_actions(e, frame);
  printf(",\"links\":[");
  for (int i = 0, m = 0; i < frame->link_n; ++i)
    if (frame->links[i][0] == e->category)
      printf("%s[%d,%d,%d,%d]", m++ ? "," : "", frame->links[i][1], frame->links[i][2], frame->links[i][3], frame->links[i][4]);
  printf("],\"unlinking\":[");
  for (int i = 0, m = 0; i < frame->unlink_n; ++i)
    if (frame->unlinks[i][0] == e->category) printf("%s[%d,%d]", m++ ? "," : "", frame->unlinks[i][1], frame->unlinks[i][2]);
  printf("]");
  if (e->bcomp_ok) {
    printf(",\"braw\":[");
    for (int k = 0; k < e->buff_count; ++k) {
      printf("%s[", k ? "," : "");
      for (int j = 0; j < 10; ++j) printf("%s%d", j ? "," : "", e->braw[k][j]);
      printf("]");
    }
    printf("],\"buffs2\":[");
    for (int k = 0; k < e->b2_n; ++k) {
      printf("%s[", k ? "," : "");
      for (int j = 0; j < 14; ++j) printf("%s%d", j ? "," : "", e->b2[k][j]);
      printf("]");
    }
    printf("],\"bcomp\":[%d,%d]", e->bcomp[0], e->bcomp[1]);
  }
  if (e->hlt_raw_ok) {
    printf(",\"hlt\":[%d,%d,%d,%d,%d,%d],\"hit_by\":[", e->hlt_raw[0], e->hlt_raw[1], e->hlt_raw[2], e->hlt_raw[3], e->hlt_raw[4], e->hlt_raw[5]);
    for (int i = 0; i < e->hit_n; ++i) printf("%s[%d,%d]", i ? "," : "", e->hit_by[i][0], e->hit_by[i][1]);
    printf("],\"hl\":[");
    for (int j = 0; j < 2; ++j) {
      printf("%s[", j ? "," : "");
      for (int i = 0; i < e->hl_n[j]; ++i) printf("%s%d", i ? "," : "", e->hl[j][i]);
      printf("]");
    }
    printf("]");
  }
  printf(",\"ent\":[");   /* troops and princess towers: ENT_RAW; the King towers' own words go to "king" (other offsets: one field, one meaning) */
  for (int k = 0; e->ent_raw_ok == 1 && k < ENT_RAW_N; ++k) printf("%s%d", k ? "," : "", e->ent_raw[k]);
  printf("],\"king\":[");
  for (int k = 0; e->ent_raw_ok == 2 && k < KING_RAW_N; ++k) printf("%s%d", k ? "," : "", e->ent_raw[k]);
  printf("],\"atk\":[");
  for (int k = 0; e->atk_raw_ok && k < ATK_RAW_N; ++k) printf("%s%d", k ? "," : "", e->atk_raw[k]);
  if (e->atk_raw_ok) {
    printf("],\"atk_lists\":[");
    for (int j = 0; j < 3; ++j) {
      printf("%s[", j ? "," : "");
      for (int i = 0; i < e->atk_list_n[j]; ++i) printf("%s%d", i ? "," : "", e->atk_list[j][i]);
      printf("]");
    }
  }
  printf("],\"mov\":[");
  for (int k = 0; e->mov_raw_ok && k < MOV_RAW_N; ++k) printf("%s%d", k ? "," : "", e->mov_raw[k]);
  if (e->mov_raw_ok) {
    printf("],\"mov_lists\":[");
    for (int j = 0; j < 2; ++j) {
      printf("%s[", j ? "," : "");
      for (int i = 0; i < e->mov_list_n[j]; ++i) printf("%s%d", i ? "," : "", e->mov_list[j][i]);
      printf("]");
    }
  }
  printf("],\"buffs\":[");
  for (int k = 0; k < e->buff_count; ++k)
    printf("%s[%u,%d,%d,%d,%d,%d,%d]", k ? "," : "", e->buffs[k].id, e->buffs[k].remaining, e->buffs[k].elapsed,
           e->buffs[k].side, e->buffs[k].speed, e->buffs[k].hit, object_id(frame, e->buffs[k].source));
  printf("]}");
}

static void read_queue(int fd, uint64_t battle, uint64_t libg, ChainFrame *frame) {
  uint64_t queue = 0, data = 0;
  int32_t count = 0;
  read_exact(fd, battle + 0x64, &frame->server_tick, 4);
  if (!read_exact(fd, battle + 0x38, &queue, 8) || !queue ||
      !read_exact(fd, queue + 0x08, &data, 8) || !read_exact(fd, queue + 0x14, &count, 4) ||
      !data || count <= 0)
    return;
  for (int i = 0; i < count && frame->queue_count < MAX_QUEUE; ++i) {
    uint64_t entry = 0, card = 0;
    uint8_t raw[0x50];
    if (!read_exact(fd, data + 8 * (uint64_t)i, &entry, 8) || !entry || !read_exact(fd, entry, raw, sizeof(raw))) continue;
    __typeof__(frame->queue[0]) *q = &frame->queue[frame->queue_count++];
    memset(q, 0, sizeof(*q));
    memcpy(&q->vt, raw, 8);
    q->vt = UNTAG(q->vt) - libg;
    memcpy(&q->t2, raw + 0x0C, 4);
    memcpy(&q->t, raw + 0x10, 4);
    memcpy(&q->hi, raw + 0x14, 4);
    memcpy(&q->lo, raw + 0x18, 4);
    memcpy(&q->a, raw + 0x28, 4);
    memcpy(&q->b, raw + 0x2C, 4);
    memcpy(&q->sid, raw + 0x30, 4);
    memcpy(&q->pd, raw + 0x48, 4);
    memcpy(&card, raw + 0x38, 8);
    q->card = -1;
    /* the ability-press class moves with the others in a shifted libg (CLAPHA_LIBG_SHIFT: Null's +0x1000, 160635009
       -0x1400): unshifted, a press there was read as a card play with a junk card id */
    if (q->vt != PLAY_ABILITY_VT_ARM + g_vt_shift && q->vt != PLAY_ABILITY_VT_X86 && UNTAG(card) > 0x100000)
      read_exact(fd, UNTAG(card) + 0x40, &q->card, 4);
  }
}

static int read_frame(int fd, uint64_t libg, uint64_t manager_rva,
                      uint64_t root_context_offset, int32_t *tick,
                      int *coherent, PlayerFrame players[2],
                      PlayerCache *cache, ChainFrame *frame, int unified) {
  uint64_t root = 0, context = 0, battle = 0;
  int32_t before = -1, after = -1;
  *tick = -1;
  *coherent = 1;
  memset(frame, 0, sizeof(*frame));
  frame->replay_tick = -1;
  frame->failure = "root_unresolved";
  if (!read_exact(fd, libg + manager_rva, &root, 8) || !root ||
      !read_exact(fd, root + root_context_offset, &context, 8) || !context ||
      !read_exact(fd, context + 0x90, &battle, 8) || !battle ||
      !read_exact(fd, battle + BATTLE_TICK, &before, 4) || before < 0)
    return 0;
  frame->root = root; frame->context = context; frame->battle = battle;
  frame->failure = "players_unresolved";
  if (cache->battle != battle) {
    memset(cache, 0, sizeof(*cache));
    cache->battle = battle;
  }
  if (!cache->player_state) {
    uint64_t direct = 0;
    if (read_exact(fd, battle + 0xA8, &direct, 8) && read_player_pair(fd, direct, players)) {
      cache->player_state = direct;
      cache->path_length = 1;
      cache->path[0] = 0xA8;
    }
  }
  uint64_t reached = battle;
  for (int i = 0; i < cache->path_length; ++i) {
    if (!reached || !read_exact(fd, reached + cache->path[i], &reached, 8)) { reached = 0; break; }
  }
  if (!cache->player_state || reached != cache->player_state ||
      !read_player_pair(fd, cache->player_state, players)) {
    cache->player_state = 0;
    /* The version-pinned unified reader accepts only the measured +0xA8
       path. Loading/menu state is not a reason to search unrelated objects. */
    if (unified) return 0;
    uint64_t now = monotonic_us();
    if (cache->last_discovery_us && now - cache->last_discovery_us < 1000000) return 0;
    cache->last_discovery_us = now;
    cache->player_state = discover_player_state(fd, battle, players, cache, frame);
    if (!cache->player_state) return 0;
  }
  frame->player_state = cache->player_state;
  frame->failure = "entities_unresolved";
  if (unified && !read_entities(fd, frame)) return 0;
  read_exact(fd, battle + 0x1BC, &frame->replay_tick, 4);
  read_queue(fd, battle, libg, frame);
  {
    uint64_t row = 0;
    if (read_exact(fd, cache->player_state + 0xB8, &row, 8) && row) read_exact(fd, UNTAG(row) + 0x40, &frame->location, 4);
    row = 0;
    if (read_exact(fd, cache->player_state + 0xD0, &row, 8) && row) read_exact(fd, UNTAG(row) + 0x40, &frame->arena, 4);
    for (int side = 0; side < 2; ++side) {   /* the profiles' card play counts (clapha-engine observe.py card_counts) */
      uint64_t prof = 0, hdr = 0, arr = 0; int32_t n = 0;
      frame->card_count_n[side] = 0;
      if (read_exact(fd, battle + 0xD8 + 8 * side, &prof, 8) && UNTAG(prof) > 0x100000 &&
          read_exact(fd, UNTAG(prof) + 0xC8, &hdr, 8) && UNTAG(hdr) > 0x100000 &&
          read_exact(fd, UNTAG(hdr), &arr, 8) && read_exact(fd, UNTAG(hdr) + 12, &n, 4) && n > 0 && n <= 256 && UNTAG(arr) > 0x100000) {
        uint64_t recs[256];
        if (read_exact(fd, UNTAG(arr), recs, (size_t)n * 8)) {
          int m = 0;
          for (int i = 0; i < n; ++i) {
            uint64_t d = 0; int32_t id = -1, v = 0;
            if (!recs[i] || !read_exact(fd, UNTAG(recs[i]) + 8, &d, 8) || !read_exact(fd, UNTAG(recs[i]) + 0x10, &v, 4)) continue;
            if (UNTAG(d) > 0x100000) read_exact(fd, UNTAG(d) + 0x40, &id, 4);
            frame->card_counts[side][m][0] = id; frame->card_counts[side][m][1] = v; ++m;
          }
          frame->card_count_n[side] = m;
        }
      }
    }
    { /* the battle clock and the logic's own words (clapha-engine observe.py battle_clock / logic_raw) */
      uint64_t clock = 0;
      frame->clock_ok = read_exact(fd, cache->player_state + 0x190, &clock, 8) && clock &&
                        read_exact(fd, UNTAG(clock) + 0x10, frame->clock, sizeof(frame->clock));
      static const uint16_t LOGIC_RAW[8] = {0x120, 0x124, 0x198, 0x1B8, 0x1E0, 0x2C0, 0x320, 0x324};
      frame->logic_raw_ok = 1;
      for (int k = 0; k < 8; ++k)
        frame->logic_raw_ok = frame->logic_raw_ok && read_exact(fd, cache->player_state + LOGIC_RAW[k], &frame->logic_raw[k], 4);
      frame->logic_raw_ok = frame->logic_raw_ok && read_exact(fd, frame->battle + 0x1F4, &frame->logic_raw[8], 4);   /* the battle context's +0x1F4 */
      /* flag bytes whose padding holds old memory on a live client (as observe.py): clock words 2 / 14, logic +0x1E0 */
      frame->clock[2] &= 0xFFFFFF; frame->clock[14] &= 0xFF; frame->logic_raw[4] &= 0xFFFF;
    }
    uint64_t settings = 0;
    row = 0;
    if (read_exact(fd, cache->player_state + 0x1B0, &settings, 8) && UNTAG(settings) > 0x100000 &&
        read_exact(fd, UNTAG(settings) + 0x18, &row, 8) && UNTAG(row) > 0x100000)
      read_exact(fd, UNTAG(row) + 0x40, &frame->game_mode, 4);
  }
  for (int side = 0; side < 2; ++side) {
    uint64_t deck = 0, slots = 0;
    int32_t n = 0;
    players[side].battle_deck_count = 0;
    if (read_exact(fd, cache->player_state + 0x88 + 8 * (uint64_t)side, &deck, 8) && UNTAG(deck) > 0x100000 &&
        read_exact(fd, UNTAG(deck) + 0x20, &slots, 8) && read_exact(fd, UNTAG(deck) + 0x20 + 12, &n, 4) &&
        UNTAG(slots) > 0x100000 && n > 0 && n <= 8) {
      for (int k = 0; k < n; ++k) {
        uint64_t slot = 0, data = 0;
        int32_t *d = players[side].battle_deck[k];
        d[0] = d[1] = d[2] = -1;
        if (!read_exact(fd, UNTAG(slots) + 8 * (uint64_t)k, &slot, 8) || UNTAG(slot) < 0x100000) continue;
        if (read_exact(fd, UNTAG(slot) + 0x10, &data, 8) && UNTAG(data) > 0x100000) read_exact(fd, UNTAG(data) + 0x40, &d[0], 4);
        read_exact(fd, UNTAG(slot) + 0x18, &d[1], 4);
        read_exact(fd, UNTAG(slot) + 0x1C, &d[2], 4);
      }
      players[side].battle_deck_count = n;
    }
    uint64_t identity = 0;
    read_exact(fd, players[side].address + 0x24, &players[side].deal_id, 4);
    if (read_exact(fd, cache->player_state + 0x30 + 8 * (uint64_t)side, &identity, 8) && identity) {
      read_exact(fd, identity, &players[side].account_hi, 4);
      read_exact(fd, identity + 4, &players[side].account_lo, 4);
    }
  }
  frame->failure = "incoherent_frame";
  if (!read_exact(fd, battle + BATTLE_TICK, &after, 4)) return 0;
  uint64_t root_after = 0, context_after = 0, battle_after = 0;
  if (!read_exact(fd, libg + manager_rva, &root_after, 8) || root_after != root ||
      !read_exact(fd, root_after + root_context_offset, &context_after, 8) || context_after != context ||
      !read_exact(fd, context_after + 0x90, &battle_after, 8) || battle_after != battle) {
    *coherent = 0;
    return 0;
  }
  *tick = after;
  *coherent = before == after;
  /* a read made while the game is inside a step: the battle clock (logic +0x190, +0x10) has moved to the new tick but the
     battle tick has not (2026-11-04, a live ladder battle: in 15 of 6,174 frames clock == tick, each one a frame check 4 found
     half-updated; between steps the clock is always tick - 1, as in the engine). Such a frame is read again */
  /* a play being applied while the frame was read (2026-11-04: the opponent's elixir paid and elixir_spent up, cards_played not
     yet -- the clock had not moved): each player's elixir, elixir spent and cards played read again must equal the frame's */
  for (int side = 0; *coherent && side < 2; ++side) {
    int32_t e = 0, spent = 0, played = 0;
    if (players[side].address &&
        read_exact(fd, players[side].address + PLAYER_ELIXIR_RAW, &e, 4) && read_exact(fd, players[side].address + 0x340, &spent, 4) &&
        read_exact(fd, players[side].address + 0x318, &played, 4) &&
        (e != players[side].elixir_raw || spent != players[side].elixir_spent || played != players[side].cards_played))
      *coherent = 0;
  }
  if (*coherent && cache->player_state) {
    uint64_t clk = 0; int32_t clock_tick = -1;
    if (read_exact(fd, cache->player_state + 0x190, &clk, 8) && clk && read_exact(fd, UNTAG(clk) + 0x10, &clock_tick, 4) &&
        clock_tick == after)
      *coherent = 0;
  }
  if (*coherent) frame->failure = "none";
  return 1;
}

static void emit_chain(const ChainFrame *f, const PlayerCache *cache, uint64_t libg) {
  printf(",\"chain\":{\"libg_base\":\"0x%" PRIx64 "\",\"root\":\"0x%" PRIx64
         "\",\"context\":\"0x%" PRIx64 "\",\"battle\":\"0x%" PRIx64
         "\",\"player_state\":\"0x%" PRIx64 "\",\"hp_state\":\"0x%" PRIx64
         "\",\"registry\":\"0x%" PRIx64 "\",\"collection\":\"0x%" PRIx64
         "\",\"data\":\"0x%" PRIx64 "\",\"player_state_path\":[",
         libg, f->root, f->context, f->battle, f->player_state, f->hp_state, f->registry, f->collection, f->data);
  for (int i = 0; i < cache->path_length; ++i) { if (i) putchar(','); printf("%u", cache->path[i]); }
  printf("]},\"failure\":\"%s\",\"applied_replay_tick\":%d,\"native_object_count\":%d,"
         "\"decoded_entity_count\":%d,\"filtered_object_count\":%d,\"discovery_nodes\":%d,"
         "\"discovery_budget_exhausted\":%s", f->failure, f->replay_tick, f->native_count,
         f->entity_count, f->filtered_count, f->discovery_nodes, f->discovery_budget_exhausted ? "true" : "false");
}

/* card_counts = the cards each player has played IN THIS BATTLE, in first-play order, and how often. On a live client the
   profile record behind it (battle +0xD8 / +0xE0 -> +0xC8) starts the battle already holding the account's career counts
   (2026-11-04 on 160635009: 44 and 37 entries adding up to 4,145 / 3,423 with 8 and 9 cards played; break-the-checks round 3,
   AH); in a replay it starts empty. So the counts at the battle's first frame are a baseline and only what is added after
   it is output -- live and replay give the same field. card_counts_since: the tick the baseline was taken (-1: none, the reader first saw the battle after tick 20 and outputs the record as it is; 0 or the first
   tick: complete; later: the reader joined mid-battle and counts only from there). State across frames of one process. */
static struct {
  uint64_t battle; int32_t seed, last_tick, since, started, base_n[2], base[2][256][2], order_n[2], order[2][64];
} ccb = {0, 0, 0, -1, 0, {0, 0}, {{{0}}}, {0, 0}, {{0}}};

static int32_t cc_find(int32_t n, int32_t list[][2], int32_t id) {
  for (int i = 0; i < n; ++i) if (list[i][0] == id) return list[i][1];
  return 0;
}

static void card_counts_this_battle(ChainFrame *frame, int32_t tick, int32_t seed, const PlayerFrame players[2]) {
  if (frame->battle != ccb.battle || seed != ccb.seed || tick < ccb.last_tick || !ccb.started) {
    /* a baseline only when the battle is seen before any play can have landed (the first lands at tick 21: issue + 21);
       seen later, the record is output as it is and since = -1 says so (check 3 found the late baseline swallowing plays) */
    int early = tick >= 0 && tick < 21;
    ccb.battle = frame->battle; ccb.seed = seed; ccb.since = early ? tick : -1; ccb.started = 1;
    for (int side = 0; side < 2; ++side) {
      ccb.base_n[side] = early ? frame->card_count_n[side] : 0;
      memcpy(ccb.base[side], frame->card_counts[side], sizeof(ccb.base[side]));
      ccb.order_n[side] = 0;
    }
  }
  ccb.last_tick = tick;
  for (int side = 0; side < 2; ++side) {
    int32_t n = frame->card_count_n[side];
    if (ccb.since < 0) {
      /* no baseline (first seen after tick 20): the record is this battle's only if its counts add up to the player's cards
         played (the engine's record, which holds nothing else); a live record with the career in it gives nothing (round 4, AM) */
      int32_t sum = 0;
      for (int i = 0; i < n; ++i) sum += frame->card_counts[side][i][1];
      int32_t played = players[side].cards_played;   /* players[] is indexed by side (emit_player) */
      if (sum != played) frame->card_count_n[side] = 0;
      continue;
    }
    for (int i = 0; i < n; ++i) {             /* a card played for the first time in this battle joins the order */
      int32_t id = frame->card_counts[side][i][0];
      if (frame->card_counts[side][i][1] - cc_find(ccb.base_n[side], ccb.base[side], id) <= 0) continue;
      int known = 0;
      for (int k = 0; k < ccb.order_n[side]; ++k) if (ccb.order[side][k] == id) known = 1;
      if (!known && ccb.order_n[side] < 64) ccb.order[side][ccb.order_n[side]++] = id;
    }
    int32_t out[64][2]; int m = 0;
    for (int k = 0; k < ccb.order_n[side]; ++k) {
      int32_t id = ccb.order[side][k];
      out[m][0] = id;
      out[m][1] = cc_find(n, frame->card_counts[side], id) - cc_find(ccb.base_n[side], ccb.base[side], id);
      if (out[m][1] > 0) ++m;
    }
    memcpy(frame->card_counts[side], out, sizeof(out));
    /* (round 4, AM) first seen after tick 20: the record still holds a live account's career counts -- output nothing (unknown,
       card_counts_since -1) rather than the career; check 5 fails a played side whose list stays empty */
    frame->card_count_n[side] = m;
  }
}

static void emit_player(const PlayerFrame *player, int side, const ChainFrame *frame) {
  printf("{\"side\":%d,\"card_counts_since\":%d,\"card_counts\":[", side, ccb.since);
  for (int i = 0; i < frame->card_count_n[side]; ++i)
    printf("%s[%d,%d]", i ? "," : "", frame->card_counts[side][i][0], frame->card_counts[side][i][1]);
  printf("],\"battle_deck\":[");
  for (int k = 0; k < player->battle_deck_count; ++k)
    printf("%s[%d,%d,%d]", k ? "," : "", player->battle_deck[k][0], player->battle_deck[k][1], player->battle_deck[k][2]);
  printf("],\"deal_id\":%d,\"account\":[%d,%d],\"address\":\"0x%" PRIx64
         "\",\"elixir_raw\":%d,\"refill_timer\":%d,\"cards_played\":%d,\"elixir_spent\":%d,\"elixir_wasted\":%d,"
         "\"first_cards\":[%d,%d,%d],\"slot_words\":[%d,%d,%d,%d,%d,%d,%d,%d],\"player_raw\":[%d,%d,%d,%d,%d,%d],"
         "\"next_deck_index\":%d,\"hand_deck_indices\":[%d,%d,%d,%d],"
         "\"cycle_deck_indices\":[",
         player->deal_id, player->account_hi, player->account_lo, player->address, player->elixir_raw, player->refill_timer,
         player->cards_played, player->elixir_spent, player->elixir_wasted, player->first_cards[0], player->first_cards[1], player->first_cards[2],
         player->slot_words[0], player->slot_words[1], player->slot_words[2], player->slot_words[3], player->slot_words[4],
         player->slot_words[5], player->slot_words[6], player->slot_words[7], player->player_raw[0], player->player_raw[1], player->player_raw[2], player->player_raw[3], player->player_raw[4], player->player_raw[5],
         player->next_deck_index, player->hand[0], player->hand[1],
         player->hand[2], player->hand[3]);
  for (int index = 0; index < player->cycle_size; ++index) {
    if (index) putchar(',');
    printf("%d", player->cycle[index]);
  }
  printf("],\"deck_card_ids\":[");
  if (player->deck_visible) {
    for (int index = 0; index < 8; ++index) {
      if (index) putchar(',');
      printf("%d", player->deck_card_ids[index]);
    }
  }
  printf("],\"deck_form_flags\":[");
  if (player->deck_visible) {
    for (int index = 0; index < 8; ++index) {
      if (index) putchar(',');
      printf("%d", player->deck_form_flags[index]);
    }
  }
  printf("],\"deck_level_index\":[");
  if (player->deck_visible) {
    for (int index = 0; index < 8; ++index) {
      if (index) putchar(',');
      printf("%d", player->deck_level_index[index]);
    }
  }
  printf("],\"evo_progress\":[");
  for (int index = 0; index < player->evo_count; ++index) {
    if (index) putchar(',');
    printf("%d", player->evo_progress[index]);
  }
  printf("],\"abilities\":[");
  int first = 1;
  for (int i = 0; i < 2; ++i) {
    if (!player->ability_present[i]) continue;
    printf("%s{\"controller_slot\":%d,\"charges\":%d,\"button\":%d,\"cooldown_ms\":%d,"
           "\"configured_ms\":%d,\"character_id\":%u,\"used\":%d,\"effect_delay_ms\":%d,\"units\":[",
           first ? "" : ",", i + 1, player->ability_charges[i], player->ability_button[i],
           player->ability_cooldown[i], player->ability_configured[i],
           player->ability_character[i], player->ability_used[i] != 0, player->ability_effect_delay[i]);
    for (int k = 0; k < player->ability_unit_count[i]; ++k)
      printf("%s%d", k ? "," : "", object_id(frame, player->ability_units[i][k]));
    printf("]}");
    first = 0;
  }
  printf("]}");
}

int main(int argc, char **argv) {
  if (argc != 5 && argc != 7) {
    fprintf(stderr,
            "usage: mumu-live-private PID INTERVAL_MS MANAGER_RVA "
            "ROOT_CONTEXT_OFFSET [--unified MAX_FRAMES(0=unlimited)]\n");
    return 2;
  }
  int pid = atoi(argv[1]);
  int interval_ms = atoi(argv[2]);
  uint64_t manager_rva = strtoull(argv[3], NULL, 0);
  uint64_t root_context_offset = strtoull(argv[4], NULL, 0);
  int unified = argc == 7;
  uint64_t max_frames = unified ? strtoull(argv[6], NULL, 10) : 0;
  if (unified && strcmp(argv[5], "--unified")) return 2;
  if (pid <= 0 || interval_ms < 20 || interval_ms > 5000 || !manager_rva ||
      !root_context_offset)
    return 2;
  /* CLAPHA_LIBG_BASE (tests only): the library's base when it is mapped without its file name, as clapha-engine's crx
     loads it (tools/winport/reader_vs_engine.py); unset on a device, where /proc/PID/maps names it */
  const char *forced = getenv("CLAPHA_LIBG_BASE");
  uint64_t libg = forced ? strtoull(forced, NULL, 0) : find_libg_base(pid);
  if (!libg) return 3;
  g_libg = libg;
  const char *shift = getenv("CLAPHA_LIBG_SHIFT");
  g_vt_shift = shift ? strtoull(shift, NULL, 0) : 0;
  char memory_path[64];
  snprintf(memory_path, sizeof(memory_path), "/proc/%d/mem", pid);
  /* (2026-11-05, Terrasse's profile of the in-process export) a caller that asks for a fixed number of frames, again and
     again in one process (play_frames.py: one frame per tick), keeps the memory handle, the frame buffer and the player cache
     between calls -- as the live loop keeps them between frames; the cache resets itself when the battle changes. The frame
     is cleared each call, as the fresh calloc was. Output: one write per frame (it was unbuffered: a write per field). */
  static int keep_fd = -1, keep_pid = 0, buffered = 0;
  static ChainFrame *keep_frame = NULL;
  static PlayerCache keep_cache;
  int keep = unified && max_frames > 0;
  int fd;
  if (keep && keep_fd >= 0 && keep_pid == pid) {
    fd = keep_fd;
  } else {
    fd = open(memory_path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return 4;
    if (keep) {
      if (keep_fd >= 0) close(keep_fd);
      keep_fd = fd; keep_pid = pid;
      memset(&keep_cache, 0, sizeof(keep_cache));
    }
  }
  if (!buffered) { setvbuf(stdout, NULL, _IOFBF, 1 << 20); buffered = 1; }
  uint64_t sequence = 0;
  PlayerCache local_cache = {0};
  PlayerCache *cachep = keep ? &keep_cache : &local_cache;
  ChainFrame *frame = keep ? keep_frame : NULL;
  if (!frame) frame = calloc(1, sizeof(*frame));
  else memset(frame, 0, sizeof(*frame));
  if (!frame) { if (!keep) close(fd); return 5; }
  if (keep) keep_frame = frame;
#define cache (*cachep)
  PlayerFrame players[2];
  while ((!max_frames || sequence < max_frames) && (kill(pid, 0) == 0 || errno == EPERM)) {
    uint64_t started = monotonic_us();
    int32_t tick = -1, seed[2] = {0, 0};
    int coherent = 1;
    int active = 0;
    for (int attempt = 0; attempt < 4; ++attempt) {
      active = read_frame(fd, libg, manager_rva, root_context_offset, &tick,
                          &coherent, players, &cache, frame, unified);
      if (!active || coherent) break;
      usleep(2000);                    /* the game is inside a step: read again once it has finished */
    }
    if (active && frame->battle) read_exact(fd, frame->battle + BATTLE_SEED, seed, sizeof(seed));
    /* schema 2 still: the unit states ("us" per object, ability "used" / "effect_delay_ms") only add keys, and
       native_core's BattleClockGuard accepts schema 2 alone */
    printf("{\"event\":\"%s\",\"schema_version\":2,\"pid\":%d,\"reader_pid\":%d,\"sequence\":%" PRIu64
           ",\"battle_active\":%s,\"coherent\":%s,\"game_tick\":%d,\"seed\":%d,\"rng\":%d,"
           "\"players\":[",
           unified ? "mumu_live_frame" : "mumu_live_private", pid, getpid(), sequence++, active ? "true" : "false",
           coherent ? "true" : "false", tick, seed[0], seed[1]);
    if (active) {
      card_counts_this_battle((ChainFrame *)frame, tick, seed[1], players);
      emit_player(&players[0], 0, frame);
      putchar(',');
      emit_player(&players[1], 1, frame);
    }
    putchar(']');
    if (active) {
      printf(",\"clock\":[");
      for (int k = 0; frame->clock_ok && k < 20; ++k) printf("%s%d", k ? "," : "", frame->clock[k]);
      printf("],\"logic_raw\":[");
      for (int k = 0; frame->logic_raw_ok && k < 9; ++k) printf("%s%d", k ? "," : "", frame->logic_raw[k]);
      printf("]");
      printf(",\"location\":%d,\"arena\":%d,\"game_mode\":%d,\"server_tick\":%d,\"queue\":[", frame->location, frame->arena,
             frame->game_mode, frame->server_tick);
      /* (round 5, AQ) the queue means the same in a conversion and live: a conversion's engine holds every command of the
         battle from tick 0 (the future), a live client a command only 6-19 ticks before it lands (1,920 commands on 160635009,
         2026-11-04: first seen 6 .. 19 ticks before, 2 at 6). So a command is output from 6 ticks before its landing until it
         lands, with its landing tick (t2; issue + 20 while a live client still shows -1) */
      int shown = 0;
      for (int i = 0; i < frame->queue_count; ++i) {
        __typeof__(frame->queue[0]) *q = &frame->queue[i];
        int32_t landing = q->t2 >= 0 ? q->t2 : q->t + 20;
        if (landing - tick < 0 || landing - tick > 6) continue;
        /* kind: the command's family, the same on every build and platform (break-the-checks round 3, AI: vt is a code
           address -- 0x1af1508 in the x86 engine, 0x1a28748 / 0x1a27348 on the phone in 160635001 / 160635009 -- so the
           model gets kind, not vt). 1 a card play, 2 an ability press, 0 another family. vt minus the build's shift is
           160635001's ARM64 address */
        uint64_t base_vt = q->vt - g_vt_shift;
        int kind = base_vt == 0x1a28748 ? 1 : base_vt == 0x1a28bc8 ? 2 : 0;
        /* (round 6, AW) an ability press: a replay stores t = landing - 2 and no hero id, and +0x48 holds heap words in the
           engine, while a live client shows the press tick (landing - 20), the hero's id in b and 0 in pd. So for a press:
           t = landing - 20, b = -1, pd = 0 -- the same in both (the hero is the unit in a) */
        int32_t out_t = kind == 2 ? landing - 20 : q->t, out_b = kind == 2 ? -1 : q->b, out_pd = kind == 2 ? 0 : q->pd;
        /* (round 7) which side issued it: the account ids in id are who, not what (a pro player in a conversion, the user
           live), so the input is the side, matched through the players' accounts (-1: no match) */
        int q_side = -1;
        for (int s2 = 0; s2 < 2; ++s2)
          if (players[s2].account_hi == q->hi && players[s2].account_lo == q->lo) q_side = s2;
        printf("%s{\"vt\":\"0x%" PRIx64 "\",\"kind\":%d,\"side\":%d,\"t\":%d,\"t2\":%d,\"id\":[%d,%d],\"a\":%d,\"b\":%d,\"sid\":%d,\"card\":%d,\"pd\":%d}",
               shown++ ? "," : "", q->vt, kind, q_side, out_t, landing, q->hi, q->lo, q->a, out_b, q->sid, q->card, out_pd);
      }
      putchar(']');
    }
    emit_chain(frame, &cache, libg);
    if (unified) {
      printf(",\"entities\":[");
      for (int i = 0; active && i < frame->entity_count; ++i) {
        const EntityFrame *e = &frame->entities[i];
        if (i) putchar(',');
        printf("{\"address\":\"0x%" PRIx64 "\",\"category\":%d,\"kind\":%d,\"side\":%d,"
               "\"x\":%d,\"y\":%d,\"card_id\":%d,\"level\":%d,\"hp\":%d,\"max_hp\":%d,"
               "\"behavior_state_raw\":%d,\"deploy_remaining\":%d,\"has_attack\":%d,"
               "\"target\":\"0x%" PRIx64 "\",\"atk_stage\":%d,\"atk_timeline\":%d,"
               "\"atk_load\":%d,\"has_move\":%d,\"charge\":%d,\"data_id\":%u,\"tx\":%d,\"ty\":%d",
               e->address, e->category, e->kind, e->side,
               e->x, e->y, e->card_id, e->level, e->hp, e->max_hp, e->behavior,
               e->deploy_remaining, e->has_attack, e->target, e->atk_stage, e->atk_timeline,
               e->atk_load, e->has_move, e->charge, e->data_id, e->target_x, e->target_y);
        if (e->us) emit_unit_states(e, frame);
        putchar('}');
      }
      putchar(']');
    }
    printf(",\"sample_monotonic_us\":%" PRIu64 ",\"read_us\":%" PRIu64 "}\n", started, monotonic_us() - started);
    fflush(stdout);                    /* one write per frame, a whole line (the live stream reads lines) */
    /* (2026-11-05, Terrasse) a caller that asks for a fixed number of frames (play_frames.py: one frame per tick, the engine
       stepped in between) must not wait out the sampling interval after the last one: that sleep was 20-50 ms per tick --
       measured by Terrasse as 42 ms a frame, 26x the old ctypes export */
    if (max_frames && sequence >= max_frames) break;
    uint64_t elapsed = monotonic_us() - started;
    uint64_t target = (uint64_t)interval_ms * 1000ULL;
    if (elapsed < target) usleep((useconds_t)(target - elapsed));
  }
#undef cache
  if (!keep) {
    close(fd);
    free(frame);
  }
  return 0;
}
