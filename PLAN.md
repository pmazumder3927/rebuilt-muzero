# REBUILT MuZero Drive Coach — Project Plan

Goal: build a MuZero-style **macro decision coach** for the 2026 FRC game **REBUILT** that recommends high-level plays every **~1–3 seconds** (not joystick control), optimizing either:

- **Playoffs**: win probability / score margin
- **Qualifications**: ranking points + match points

This repo will implement a **blazing-fast macro-simulator** first, then MuZero self-play training, then a real-time “drive coach” UI.

---

## 0) Key decision (affects action/obs design)

We can support both, but we should decide early which is the primary product:

1. **Single-robot coach**: recommend the best intent for *your robot* given teammates/opponents modeled as policies.
2. **Whole-alliance coordinator**: recommend role assignments for your **3 robots** (scorer/collector/defender/climber/feeder).

The macro-sim will be written to support both by:

- Representing all **6 robots** in state.
- Allowing `step()` to take either:
  - actions for all 6 robots, or
  - actions for your 3 robots + an opponent policy/baseline for the rest.

---

## 1) What must be modeled in REBUILT (strategy-critical mechanics)

These directly shape simulator state, transitions, and reward.

### Match phases + HUB activation windows

- Match timeline:
  - **AUTO**: 20s
  - **TELEOP**: 2:20 (140s) = **Transition 10s** + **4× Alliance Shifts (25s each)** + **End Game 30s**
- **Both HUBs active** during: AUTO, Transition, End Game
- During each **Alliance Shift**: only **one alliance’s HUB is active**
  - Shift order: determined by who scored more **FUEL in AUTO** (random if tied), then alternates.

Strategic tension:

- HUB active → cycle + cash in
- HUB inactive → scoring = 0 → steal/deny/stage/defend/prep climb

### Scoring (reward target)

- **FUEL scored in an active HUB**: +1 point (AUTO + TELEOP)
- **Inactive HUB**: 0 points
- **Tower** scoring exists (levels differ in AUTO vs TELEOP/ENDGAME)
- Ranking bonuses / RPs exist (quals mode)

### Stochasticity: HUB redistributes scored fuel

- FUEL scored is redistributed to the neutral zone through **one of four exits randomly**.
- This stochasticity is a primary driver of “steal vs cycle vs stage” decisions.

### Human player logistics via OUTPOST

- OUTPOST chute stores ~**25 FUEL** behind a door.
- Robots can deliver FUEL via the OUTPOST CORRAL to be fed back later.
- Macro-sim should make OUTPOST staging valuable when HUB is inactive.

### Rule constraints (sim needs these penalties)

- **Must be at least partially in ALLIANCE ZONE to score in your HUB**.
- **No “catching/redirecting” HUB-released fuel** (penalize “camp under hub to farm ejected fuel” behaviors).
- **Last 30s tower protection**: no contact with opponent robot touching their tower (major foul + can award them L3 tower points).
- **3-second pin limit** (penalty risk if exceeded).

---

## 2) Environment approach

We implement **Option A: fast macro-sim** now, then optionally calibrate it from xRC later.

### Why macro-sim

- MuZero needs huge experience (self-play + MCTS).
- We model tasks as **timed macros** instead of physics:
  - travel time
  - intake time
  - shooting time
  - defense delays / accuracy reduction
  - penalty risk

Target: **thousands of matches/hour** on a laptop (ideally much higher).

---

## 3) The “Drive Coach MDP”

### 3.1 Observation (structured, coach-realistic)

**Match state**

- `t` (seconds remaining)
- `phase` ∈ {AUTO, TRANSITION, SHIFT1..SHIFT4, ENDGAME}
- `active_hubs` (both / red / blue)
- `score_red`, `score_blue`
- penalty points (or foul counters)

**Resource state (coarse bins)**

- `neutral_bins[NB]` (e.g., 8–12 regions)
- `depot_fuel[2]`
- `outpost_chute[2]` (capacity 25)
- optional: `near_hub_bin[2]` as a convenience feature

**Robot state (6 robots, small vector per robot)**

- `region_id` (coarse location)
- `fuel_carried` (0..capacity)
- `busy_until` or remaining task time
- `climb_capability` (max level)
- `climb_ready` flag
- capability tiers (inputs for generalization):
  - speed tier
  - intake tier
  - shooting accuracy tier
  - climb time tier

### 3.2 Actions (macro, factored)

For a **single robot**:

1. `COLLECT_NEUTRAL(bin_id)`
2. `COLLECT_DEPOT`
3. `SCORE_HUB`
4. `DELIVER_OUTPOST`
5. `DEFEND_OPPONENT_HUB_LANE`
6. `DEFEND_OPPONENT_COLLECTOR`
7. `PREP_CLIMB(level)`
8. `CLIMB(level)`

For **whole-alliance coordination** (preferred for MCTS tractability):

- At each decision step, output `(role_i, target_i)` for each of your 3 robots
  - roles: {scorer, collector, defender, climber, feeder}
  - targets: {your hub lane, neutral bin k, outpost, tower}

### 3.3 Reward (two modes)

**Playoffs mode**

- terminal reward: `score_us - score_them`
- shaping: incremental points

**Qualifications mode**

- match points (W/T/L) + bonus RP attainment
- secondary: score margin as tie-breaker

**Penalty shaping**

- negative reward for fouls/major fouls
- explicit risk penalties for endgame tower-contact defense

---

## 4) Macro-simulator design (first implementation target)

### 4.1 Internal simulator state

- match clock + phase
- hub active schedule (depends on auto fuel outcome)
- fuel counts per region bin + depot/outpost storage
- per-robot:
  - region
  - carried fuel
  - task in progress + finish time
  - cooldowns (optional)
- tower/climb state
- penalties accumulated

### 4.2 Transition model (“timed tasks”, not physics)

Examples:

- `COLLECT_NEUTRAL(k)`:
  - travel time from current region → `NEUTRAL[k]`
  - intake time (tiered)
  - adds fuel to robot from that bin
  - may be delayed by defense probabilistically

- `SCORE_HUB`:
  - must be in alliance zone (or apply major foul)
  - shooting time + accuracy model
  - if hub active: +points and fuel removed from robot
  - scored fuel reappears via stochastic HUB exit redistribution into neutral bins

- `DEFEND_*`:
  - applies modifiers (opponent travel + miss probability)
  - includes pin-rule timers and penalty risk

### 4.3 Shift timing is explicit

The sim must make these strategies possible:

- stage fuel during inactive shift, burst score during active shift
- deny/steal when you can’t score
- endgame pivot: stop defense near towers due to major foul swing

---

## 5) MuZero training plan (after macro-sim is validated)

### 5.1 What is learned

- representation: `s_t = h(o_t)`
- dynamics: `(s_{t+1}, r_{t+1}) = g(s_t, a_t)`
- prediction: `(π_t, v_t) = f(s_t)`
- action selection: MCTS over macro actions / role assignments

### 5.2 Self-play setup

- symmetric self-play (alliance vs alliance)
- opponent mixture: scripted baselines + past checkpoints

### 5.3 Curriculum

1. no defense, deterministic redistribution
2. stochastic redistribution
3. defense delays + pin constraints
4. endgame tower protection (major foul)
5. outpost logistics + chute/corral limits

### 5.4 Baselines (needed early)

- pure cycler
- shift-aware burst scorer
- “defend when inactive” heuristic

Use baselines for:

- sim sanity checks
- early opponents
- regression testing

---

## 6) “Drive coach” tool (deployment plan)

### 6.1 Real-time state estimation (belief state)

In real matches, fuel distribution is uncertain. Use a belief:

- initialize from known staged counts
- update using your own observed collect/score
- allow human inputs (“their depot looks empty”)
- optionally integrate AprilTags for zone-level pose

### 6.2 Human-usable outputs

- top-3 plays for the next decision window
- predicted score swing (mean ± uncertainty)
- risk flags (tower contact risk, foul risk, pin risk)
- natural-language summary (“Shift 2 active: shoot now; Shift 3 inactive: deny + stage; begin climb at 0:35”)

### 6.3 Legality constraints

- advisory-only (no illegal control path)
- obey driver station device/wireless restrictions (safest: Wi‑Fi/Bluetooth off on DS setup)

---

## 7) Implementation checklist (repo milestones)

### A — Game engine (no ML yet)

1. match clock + phase schedule + hub active rules
2. scoring:
   - fuel points only when hub active
   - tower points by level + phase
3. hub redistribution stochasticity (4 exits → neutral bins)
4. fouls as negative reward:
   - scoring from outside alliance zone (major)
   - endgame tower contact (major + big swing)
   - pinning too long

### B — Robot archetypes

5. define robot archetypes (cycler/defender/climber/etc.)
6. randomize archetypes per episode (serializable)

### C — Baselines

7. implement 2–3 scripted strategies

### D — MuZero

8. implement `h/g/f` networks + MCTS over macro actions
9. train via self-play vs mixture (scripted + checkpoints)
10. evaluate on a held-out scenario suite (seeded)

### E — Deploy

11. build dashboard + input adapters (manual/FMS/pose optional)
12. show top-3 plays + explanation

