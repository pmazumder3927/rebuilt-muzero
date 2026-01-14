# Macro-Sim Visualization (Debug UI)

This repo includes a matplotlib-based debug renderer for the REBUILT macro-simulator.

It draws:

- a field background image (optional)
- neutral fuel bins (size + label = count; exit bins highlighted)
- robots (position, carried fuel, climb level)
- current task lines + task name + remaining busy time
- a right-hand info panel (phase, HUB active, scores, inventories)

## What do `N0..N7` mean?

`N0..N7` are **coarse “neutral fuel bins”** in the macro-sim. They are **not** official FIELD labels.

The simulator tracks FUEL as integer counts per bin (plus DEPOT + OUTPOST storage), and uses these bins for:

- collecting decisions (`COLLECT_NEUTRAL(Nk)`)
- stochastic HUB fuel redistribution (the 4 HUB exits map into 4 bins)
- a lightweight proxy for “where fuel is” without simulating every individual piece

You can move/reshape bins by editing `GameConfig.region_coords_ft` (or by changing the defaults in `rebuilt_muzero/sim/config.py`).

## Quick start

1) (Optional but recommended) fetch a field image + obstacle layout from the CycleTimeHeatmap project:

`python3 scripts/fetch_cycleheatmap_assets.py --dest assets`

This writes `assets/field.png` and `assets/saved_layout.json` (ignored by git).

2) Run the visualizer:

`python3 scripts/visualize_macro_sim.py --policy greedy`

Useful options:

- `--policy random|greedy|idle`
- `--fps 12`
- `--save-frames .tmp/vizframes` (renders PNG frames without showing a window if you also pass `--no-show`)
- `--field-image <path>` / `--layout-json <path>`

## Programmatic rendering

Use `rebuilt_muzero.sim.render_env_matplotlib()` from a notebook or another script:

- `rebuilt_muzero/sim/render.py:1`
