# Visualization

A matplotlib debug renderer for the macro simulator. Useful for sanity-checking
configs, watching scripted policies, and producing GIFs.

It draws:

- An optional field background image
- Neutral fuel bins (marker size + label = count; HUB-exit bins highlighted)
- Robots (position, carried fuel, climb level)
- Current task lines with task name + remaining busy time
- A right-hand info panel with phase, active HUB(s), scores, inventories

## What are `N0..N7`?

`N0..N7` are coarse *neutral fuel bins* used by the macro sim — they are not
official FIELD labels. The simulator tracks FUEL as integer counts per bin
(plus DEPOT and OUTPOST storage), and uses these bins for:

- collecting decisions (`COLLECT_NEUTRAL(Nk)`)
- stochastic HUB fuel redistribution (4 HUB exits map into 4 bins)
- a lightweight proxy for "where fuel is" without simulating every piece

You can change bin positions by editing `GameConfig.region_coords_ft`
(see `rebuilt_muzero/sim/config.py`).

## Quick start

Optional: fetch a field image + obstacle layout from CycleTimeHeatmap:

```bash
python scripts/fetch_cycleheatmap_assets.py --dest assets
```

This writes `assets/field.png` and `assets/saved_layout.json` (gitignored).

Run the visualizer:

```bash
python scripts/visualize_macro_sim.py --policy greedy
```

Useful options:

| Flag | Effect |
| --- | --- |
| `--policy random\|greedy\|idle` | Action policy for all 6 robots |
| `--fps 12` | Playback speed |
| `--save-frames .tmp/vizframes --no-show` | Render PNG frames headlessly |
| `--field-image <path>` / `--layout-json <path>` | Override asset paths |

## Programmatic rendering

`rebuilt_muzero.sim.render_env_matplotlib(env, ...)` returns `(fig, (ax_field,
ax_info))`. See `rebuilt_muzero/sim/render.py`.
