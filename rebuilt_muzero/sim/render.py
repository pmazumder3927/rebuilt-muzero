from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rebuilt_muzero.sim.actions import ActionKind


@dataclass(frozen=True, slots=True)
class RenderConfig:
    field_extent_ft: tuple[float, float, float, float] | None = None  # (xmin,xmax,ymin,ymax)
    background_alpha: float = 0.95
    fuel_marker_scale: float = 14.0
    robot_marker_size: float = 90.0
    show_bin_labels: bool = True
    show_robot_labels: bool = True
    show_task_lines: bool = True
    show_hub_exit_bins: bool = True
    show_neutral_fuel_box: bool = True
    show_center_line: bool = True
    show_layout_obstacles: bool = True
    layout_obstacle_alpha: float = 0.6


def _default_extent_from_coords(coords: np.ndarray) -> tuple[float, float, float, float]:
    xmin = float(np.min(coords[:, 0]))
    xmax = float(np.max(coords[:, 0]))
    ymin = float(np.min(coords[:, 1]))
    ymax = float(np.max(coords[:, 1]))
    dx = max(5.0, 0.15 * (xmax - xmin))
    dy = max(3.0, 0.15 * (ymax - ymin))
    return xmin - dx, xmax + dx, ymin - dy, ymax + dy


def _load_cycleheatmap_layout(path: Path) -> dict[str, Any] | None:
    try:
        import json

        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            return None
        if not all(k in data for k in ("w", "h", "blocked")):
            return None
        return data
    except Exception:
        return None


def _grid_cell_to_field_ft(
    *,
    r: int,
    c: int,
    w: int,
    h: int,
    field_length_ft: float,
    field_width_ft: float,
) -> tuple[float, float, float, float]:
    """
    Map a CycleTimeHeatmap grid cell (r,c) to field coordinates in feet.

    The heatmap uses a w×h grid spanning the playable field. We scale cell size so the
    entire grid spans `field_length_ft × field_width_ft`.
    """
    cell_dx = float(field_length_ft) / float(w)
    cell_dy = float(field_width_ft) / float(h)
    x = (float(c) + 0.5) * cell_dx - float(field_length_ft) / 2.0
    y = float(field_width_ft) / 2.0 - (float(r) + 0.5) * cell_dy
    return x, y, cell_dx, cell_dy


def _region_name(region_id: int, *, n_neutral_bins: int) -> str:
    if region_id == 0:
        return "RED_HUB"
    if region_id == 1:
        return "BLUE_HUB"
    if 2 <= region_id < 2 + n_neutral_bins:
        return f"N{region_id - 2}"
    if region_id == 2 + n_neutral_bins:
        return "RED_OUTPOST"
    if region_id == 2 + n_neutral_bins + 1:
        return "BLUE_OUTPOST"
    if region_id == 2 + n_neutral_bins + 2:
        return "RED_TOWER"
    if region_id == 2 + n_neutral_bins + 3:
        return "BLUE_TOWER"
    return f"REGION_{region_id}"


def render_env_matplotlib(
    env: Any,
    *,
    field_image_path: str | Path | None = None,
    layout_json_path: str | Path | None = None,
    render_config: RenderConfig | None = None,
    fig: Any | None = None,
    axes: tuple[Any, Any] | None = None,
) -> tuple[Any, tuple[Any, Any]]:
    """
    Create a static matplotlib debug rendering of a `RebuiltMacroSim`-like object.

    Returns `(fig, (ax_field, ax_info))`.
    """
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from matplotlib.patches import Rectangle

    cfg = env.config
    state = env.state
    if state is None:
        raise RuntimeError("env.state is None; call reset() first.")

    rc = render_config or RenderConfig()

    if axes is None:
        fig, (ax_field, ax_info) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [3.2, 1.4]})
    else:
        ax_field, ax_info = axes
        fig = fig or ax_field.figure

    ax_field.clear()
    ax_info.clear()

    coords = cfg.region_coords_ft
    if coords is None:
        raise ValueError("config.region_coords_ft is required for rendering.")

    field_len = float(getattr(cfg, "field_length_ft", 54.0))
    field_wid = float(getattr(cfg, "field_width_ft", 27.0))
    fxmin = -field_len / 2.0
    fxmax = field_len / 2.0
    fymin = -field_wid / 2.0
    fymax = field_wid / 2.0
    pad = 1.5
    extent = rc.field_extent_ft or (fxmin - pad, fxmax + pad, fymin - pad, fymax + pad)
    xmin, xmax, ymin, ymax = extent

    # Background
    img_path = Path(field_image_path) if field_image_path is not None else None
    if img_path is not None and img_path.exists():
        img = mpimg.imread(str(img_path))
        ax_field.imshow(img, extent=(fxmin, fxmax, fymin, fymax), origin="upper", alpha=rc.background_alpha)
    else:
        ax_field.add_patch(Rectangle((fxmin, fymin), fxmax - fxmin, fymax - fymin, fill=False, lw=2.0, ec="0.35"))

    # Optional obstacle overlay from CycletimeHeatmap saved_layout.json
    if rc.show_layout_obstacles and layout_json_path is not None:
        layout_path = Path(layout_json_path)
        if layout_path.exists():
            layout = _load_cycleheatmap_layout(layout_path)
            if layout is not None:
                w = int(layout["w"])
                h = int(layout["h"])
                blocked = layout.get("blocked", [])
                for cell in blocked:
                    if not (isinstance(cell, (list, tuple)) and len(cell) == 2):
                        continue
                    r, c = int(cell[0]), int(cell[1])
                    x, y, dx, dy = _grid_cell_to_field_ft(r=r, c=c, w=w, h=h, field_length_ft=field_len, field_width_ft=field_wid)
                    ax_field.add_patch(
                        Rectangle(
                            (x - dx * 0.5, y - dy * 0.5),
                            dx,
                            dy,
                            fc=(0.0, 0.0, 0.0, rc.layout_obstacle_alpha),
                            ec="none",
                            zorder=2,
                        )
                    )

    ax_field.set_aspect("equal", adjustable="box")
    ax_field.set_xlim(xmin, xmax)
    ax_field.set_ylim(ymin, ymax)
    ax_field.set_xticks([])
    ax_field.set_yticks([])

    # Zone shading / HUB active indicator
    mask = int(env.active_hubs_mask())
    red_active = bool(mask & 0b01)
    blue_active = bool(mask & 0b10)

    zone_depth = float(getattr(cfg, "alliance_zone_depth_ft", 13.0))
    ax_field.add_patch(
        Rectangle((fxmin, fymin), zone_depth, fymax - fymin, fc=(1.0, 0.2, 0.2, 0.10 if red_active else 0.04), ec="none", zorder=1)
    )
    ax_field.add_patch(
        Rectangle((fxmax - zone_depth, fymin), zone_depth, fymax - fymin, fc=(0.2, 0.4, 1.0, 0.10 if blue_active else 0.04), ec="none", zorder=1)
    )

    if rc.show_center_line:
        ax_field.plot([0.0, 0.0], [fymin, fymax], color=(1.0, 1.0, 1.0, 0.35), lw=2.0, zorder=3)

    if rc.show_neutral_fuel_box:
        box_w = float(getattr(cfg, "neutral_fuel_box_width_ft", 17.0))
        box_d = float(getattr(cfg, "neutral_fuel_box_depth_ft", 6.0))
        ax_field.add_patch(
            Rectangle(
                (-box_d / 2.0, -box_w / 2.0),
                box_d,
                box_w,
                fill=False,
                lw=2.0,
                ec=(1.0, 1.0, 1.0, 0.25),
                zorder=3,
            )
        )

    # Neutral bins
    n_bins = int(cfg.n_neutral_bins)
    bin_region_ids = np.arange(2, 2 + n_bins, dtype=np.int32)
    bin_xy = coords[bin_region_ids]
    fuel = state.neutral_fuel.astype(np.float32)
    sizes = rc.fuel_marker_scale * np.sqrt(np.maximum(fuel, 0.0) + 1.0)

    edgecolors = np.full((n_bins, 4), (0.15, 0.15, 0.15, 0.85), dtype=np.float32)
    if rc.show_hub_exit_bins:
        exit_bins = set(int(x) for x in cfg.hub_exit_bin_ids)
        for i in range(n_bins):
            if i in exit_bins:
                edgecolors[i] = (1.0, 0.9, 0.2, 0.95)

    ax_field.scatter(bin_xy[:, 0], bin_xy[:, 1], s=sizes, c=fuel, cmap="YlGn", vmin=0.0, edgecolors=edgecolors, linewidths=1.5)

    if rc.show_bin_labels:
        for i in range(n_bins):
            ax_field.text(
                float(bin_xy[i, 0]),
                float(bin_xy[i, 1]) + 1.2,
                f"N{i}\n{int(fuel[i])}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="0.15",
                bbox=dict(boxstyle="round,pad=0.15", fc=(1, 1, 1, 0.55), ec="none"),
            )

    # Key regions: outposts/towers/zones
    red_outpost_id = 2 + n_bins
    blue_outpost_id = 2 + n_bins + 1
    red_tower_id = 2 + n_bins + 2
    blue_tower_id = 2 + n_bins + 3

    ax_field.scatter([coords[0, 0]], [coords[0, 1]], s=140, c="tab:red", marker="h", alpha=0.95, zorder=4)
    ax_field.scatter([coords[1, 0]], [coords[1, 1]], s=140, c="tab:blue", marker="h", alpha=0.95, zorder=4)
    ax_field.text(coords[0, 0], coords[0, 1] + 1.6, "RED HUB", ha="center", va="bottom", fontsize=9, color="tab:red")
    ax_field.text(coords[1, 0], coords[1, 1] + 1.6, "BLUE HUB", ha="center", va="bottom", fontsize=9, color="tab:blue")

    ax_field.scatter([coords[red_outpost_id, 0]], [coords[red_outpost_id, 1]], s=120, c="tab:red", marker="^", alpha=0.9)
    ax_field.scatter([coords[blue_outpost_id, 0]], [coords[blue_outpost_id, 1]], s=120, c="tab:blue", marker="^", alpha=0.9)
    ax_field.text(coords[red_outpost_id, 0], coords[red_outpost_id, 1] - 1.6, "RED OUTPOST", ha="center", va="top", fontsize=8, color="tab:red")
    ax_field.text(coords[blue_outpost_id, 0], coords[blue_outpost_id, 1] - 1.6, "BLUE OUTPOST", ha="center", va="top", fontsize=8, color="tab:blue")

    ax_field.scatter([coords[red_tower_id, 0]], [coords[red_tower_id, 1]], s=160, c="tab:red", marker="*", alpha=0.9)
    ax_field.scatter([coords[blue_tower_id, 0]], [coords[blue_tower_id, 1]], s=160, c="tab:blue", marker="*", alpha=0.9)
    ax_field.text(coords[red_tower_id, 0], coords[red_tower_id, 1] + 1.6, "RED TOWER", ha="center", va="bottom", fontsize=8, color="tab:red")
    ax_field.text(coords[blue_tower_id, 0], coords[blue_tower_id, 1] + 1.6, "BLUE TOWER", ha="center", va="bottom", fontsize=8, color="tab:blue")

    # Robots
    robot_xy = coords[state.robot_region.astype(np.int32)]
    colors = ["tab:red"] * 3 + ["tab:blue"] * 3
    ax_field.scatter(robot_xy[:, 0], robot_xy[:, 1], s=rc.robot_marker_size, c=colors, marker="o", edgecolors="white", linewidths=1.5, zorder=5)

    if rc.show_task_lines:
        for rid in range(6):
            task_action_id = int(state.robot_task_action_id[rid])
            if task_action_id < 0:
                continue
            if int(state.robot_busy_until[rid]) <= int(state.t):
                continue
            tx = coords[int(state.robot_task_target_region[rid]), 0]
            ty = coords[int(state.robot_task_target_region[rid]), 1]
            ax_field.plot([robot_xy[rid, 0], tx], [robot_xy[rid, 1], ty], color=colors[rid], alpha=0.55, lw=2.0, zorder=4)

    if rc.show_robot_labels:
        for rid in range(6):
            alliance = "R" if rid < 3 else "B"
            carried = int(state.robot_carried[rid])
            climbed = int(state.robot_climbed_level[rid])
            busy = max(0, int(state.robot_busy_until[rid]) - int(state.t))
            action_id = int(state.robot_task_action_id[rid])
            region_name = _region_name(int(state.robot_region[rid]), n_neutral_bins=n_bins)
            if action_id >= 0 and int(state.robot_busy_until[rid]) > int(state.t):
                kind = ActionKind(int(env._action_kind[action_id])).name
                reserved = int(state.robot_task_reserved_fuel[rid])
                label = f"{alliance}{rid%3} {region_name} c={carried} rsv={reserved} L{climbed} {kind} {busy}s"
            else:
                label = f"{alliance}{rid%3} {region_name} c={carried} L{climbed}"
            ax_field.text(
                float(robot_xy[rid, 0]),
                float(robot_xy[rid, 1]) - 1.5,
                label,
                ha="center",
                va="top",
                fontsize=8,
                color="0.1",
                bbox=dict(boxstyle="round,pad=0.2", fc=(1, 1, 1, 0.6), ec="none"),
                zorder=6,
            )

    # Info panel
    ax_info.axis("off")
    phase_name = env.phase_at(int(state.t)).name
    hubs_str = ("RED" if red_active else "") + (" & " if (red_active and blue_active) else "") + ("BLUE" if blue_active else "")
    if red_active and blue_active:
        hubs_str = "BOTH"
    elif red_active:
        hubs_str = "RED"
    elif blue_active:
        hubs_str = "BLUE"
    else:
        hubs_str = "NONE"

    lines: list[str] = []
    lines.append(f"t = {int(state.t)}s  (remaining {env.total_match_s() - int(state.t)}s)")
    lines.append(f"phase = {phase_name}")
    lines.append(f"active HUB(s) = {hubs_str}")
    lines.append("")
    lines.append(f"score:  RED {int(state.score[0])}   BLUE {int(state.score[1])}")
    lines.append(f"penalty: RED {int(state.penalty_points[0])}   BLUE {int(state.penalty_points[1])}")
    lines.append(f"total:  RED {int(state.score[0] + state.penalty_points[0])}   BLUE {int(state.score[1] + state.penalty_points[1])}")
    lines.append("")
    lines.append(f"neutral fuel total: {int(np.sum(state.neutral_fuel))}")
    lines.append(f"depot fuel:  RED {int(state.depot_fuel[0])}   BLUE {int(state.depot_fuel[1])}")
    lines.append(f"outpost chute: RED {int(state.outpost_chute[0])}   BLUE {int(state.outpost_chute[1])} (cap {int(cfg.outpost_chute_capacity)})")
    lines.append(f"outpost corral: RED {int(state.outpost_corral[0])}   BLUE {int(state.outpost_corral[1])}")
    lines.append("")
    lines.append("robots:")
    for rid in range(6):
        alliance = "RED" if rid < 3 else "BLUE"
        carried = int(state.robot_carried[rid])
        region = int(state.robot_region[rid])
        region_name = _region_name(region, n_neutral_bins=n_bins)
        busy = max(0, int(state.robot_busy_until[rid]) - int(state.t))
        task = int(state.robot_task_action_id[rid])
        climbed = int(state.robot_climbed_level[rid])
        if task >= 0 and int(state.robot_busy_until[rid]) > int(state.t):
            task_name = ActionKind(int(env._action_kind[task])).name
            reserved = int(state.robot_task_reserved_fuel[rid])
            lines.append(
                f"  {alliance}[{rid%3}] region={region_name}({region}) carried={carried} rsv={reserved} climb=L{climbed} busy={busy}s task={task_name}"
            )
        else:
            lines.append(f"  {alliance}[{rid%3}] region={region_name}({region}) carried={carried} climb=L{climbed} busy={busy}s task=IDLE")

    ax_info.text(0.02, 0.98, "\n".join(lines), ha="left", va="top", fontsize=10, family="monospace")

    fig.tight_layout()
    return fig, (ax_field, ax_info)
