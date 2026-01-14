from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class RobotSpec:
    # Core throughput
    fuel_capacity: int
    intake_fuel_per_s: float
    shoot_fuel_per_s: float
    shoot_accuracy: float

    # Drive (in field units; the calibration dataset uses ft/s and ft/s^2)
    max_speed: float
    acceleration: float

    # Shooting overheads (seconds)
    align_time_s: float
    dump_time_s: float

    # Feature flags (from `massive_results.json`)
    shoot_on_move: bool
    shoot_while_intake: bool

    # Stochasticity / defense sensitivity (coarse)
    cycle_variance_s: float
    defense_penalty: float

    # Endgame
    max_climb_level: int
    climb_time_s_by_level: Mapping[int, int]


@dataclass(frozen=True, slots=True)
class GameConfig:
    # Official field dimensions (feet). Used for rendering/layout defaults.
    field_length_ft: float = 651.2 / 12.0
    field_width_ft: float = 317.7 / 12.0

    # Official zone/layout dimensions (feet). Used for default region coordinates and renderer overlays.
    alliance_zone_depth_ft: float = 158.6 / 12.0
    neutral_fuel_box_width_ft: float = 206.0 / 12.0
    neutral_fuel_box_depth_ft: float = 72.0 / 12.0
    hub_distance_from_alliance_wall_ft: float = 158.6 / 12.0

    # Time (seconds)
    auto_s: int = 20
    transition_s: int = 10
    shift_s: int = 25
    n_shifts: int = 4
    endgame_s: int = 30
    decision_interval_s: int = 1
    hub_mode: str = "rebuilt"  # "rebuilt" | "always_on"

    # Fuel bins
    n_neutral_bins: int = 8
    outpost_chute_capacity: int = 25

    # Initial fuel (official staging defaults with no preloads: 504 total fuel)
    initial_neutral_fuel_per_bin: int = 50
    initial_depot_fuel: int = 24
    initial_outpost_chute_fuel: int = 24

    # Scoring (placeholders; tune later)
    fuel_point_value: int = 1
    tower_points_auto_by_level: tuple[int, int, int] = (2, 5, 10)
    tower_points_teleop_by_level: tuple[int, int, int] = (1, 3, 6)

    # Fouls / penalties (placeholders; tune later)
    major_foul_points: int = 10
    minor_foul_points: int = 3
    pin_limit_s: int = 3

    # HUB redistribution: 4 exits -> neutral bins.
    # Default: uniform exits and a simple mapping to 4 bins.
    # For bins >4, the remaining bins get 0 probability until configured.
    hub_exit_bin_ids: tuple[int, int, int, int] = (0, 1, 2, 3)
    hub_exit_probs: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)
    missed_shot_bin_id_by_alliance: tuple[int, int] = (0, 0)

    # Region distance matrix (feet). Shape: (n_regions, n_regions).
    # Regions are enumerated in `rebuilt_muzero.sim.state`.
    region_coords_ft: np.ndarray | None = None
    region_distance_ft: np.ndarray | None = None
    drive_overhead_s: float = 1.0

    # Human-player processing: move fuel from corral -> chute each second (optional).
    outpost_fill_fuel_per_s: int = 5

    # Macro action overheads (seconds)
    collect_overhead_s: float = 0.6
    deliver_overhead_s: float = 0.6

    # Defense model (very coarse)
    defend_duration_s: int = 1

    # Endgame tower protection foul model (coarse)
    tower_contact_foul_prob: float = 0.2
    tower_contact_awards_level3: bool = True

    def total_match_s(self) -> int:
        return self.auto_s + self.transition_s + self.n_shifts * self.shift_s + self.endgame_s


def _default_region_coords(n_neutral_bins: int) -> np.ndarray:
    """
    Coarse region coordinates (feet) used to build a default distance matrix.
    This is NOT official geometry; it's a consistent distance prior for the macro-sim.
    """
    # Regions:
    # 0 red "zone" (modeled at the red HUB), 1 blue "zone" (blue HUB),
    # 2.. neutral bins, then red outpost, blue outpost, red tower, blue tower
    coords: list[tuple[float, float]] = []

    # Use the official FIELD size as a consistent reference frame.
    field_length_ft = 651.2 / 12.0
    field_width_ft = 317.7 / 12.0
    half_len = field_length_ft / 2.0
    half_wid = field_width_ft / 2.0

    # HUB centers are located 158.6in from the ALLIANCE WALL and centered between BUMPS.
    hub_from_wall_ft = 158.6 / 12.0
    x_red_hub = -half_len + hub_from_wall_ft
    x_blue_hub = half_len - hub_from_wall_ft
    coords.append((x_red_hub, 0.0))  # red HUB ("zone" proxy)
    coords.append((x_blue_hub, 0.0))  # blue HUB ("zone" proxy)

    # Neutral fuel staging is roughly within a central box (206in wide × 72in deep).
    box_w_ft = 206.0 / 12.0
    box_d_ft = 72.0 / 12.0
    cols = int(np.ceil(float(n_neutral_bins) / 2.0))
    if cols < 1:
        cols = 1
    x_positions = np.linspace(-box_d_ft / 2.0 + box_d_ft / (2.0 * cols), box_d_ft / 2.0 - box_d_ft / (2.0 * cols), cols)
    y_top = box_w_ft / 2.0
    y_bottom = -box_w_ft / 2.0
    for i in range(n_neutral_bins):
        col = i // 2
        y = y_top if (i % 2 == 0) else y_bottom
        x = float(x_positions[min(col, cols - 1)])
        coords.append((x, y))

    # OUTPOST: one per alliance, located at the (south) end of each ALLIANCE WALL.
    # Use OUTPOST AREA width (71in) as a proxy to place the marker.
    outpost_w_ft = 71.0 / 12.0
    y_outpost = -half_wid + outpost_w_ft / 2.0

    # TOWER: integrated into ALLIANCE WALL between DRIVER STATION 2 and 3.
    # Use the stated shelf width (69in) as a rough driver station width for layout placement.
    ds_w_ft = 69.0 / 12.0
    tower_w_ft = 49.25 / 12.0
    y_tower = -half_wid + outpost_w_ft + 2.0 * ds_w_ft + tower_w_ft / 2.0

    coords.append((-half_len, y_outpost))  # red outpost
    coords.append((half_len, y_outpost))  # blue outpost
    coords.append((-half_len, y_tower))  # red tower
    coords.append((half_len, y_tower))  # blue tower
    return np.asarray(coords, dtype=np.float32)


def _coords_to_distance_ft(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    return dist.astype(np.float32)


def default_robot_specs() -> tuple[RobotSpec, ...]:
    # Simple archetypes seeded from `massive_results.json` medians.
    # Indexing convention (per alliance): [cycler, defender, climber]
    cycler = RobotSpec(
        fuel_capacity=70,
        intake_fuel_per_s=11.0,
        shoot_fuel_per_s=25.0,
        shoot_accuracy=0.92,
        max_speed=17.0,
        acceleration=26.0,
        align_time_s=0.5,
        dump_time_s=0.3,
        shoot_on_move=True,
        shoot_while_intake=True,
        cycle_variance_s=1.0,
        defense_penalty=0.15,
        max_climb_level=2,
        climb_time_s_by_level={1: 6, 2: 12, 3: 30},
    )
    defender = RobotSpec(
        fuel_capacity=35,
        intake_fuel_per_s=7.0,
        shoot_fuel_per_s=10.0,
        shoot_accuracy=0.88,
        max_speed=16.0,
        acceleration=28.0,
        align_time_s=0.5,
        dump_time_s=0.3,
        shoot_on_move=False,
        shoot_while_intake=False,
        cycle_variance_s=1.0,
        defense_penalty=0.10,
        max_climb_level=1,
        climb_time_s_by_level={1: 8, 2: 20, 3: 35},
    )
    climber = RobotSpec(
        fuel_capacity=40,
        intake_fuel_per_s=6.0,
        shoot_fuel_per_s=12.0,
        shoot_accuracy=0.90,
        max_speed=14.5,
        acceleration=22.0,
        align_time_s=0.5,
        dump_time_s=0.3,
        shoot_on_move=False,
        shoot_while_intake=False,
        cycle_variance_s=1.0,
        defense_penalty=0.15,
        max_climb_level=3,
        climb_time_s_by_level={1: 6, 2: 10, 3: 16},
    )
    return (cycler, defender, climber, cycler, defender, climber)


def default_config(*, n_neutral_bins: int = 8) -> GameConfig:
    coords = _default_region_coords(n_neutral_bins)
    region_distance_ft = _coords_to_distance_ft(coords)

    # Official staging with no preloaded robots: 24 in each DEPOT + 24 in each OUTPOST CHUTE, rest in NEUTRAL.
    total_fuel = 504
    depot_each = 24
    chute_each = 24
    neutral_total = total_fuel - 2 * depot_each - 2 * chute_each
    neutral_per_bin = int(np.floor(float(neutral_total) / max(1, int(n_neutral_bins))))
    return GameConfig(
        n_neutral_bins=n_neutral_bins,
        region_coords_ft=coords,
        region_distance_ft=region_distance_ft,
        initial_neutral_fuel_per_bin=neutral_per_bin,
        initial_depot_fuel=depot_each,
        initial_outpost_chute_fuel=chute_each,
        hub_exit_bin_ids=(
            0,
            max(0, n_neutral_bins // 3),
            max(0, (2 * n_neutral_bins) // 3),
            max(0, n_neutral_bins - 1),
        ),
        missed_shot_bin_id_by_alliance=(0, max(0, n_neutral_bins - 1)),
    )
