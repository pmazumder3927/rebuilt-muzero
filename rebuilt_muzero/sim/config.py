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

    # Initial fuel (placeholders; tune to the official game once known)
    initial_neutral_fuel_per_bin: int = 30
    initial_depot_fuel: int = 60
    initial_outpost_chute_fuel: int = 0

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
    region_distance_ft: np.ndarray | None = None
    drive_overhead_s: float = 0.5

    # Human-player processing: move fuel from corral -> chute each second (optional).
    outpost_fill_fuel_per_s: int = 5

    # Macro action overheads (seconds)
    collect_overhead_s: float = 0.4
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
    # 0 red zone, 1 blue zone, 2.. neutral bins, then red outpost, blue outpost, red tower, blue tower
    coords: list[tuple[float, float]] = []
    coords.append((-22.0, 0.0))  # red zone
    coords.append((22.0, 0.0))  # blue zone

    # Neutral bins spread across the midfield.
    for i in range(n_neutral_bins):
        x = -8.0 + 16.0 * (i / max(1, n_neutral_bins - 1))
        y = 6.0 if i % 2 == 0 else -6.0
        coords.append((x, y))

    coords.append((-24.0, -12.0))  # red outpost
    coords.append((24.0, -12.0))  # blue outpost
    coords.append((-24.0, 12.0))  # red tower
    coords.append((24.0, 12.0))  # blue tower
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
    return GameConfig(
        n_neutral_bins=n_neutral_bins,
        region_distance_ft=region_distance_ft,
        missed_shot_bin_id_by_alliance=(0, max(0, n_neutral_bins - 1)),
    )
