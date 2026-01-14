from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class Alliance(IntEnum):
    RED = 0
    BLUE = 1


class Phase(IntEnum):
    AUTO = 0
    TRANSITION = 1
    SHIFT1 = 2
    SHIFT2 = 3
    SHIFT3 = 4
    SHIFT4 = 5
    ENDGAME = 6


def n_regions(n_neutral_bins: int) -> int:
    # 0 red zone, 1 blue zone, 2..neutral bins, then red outpost, blue outpost, red tower, blue tower
    return 2 + n_neutral_bins + 4


def red_zone_region() -> int:
    return 0


def blue_zone_region() -> int:
    return 1


def neutral_bin_region(bin_id: int) -> int:
    return 2 + bin_id


def red_outpost_region(n_neutral_bins: int) -> int:
    return 2 + n_neutral_bins


def blue_outpost_region(n_neutral_bins: int) -> int:
    return 2 + n_neutral_bins + 1


def red_tower_region(n_neutral_bins: int) -> int:
    return 2 + n_neutral_bins + 2


def blue_tower_region(n_neutral_bins: int) -> int:
    return 2 + n_neutral_bins + 3


def is_in_alliance_zone(region_id: int, alliance: int, n_neutral_bins: int) -> bool:
    if alliance == Alliance.RED:
        return region_id in {red_zone_region(), red_outpost_region(n_neutral_bins), red_tower_region(n_neutral_bins)}
    return region_id in {blue_zone_region(), blue_outpost_region(n_neutral_bins), blue_tower_region(n_neutral_bins)}


@dataclass(slots=True)
class SimState:
    # Clock (seconds elapsed since match start)
    t: int
    first_shift_active_alliance: int  # set after AUTO

    # Score + penalties (stored as points already)
    score: np.ndarray  # shape (2,), int32
    penalty_points: np.ndarray  # shape (2,), int32

    # Auto fuel scored (for shift order)
    auto_fuel_scored: np.ndarray  # shape (2,), int32

    # Fuel storage
    neutral_fuel: np.ndarray  # shape (n_neutral_bins,), int32
    depot_fuel: np.ndarray  # shape (2,), int32
    outpost_chute: np.ndarray  # shape (2,), int32
    outpost_corral: np.ndarray  # shape (2,), int32

    # Robots (6)
    robot_region: np.ndarray  # shape (6,), int16
    robot_carried: np.ndarray  # shape (6,), int16
    robot_busy_until: np.ndarray  # shape (6,), int16 (absolute time seconds)
    robot_task_action_id: np.ndarray  # shape (6,), int16 (-1 if none)
    robot_task_target_region: np.ndarray  # shape (6,), int16
    robot_task_reserved_fuel: np.ndarray  # shape (6,), int16
    robot_climbed_level: np.ndarray  # shape (6,), int8
    robot_pin_time: np.ndarray  # shape (6,), int8
