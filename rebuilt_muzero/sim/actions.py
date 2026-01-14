from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class ActionKind(IntEnum):
    COLLECT_NEUTRAL = 0
    COLLECT_DEPOT = 1
    SCORE_HUB = 2
    DELIVER_OUTPOST = 3
    DEFEND_OPPONENT_HUB_LANE = 4
    DEFEND_OPPONENT_COLLECTOR = 5
    PREP_CLIMB = 6
    CLIMB = 7
    IDLE = 8


@dataclass(frozen=True, slots=True)
class DecodedAction:
    kind: ActionKind
    arg: int = 0  # bin_id or climb_level (depending on kind)


def action_space_size(*, n_neutral_bins: int, climb_levels: tuple[int, ...] = (1, 2, 3)) -> int:
    # [collect_neutral per bin] + [collect_depot, score_hub, deliver_outpost, defend_lane, defend_collector]
    # + [prep_climb per level] + [climb per level]
    return n_neutral_bins + 5 + 2 * len(climb_levels) + 1


def decode_action(
    action_id: int, *, n_neutral_bins: int, climb_levels: tuple[int, ...] = (1, 2, 3)
) -> DecodedAction:
    if action_id < 0:
        raise ValueError(f"action_id must be >= 0, got {action_id}")

    if action_id < n_neutral_bins:
        return DecodedAction(ActionKind.COLLECT_NEUTRAL, action_id)

    base = n_neutral_bins
    if action_id == base:
        return DecodedAction(ActionKind.COLLECT_DEPOT)
    if action_id == base + 1:
        return DecodedAction(ActionKind.SCORE_HUB)
    if action_id == base + 2:
        return DecodedAction(ActionKind.DELIVER_OUTPOST)
    if action_id == base + 3:
        return DecodedAction(ActionKind.DEFEND_OPPONENT_HUB_LANE)
    if action_id == base + 4:
        return DecodedAction(ActionKind.DEFEND_OPPONENT_COLLECTOR)

    remaining = action_id - (base + 5)
    if remaining < len(climb_levels):
        return DecodedAction(ActionKind.PREP_CLIMB, climb_levels[remaining])

    remaining -= len(climb_levels)
    if remaining < len(climb_levels):
        return DecodedAction(ActionKind.CLIMB, climb_levels[remaining])

    if remaining == len(climb_levels):
        return DecodedAction(ActionKind.IDLE, 0)

    raise ValueError(
        f"action_id {action_id} is out of range for n_neutral_bins={n_neutral_bins}, climb_levels={climb_levels}"
    )
