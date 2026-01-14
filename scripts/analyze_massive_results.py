from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _percentiles(x: np.ndarray, ps: list[float]) -> list[float]:
    return [float(np.quantile(x, p)) for p in ps]


def main() -> int:
    parser = argparse.ArgumentParser(description="Quick stats for massive_results.json")
    parser.add_argument("--path", type=Path, default=Path("massive_results.json"))
    args = parser.parse_args()

    data = json.loads(args.path.read_text())
    if not isinstance(data, list) or (data and not isinstance(data[0], dict)):
        raise SystemExit("Expected a JSON list[dict].")

    print(f"path: {args.path}")
    print(f"rows: {len(data)}")
    keys = sorted(set().union(*(d.keys() for d in data)))
    print(f"keys ({len(keys)}): {keys}")

    numeric_fields = [
        "capacity",
        "shooting_rate",
        "intake_rate",
        "accuracy",
        "max_speed",
        "acceleration",
        "align_time",
        "dump_time",
        "cycle_variance",
        "defense_penalty",
        "fuel",
        "blue_pts",
        "red_pts",
    ]

    print("\n**Numeric fields** (min / p50 / p90 / max)")
    for k in numeric_fields:
        vals = [d.get(k) for d in data]
        if any(v is None for v in vals):
            print(f"{k}: missing={sum(v is None for v in vals)}")
            continue
        arr = np.asarray(vals, dtype=np.float64)
        p50, p90 = _percentiles(arr, [0.5, 0.9])
        print(f"{k:>16}: {arr.min():>7.3f}  {p50:>7.3f}  {p90:>7.3f}  {arr.max():>7.3f}")

    print("\n**Category counts**")
    cats = Counter(d.get("category", "<?>") for d in data)
    for name, count in cats.most_common():
        print(f"{name:>20}: {count}")

    print("\n**Fuel by turret_type** (mean / p50 / p90)")
    by_turret: dict[str, list[float]] = defaultdict(list)
    for d in data:
        by_turret[str(d.get("turret_type", "<?>"))].append(float(d["fuel"]))
    for turret, vals in sorted(by_turret.items()):
        arr = np.asarray(vals, dtype=np.float64)
        p50, p90 = _percentiles(arr, [0.5, 0.9])
        print(f"{turret:>12}: {arr.mean():>7.2f}  {p50:>7.2f}  {p90:>7.2f}")

    for flag in ("shoot_on_move", "shoot_while_intake"):
        print(f"\n**Fuel by {flag}** (mean / p50 / p90)")
        by_flag: dict[str, list[float]] = defaultdict(list)
        for d in data:
            by_flag[str(bool(d.get(flag)))].append(float(d["fuel"]))
        for v, vals in sorted(by_flag.items()):
            arr = np.asarray(vals, dtype=np.float64)
            p50, p90 = _percentiles(arr, [0.5, 0.9])
            print(f"{v:>5}: {arr.mean():>7.2f}  {p50:>7.2f}  {p90:>7.2f}")

    # Median spec as a concrete calibration anchor
    # (closest row to medians over a subset of key fields).
    subset = ["capacity", "shooting_rate", "intake_rate", "accuracy", "max_speed", "acceleration", "align_time", "dump_time"]
    med = {k: float(np.median([d[k] for d in data])) for k in subset}
    best_i = None
    best_dist = float("inf")
    for i, d in enumerate(data):
        dist = 0.0
        for k in subset:
            dist += (float(d[k]) - med[k]) ** 2
        if dist < best_dist:
            best_dist = dist
            best_i = i
    assert best_i is not None
    print("\n**Median-ish row** (useful default anchor)")
    print(json.dumps(data[best_i], indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

