from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


REPO_URL = "https://github.com/Bjorn4481/CycletimeHeatmap.git"


def _ensure_repo(root: Path, *, force: bool) -> Path:
    repo_dir = root / ".tmp" / "CycletimeHeatmap"
    repo_dir.parent.mkdir(parents=True, exist_ok=True)

    if repo_dir.exists() and force:
        shutil.rmtree(repo_dir)

    if repo_dir.exists():
        return repo_dir

    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(repo_dir)], check=True)
    return repo_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch field/debug assets from CycletimeHeatmap.")
    parser.add_argument("--dest", type=Path, default=Path("assets"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    repo_dir = _ensure_repo(root, force=args.force)

    field_png = args.dest / "field.png"
    layout_json = args.dest / "saved_layout.json"

    args.dest.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(repo_dir / "field.png", field_png)
    shutil.copyfile(repo_dir / "saved_layout.json", layout_json)

    print(f"wrote: {field_png}")
    print(f"wrote: {layout_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
