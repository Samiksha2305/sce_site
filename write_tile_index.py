#!/usr/bin/env python3
"""Rebuild atlas/index.html as a clickable SoCal tile map. No MESMA rebuild.

  python write_tile_index.py /path/to/atlas
"""
from __future__ import annotations

import sys
from pathlib import Path

from build_atlas_cyverse import find_new_data, write_atlas_index


def main() -> None:
    atlas = Path(sys.argv[1] if len(sys.argv) > 1 else "atlas")
    if not atlas.exists():
        sys.exit(f"Atlas folder not found: {atlas}")
    folders = [
        p.name
        for p in sorted(atlas.iterdir())
        if p.is_dir() and (p / "mesma_grid.js").exists()
    ]
    mesma = find_new_data() / "2026-07-28" / "mesma"
    write_atlas_index(atlas, folders, mesma_dir=mesma if mesma.exists() else None)


if __name__ == "__main__":
    main()
