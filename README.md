# SCE postfire landscape atlas

CyVerse notebooks and companion files for the Southern California MESMA postfire atlas.

## Run on CyVerse

1. Upload this repo plus the MESMA cubes into `/home/jovyan/data-store`.
2. Open `cyverse_pipeline_2026_07_28.ipynb` (or `_all`) and run all cells.
3. The notebook writes `atlas/` with `index.html` and one folder per Sentinel-2 tile.
4. Preview with `python3 -m http.server` and JupyterHub `/proxy/<port>/`.

## Site files

- `build_atlas_cyverse.py` — builds the clickable tile picker and scene pages
- `postfire_landscape_atlas1.html` — scene template (MESMA overlay + Esri basemap)
- `export_time_since_fire.py` — time-since-fire PNG helper
- `write_tile_index.py` — tile index writer

Do not commit `atlas/` or the raw cubes. For Cloudflare Pages, upload the built `atlas/` contents (so `index.html` is at the site root).
