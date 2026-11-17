# SCE postfire landscape atlas

CyVerse notebooks and companion files for the Southern California MESMA postfire atlas.

## Run order on CyVerse

1. `cyverse_pipeline_2026_07_28_all.ipynb` (`1_…` on CyVerse) — Sentinel-2 stacks. Skip if those images already exist.
2. `tsf_by_year_2015_2020.ipynb` (`3_…` on CyVerse) — fires + time-since-fire. Skip if `fires.geojson` already exists.
3. `mesma_fire_grids_by_year_3.ipynb` — fire list, grid/perimeter map, then MESMA. Start with `YEARS_TO_RUN = [2015]`; after that year finishes, change to `[2016]`, then 2017…2020.
4. `build_atlas_cyverse.ipynb` — rebuild the website after MESMA years are done.

Skip `cyverse_pipeline_2026_07_28.ipynb` (`2_…`) unless you need the older single-scene path. Skip older `mesma_fire_grids_by_year.ipynb` / `_2` copies.

## Atlas preview

The pipeline notebooks write `atlas/` with `index.html` and one folder per Sentinel-2 tile. Preview with `python3 -m http.server` and JupyterHub `/proxy/<port>/`.

On a scene page: **Time since fire → Fires 2015–2026** (regional map + grid toggle) or **fire_grids** (this tile).

## Site files

- `build_atlas_cyverse.py` — builds the clickable tile picker and scene pages
- `postfire_landscape_atlas1.html` — scene template (MESMA overlay, fire_grids, TSF tabs)
- `export_time_since_fire.py` — days-since-last-burn helper
- `export_tsf_by_year.py` — TSF at ignition by fire year
- `write_tile_index.py` — tile index writer

Do not commit `atlas/` or the raw cubes. For Cloudflare Pages, upload the built `atlas/` contents (so `index.html` is at the site root).
