#!/usr/bin/env python3
"""Build a small postfire atlas from a huge MESMA fractions cube.

Run this on CyVerse VICE (Jupyter Lab) so the 6+ GB rasters stay on the
data store. Only the atlas (HTML + JS + PNG, a few MB) is written out.

Default paths match:
  /iplant/home/samiksha23/SCE-CU/sentinel2_data/new_data/2026-07-28/mesma/11_S_LA_2026_7_28/
  /iplant/home/samiksha23/SCE-CU/sentinel2_data/new_data/Time_since_fire/
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def find_home() -> Path:
    for p in (
        Path("/data-store/iplant/home/samiksha23"),
        Path("/home/jovyan/data-store/iplant/home/samiksha23"),
        Path("/home/jovyan/data-store"),
        Path("/data-store"),
        Path("/iplant/home/samiksha23"),
    ):
        if p.exists():
            return p
    return Path("/data-store/iplant/home/samiksha23")


def find_new_data(home: Path | None = None) -> Path:
    home = home or find_home()
    candidates = [
        home / "SCE-CU/sentinel2_data/new_data",
        home / "sentinel2_data/new_data",
        Path.cwd() / "new_data",
        Path.cwd(),
    ]
    # Walk up from cwd in case the notebook was opened inside a tile folder
    here = Path.cwd().resolve()
    for parent in [here, *here.parents]:
        if parent.name == "new_data":
            candidates.append(parent)
        cand = parent / "new_data"
        candidates.append(cand)
        mesma = parent / "mesma"
        if parent.name == "2026-07-28" and mesma.exists():
            candidates.append(parent.parent)
    for p in candidates:
        if (p / "2026-07-28" / "mesma").exists():
            return p
        if p.name == "new_data" and (p / "2026-07-28" / "mesma").exists():
            return p
    return home / "SCE-CU/sentinel2_data/new_data"


def parse_hdr(fractions: Path) -> dict:
    hdr = fractions.parent / (fractions.name + ".hdr")
    if not hdr.exists():
        hdr = fractions.with_suffix(".hdr")
    text = hdr.read_text(errors="ignore")
    meta = {
        "samples": None,
        "lines": None,
        "bands": None,
        "dtype": 5,
        "origin_x": 199980.0,
        "origin_y": 4000020.0,
        "res": 10.0,
        "zone": 11,
    }
    for line in text.splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip().lower(), v.strip()
        if k in ("samples", "lines", "bands"):
            meta[k] = int(v)
        elif k == "data type":
            meta["dtype"] = int(v)
        elif k == "map info":
            inner = v.strip().strip("{}")
            parts = [p.strip() for p in inner.split(",")]
            try:
                meta["origin_x"] = float(parts[3])
                meta["origin_y"] = float(parts[4])
                meta["res"] = float(parts[5])
                meta["zone"] = int(parts[7])
            except Exception:
                pass
    np_dtype = {4: "<f4", 5: "<f8"}.get(meta["dtype"], "<f8")
    meta["np_dtype"] = np_dtype
    return meta


def memmap_envi(path: Path, meta: dict) -> np.memmap:
    shape = (meta["bands"], meta["lines"], meta["samples"])
    expected = int(np.prod(shape) * np.dtype(meta["np_dtype"]).itemsize)
    size = path.stat().st_size
    if size != expected:
        print(f"Warning: size {size} != expected {expected}", flush=True)
    return np.memmap(path, dtype=meta["np_dtype"], mode="r", shape=shape)


def valid_bbox(mm: np.memmap, shade_band: int = 6, chunk: int = 200):
    n = mm.shape[1]
    min_r, max_r, min_c, max_c = n, -1, n, -1
    n_valid = 0
    for r0 in range(0, n, chunk):
        r1 = min(n, r0 + chunk)
        shade = np.array(mm[shade_band, r0:r1, :])
        mask = shade > 1e-6
        if not mask.any():
            continue
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        min_r = min(min_r, r0 + int(rows[0]))
        max_r = max(max_r, r0 + int(rows[-1]))
        min_c = min(min_c, int(cols[0]))
        max_c = max(max_c, int(cols[-1]))
        n_valid += int(mask.sum())
    if max_r < 0:
        raise SystemExit("No modeled MESMA pixels found (shade band is empty).")
    return min_r, max_r + 1, min_c, max_c + 1, n_valid


def aggregate(crop: np.ndarray, valid: np.ndarray, out_rows: int, out_cols: int):
    names = ["none", "npv", "pv", "rock", "soil", "water", "shade"]
    H, W = crop.shape[1], crop.shape[2]
    frac = np.clip(crop, 0, 1)
    unshaded = np.clip(1.0 - frac[6], 1e-6, 1)
    norm = frac.copy()
    for i in range(6):
        norm[i] = np.where(valid, frac[i] / unshaded, 0)

    agg = {k: np.full((out_rows, out_cols), np.nan, np.float32) for k in names}
    agg.update(
        {
            "pv_n": np.full((out_rows, out_cols), np.nan, np.float32),
            "npv_n": np.full((out_rows, out_cols), np.nan, np.float32),
            "rock_n": np.full((out_rows, out_cols), np.nan, np.float32),
            "soil_n": np.full((out_rows, out_cols), np.nan, np.float32),
            "water_n": np.full((out_rows, out_cols), np.nan, np.float32),
            "none_n": np.full((out_rows, out_cols), np.nan, np.float32),
        }
    )
    row_edges = np.linspace(0, H, out_rows + 1).astype(int)
    col_edges = np.linspace(0, W, out_cols + 1).astype(int)
    valid_cells = np.zeros((out_rows, out_cols), bool)
    for i in range(out_rows):
        rr0, rr1 = row_edges[i], max(row_edges[i + 1], row_edges[i] + 1)
        for j in range(out_cols):
            cc0, cc1 = col_edges[j], max(col_edges[j + 1], col_edges[j] + 1)
            m = valid[rr0:rr1, cc0:cc1]
            if not m.any():
                continue
            valid_cells[i, j] = True
            sl = frac[:, rr0:rr1, cc0:cc1]
            nl = norm[:, rr0:rr1, cc0:cc1]
            for bi, k in enumerate(names):
                agg[k][i, j] = sl[bi][m].mean()
            for bi, k in enumerate(["none_n", "npv_n", "pv_n", "rock_n", "soil_n", "water_n"]):
                agg[k][i, j] = nl[bi][m].mean()
    return agg, valid_cells


def pack(a: np.ndarray, valid_cells: np.ndarray, scale: int = 1000):
    out = []
    rows, cols = a.shape
    for i in range(rows):
        for j in range(cols):
            if not valid_cells[i, j] or not np.isfinite(a[i, j]):
                out.append(-1)
            else:
                out.append(int(round(float(np.clip(a[i, j], 0, 1) * scale))))
    return out


def pack_days(a: np.ndarray, valid_cells: np.ndarray):
    out = []
    rows, cols = a.shape
    for i in range(rows):
        for j in range(cols):
            if not valid_cells[i, j] or not np.isfinite(a[i, j]) or a[i, j] < 0:
                out.append(-1)
            else:
                out.append(int(round(float(a[i, j]))))
    return out


def utm_to_ll(x: float, y: float, zone: int = 11):
    try:
        p = subprocess.run(
            ["gdaltransform", "-s_srs", f"EPSG:326{zone:02d}", "-t_srs", "EPSG:4326"],
            input=f"{x} {y}\n".encode(),
            capture_output=True,
            check=False,
        )
        parts = p.stdout.decode().strip().split()
        return float(parts[0]), float(parts[1])
    except Exception:
        lon = (x - 500000) / (0.9996 * 6378137) * (180 / math.pi) + (-177 + zone * 6)
        lat = y / 111320.0
        return lon, lat


def save_png(path: Path, rgb: np.ndarray):
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    try:
        from PIL import Image

        Image.fromarray(rgb, mode="RGB").save(path)
        return
    except Exception:
        pass
    try:
        import imageio.v2 as imageio

        imageio.imwrite(path, rgb)
        return
    except Exception:
        pass
    # PPM fallback (always works)
    ppm = path.with_suffix(".ppm")
    h, w, _ = rgb.shape
    with open(ppm, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode())
        f.write(rgb.tobytes())
    try:
        subprocess.run(["gdal_translate", "-q", "-of", "PNG", str(ppm), str(path)], check=False)
        ppm.unlink(missing_ok=True)
        Path(str(path) + ".aux.xml").unlink(missing_ok=True)
    except Exception:
        print(f"Wrote {ppm} (install pillow to get PNG)", flush=True)


def find_tsf_raster(tsf_dir: Path) -> Path | None:
    if not tsf_dir.exists():
        return None
    hits = []
    for pat in ("*.tif", "*.tiff", "*.img", "*.bsq"):
        hits.extend(tsf_dir.rglob(pat))
    hits = [p for p in hits if p.is_file() and "html" not in p.name.lower()]
    if not hits:
        return None
    hits.sort(key=lambda p: p.stat().st_size, reverse=True)
    return hits[0]


def tile_label(folder_or_file: str) -> tuple[str, str]:
    """11_S_KR_2026_7_28 -> ('11SKR', '2026-07-28')."""
    name = Path(folder_or_file).name
    tile, date = name, "2026-07-28"
    if "_S_" in name:
        parts = name.split("_")
        try:
            tile = f"{parts[0]}{parts[1]}{parts[2]}"
            date = f"{parts[3]}-{int(parts[4]):02d}-{int(parts[5]):02d}"
        except Exception:
            pass
    return tile, date


def write_atlas_index(atlas_root: Path, tile_folders: list[str]) -> None:
    """Scene picker: atlas/index.html plus atlas_tiles.js in each tile folder."""
    atlas_root.mkdir(parents=True, exist_ok=True)
    entries = []
    for folder in tile_folders:
        label, date = tile_label(folder)
        html = atlas_root / folder / "postfire_landscape_atlas1.html"
        if not html.exists():
            continue
        entry = {"id": folder, "label": label, "date": date}
        js = atlas_root / folder / "mesma_grid.js"
        if js.exists():
            try:
                payload = json.loads(js.read_text().split("=", 1)[1].strip().rstrip(";"))
                ll = payload.get("lonlat") or {}
                if all(k in ll for k in ("west", "east", "south", "north")):
                    entry["ring"] = [
                        [ll["south"], ll["west"]],
                        [ll["south"], ll["east"]],
                        [ll["north"], ll["east"]],
                        [ll["north"], ll["west"]],
                    ]
            except Exception:
                pass
        entries.append(entry)
    js = "window.ATLAS_TILES=" + json.dumps(entries, separators=(",", ":")) + ";\n"
    (atlas_root / "atlas_tiles.js").write_text(js)
    for folder in tile_folders:
        dest = atlas_root / folder
        if dest.is_dir():
            (dest / "atlas_tiles.js").write_text(js)
    cards = "\n".join(
        f'<a class="scene" href="./{e["id"]}/postfire_landscape_atlas1.html">'
        f'<span class="scene-id">{e["label"]}</span>'
        f'<span class="scene-date">{e["date"]}</span>'
        f'<span class="scene-folder">{e["id"]}</span></a>'
        for e in entries
    )
    (atlas_root / "index.html").write_text(
        f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"/><meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Postfire recovery atlas — choose a scene</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Bitter:wght@600;700&family=IBM+Plex+Sans:wght@400;500&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root{{--page:#E9DFC1;--card:#F7F1DD;--ink:#221F17;--ink-soft:#5C5645;--line:#CDBF93;}}
  body{{margin:0;background:var(--page);color:var(--ink);font-family:'IBM Plex Sans',system-ui,sans-serif;}}
  .wrap{{max-width:820px;margin:0 auto;padding:36px 22px 70px;}}
  h1{{font-family:'Bitter',serif;font-size:30px;margin:0 0 8px;}}
  p{{color:var(--ink-soft);line-height:1.55;max-width:540px;}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:14px;margin-top:28px;}}
  .scene{{display:flex;flex-direction:column;gap:4px;background:var(--card);border:1.5px solid var(--ink);
    border-radius:10px;padding:16px 18px;text-decoration:none;color:var(--ink);}}
  .scene:hover{{background:#EFE5C7;}}
  .scene-id{{font-family:'Bitter',serif;font-weight:700;font-size:20px;}}
  .scene-date{{font-size:13px;color:var(--ink-soft);}}
  .scene-folder{{font-family:'IBM Plex Mono',monospace;font-size:11px;color:#8C826A;}}
</style></head><body><div class="wrap">
<h1>Postfire recovery atlas</h1>
<p>Choose a Sentinel-2 MESMA scene to open. Keep each tile folder together
(<code>postfire_landscape_atlas1.html</code>, <code>mesma_grid.js</code>,
<code>mesma_original.png</code>, <code>atlas_tiles.js</code>).</p>
<div class="grid">{cards or "<p>No atlases built yet.</p>"}</div>
</div></body></html>
"""
    )
    print(f"Wrote scene picker: {atlas_root / 'index.html'} ({len(entries)} scene(s))", flush=True)
    copy_tsf_html(atlas_root)
    copy_tsf_year_layers(atlas_root)


BURN_MAP_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Southern California days since last burn</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
  html,body,#map{height:100%;margin:0;}
  .legend{position:fixed;top:12px;left:12px;z-index:9999;background:#fff;padding:12px 14px;
    border:2px solid #444;border-radius:6px;font:13px Arial,sans-serif;max-width:280px;
    box-shadow:2px 2px 6px rgba(0,0,0,.25);}
  .legend b{display:block;margin-bottom:8px;}
  .row{display:flex;align-items:center;margin:4px 0;}
  .row i{width:16px;height:16px;margin-right:8px;border:1px solid #555;display:inline-block;}
</style>
</head>
<body>
<div id="map"></div>
<div class="legend">
  <b>Days Since Last Burn</b>
  <div class="row"><i style="background:#ffff00"></i>1 – 365 days (0 – 1 year)</div>
  <div class="row"><i style="background:#ffd000"></i>366 – 730 days (1 – 2 years)</div>
  <div class="row"><i style="background:#ff9f00"></i>731 – 1,825 days (2 – 5 years)</div>
  <div class="row"><i style="background:#ff6b00"></i>1,826 – 3,650 days (5 – 10 years)</div>
  <div class="row"><i style="background:#f03b20"></i>3,651 – 7,300 days (10 – 20 years)</div>
  <div class="row"><i style="background:#b10026"></i>&gt; 7,300 days (&gt; 20 years)</div>
  <div class="row"><i style="background:#808080"></i>0 days (No fire detected)</div>
  <hr>
  <div><b>Reference date:</b> 2026-07-28</div>
  <div><b>Data source:</b> MODIS MCD64A1</div>
  <div style="margin-top:6px;color:#555;">Fixed PNG overlay (does not expire)</div>
</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const map = L.map('map').setView([34.80, -118.775], 7);
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}', {
  attribution: 'Tiles &copy; Esri'
}).addTo(map);
L.imageOverlay('southern_california_burn_map.png', [[32.50, -122.80], [37.10, -114.75]], {opacity: 0.75}).addTo(map);
</script>
</body>
</html>
"""


def write_burn_map_html(atlas_root: Path) -> None:
    """Write a Leaflet HTML that overlays the local PNG. Never reuse expired EE Folium."""
    (atlas_root / "southern_california_burn_map.html").write_text(BURN_MAP_HTML)
    print("Wrote atlas/southern_california_burn_map.html (PNG overlay, no Earth Engine)", flush=True)


def copy_tsf_html(atlas_root: Path) -> None:
    """Copy the existing Time_since_fire HTML (and sibling PNG) into atlas/ and each tile folder."""
    home = find_home()
    name = "southern_california_burn_map.html"
    dirs = [
        Path.cwd(),
        Path.cwd() / "new_data" / "Time_since_fire",
        find_new_data(home) / "Time_since_fire",
        Path("/home/jovyan/data-store/new_data/Time_since_fire"),
        Path("/data-store/iplant/home/samiksha23/SCE-CU/sentinel2_data/new_data/Time_since_fire"),
        Path("/home/jovyan/data-store/iplant/home/samiksha23/SCE-CU/sentinel2_data/new_data/Time_since_fire"),
        home / "SCE-CU/sentinel2_data/new_data/Time_since_fire",
    ]
    found = [d / name for d in dirs if (d / name).exists()]
    if not found:
        print("No southern_california_burn_map.html in Time_since_fire; TSF tab will be empty.", flush=True)
        return
    src = next((p for p in found if p.with_suffix(".png").exists()), found[0])
    dests = [atlas_root / name]
    if atlas_root.exists():
        dests.extend(
            d / name
            for d in atlas_root.iterdir()
            if d.is_dir() and (d / "mesma_grid.js").exists()
        )
    for dest in dests:
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        png = src.with_suffix(".png")
        if png.exists():
            shutil.copy2(png, dest.with_suffix(".png"))
    print(f"Attached {src} ({src.stat().st_size/1e6:.2f} MB) to {len(dests)} atlas location(s)", flush=True)


TSF_YEAR_KEEP = (
    "burn_year",
    "burn_date",
    "Incid_Name",
    "tsf_at_ignition",
    "tsf_as_of_dec_prev",
    "days_from_dec_prev",
    "as_of_date",
    "BurnBndAc",
    "date_source",
)


def slim_tsf_geojson(src: Path, dest: Path) -> int:
    data = json.loads(src.read_text())
    out_feats = []
    for feat in data.get("features") or []:
        props = feat.get("properties") or {}
        slim = {k: props[k] for k in TSF_YEAR_KEEP if k in props}
        if "burn_year" in slim:
            try:
                slim["burn_year"] = int(slim["burn_year"])
            except (TypeError, ValueError):
                pass
        for key in ("tsf_at_ignition", "tsf_as_of_dec_prev", "days_from_dec_prev"):
            if key in slim:
                try:
                    slim[key] = round(float(slim[key]), 1)
                except (TypeError, ValueError):
                    pass
        geom = feat.get("geometry")
        if not geom:
            continue
        out_feats.append({"type": "Feature", "properties": slim, "geometry": geom})
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps({"type": "FeatureCollection", "features": out_feats}, separators=(",", ":"))
    )
    return len(out_feats)


def find_tsf_year_dir() -> Path | None:
    home = find_home()
    folders = ("by_year_2015_2026", "by_year_2015_2020")
    roots = [
        Path.cwd() / "Time_since_fire",
        Path.cwd(),
        find_new_data(home) / "Time_since_fire",
        Path("/home/jovyan/data-store/new_data/Time_since_fire"),
        Path("/data-store/iplant/home/samiksha23/SCE-CU/sentinel2_data/new_data/Time_since_fire"),
        Path("/home/jovyan/data-store/iplant/home/samiksha23/SCE-CU/sentinel2_data/new_data/Time_since_fire"),
        home / "SCE-CU/sentinel2_data/new_data/Time_since_fire",
    ]
    names = ("fires.geojson", "fires_web.geojson", "fires_2015_2026.geojson", "fires_2015_2020.geojson")
    for folder in folders:
        for root in roots:
            d = root / folder if root.name != folder else root
            if any((d / name).exists() for name in names):
                return d
    return None


def copy_tsf_year_layers(atlas_root: Path) -> None:
    """Copy 2015–2026 fire perimeters into atlas/tsf_years/ for the Fires layer."""
    src_dir = find_tsf_year_dir()
    if src_dir is None:
        print(
            "No year-fire GeoJSON yet; Fires tab will ask you to run tsf_by_year_2015_2020.ipynb.",
            flush=True,
        )
        return
    src = next(
        (
            src_dir / name
            for name in ("fires.geojson", "fires_web.geojson", "fires_2015_2026.geojson", "fires_2015_2020.geojson")
            if (src_dir / name).exists()
        ),
        None,
    )
    dest = atlas_root / "tsf_years" / "fires.geojson"
    n = slim_tsf_geojson(src, dest)
    print(f"Attached {n} year-fire perimeters to {dest}", flush=True)


def discover_fractions(date: str, home: Path, mesma_dir: Path | None = None) -> list[Path]:
    """Find ENVI *_fractions cubes in new_data/<date>/mesma/<tile>/."""
    roots = []
    if mesma_dir is not None:
        roots.append(Path(mesma_dir))
    new_data = find_new_data(home)
    roots.extend(
        [
            new_data / date / "mesma",
            home / "SCE-CU/sentinel2_data/new_data" / date / "mesma",
        ]
    )
    hits: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        print(f"Scanning MESMA folder: {root}", flush=True)
        for tile_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for p in sorted(tile_dir.iterdir()):
                if not p.is_file():
                    continue
                # ENVI binary is named ..._fractions with no extension
                if p.name.endswith("_fractions") and p.suffix == "":
                    hits.append(p)
        # fallback glob if files are not in per-tile folders
        for p in root.rglob("*_fractions"):
            if p.is_file() and p.suffix == "":
                hits.append(p)
    uniq = []
    seen = set()
    for p in hits:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def sample_tsf(tsf_path: Path, xs: np.ndarray, ys: np.ndarray, zone: int = 11) -> np.ndarray | None:
    """Sample TSF raster at UTM x/y arrays (flattened)."""
    try:
        from osgeo import gdal, osr

        ds = gdal.Open(str(tsf_path))
        if ds is None:
            return None
        band = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
        gt = ds.GetGeoTransform()
        src_srs = osr.SpatialReference()
        src_srs.ImportFromWkt(ds.GetProjection() or "")
        dst_srs = osr.SpatialReference()
        dst_srs.ImportFromEPSG(32600 + int(zone))
        if src_srs.ExportToWkt() and not src_srs.IsSame(dst_srs):
            ct = osr.CoordinateTransformation(dst_srs, src_srs)
        else:
            ct = None
        inv = gdal.InvGeoTransform(gt)
        out = np.full(xs.size, np.nan, np.float32)
        for i, (x, y) in enumerate(zip(xs.ravel(), ys.ravel())):
            px, py = x, y
            if ct is not None:
                px, py, _ = ct.TransformPoint(float(x), float(y))
            c = inv[0] + px * inv[1] + py * inv[2]
            r = inv[3] + px * inv[4] + py * inv[5]
            ci, ri = int(c), int(r)
            if 0 <= ri < band.shape[0] and 0 <= ci < band.shape[1]:
                v = band[ri, ci]
                if v > 0:
                    out[i] = v
        return out.reshape(xs.shape)
    except Exception as exc:
        print(f"TSF sample skipped: {exc}", flush=True)
        return None


def main():
    home = find_home()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fractions",
        default=str(
            home
            / "SCE-CU/sentinel2_data/new_data/2026-07-28/mesma/11_S_LA_2026_7_28"
            / "11_S_LA_2026_7_28_mesma_20260903T080145_fractions"
        ),
    )
    parser.add_argument(
        "--tsf-dir",
        default=str(find_new_data(home) / "Time_since_fire"),
    )
    parser.add_argument(
        "--out",
        default=str(home / "SCE-CU/sentinel2_data/new_data/2026-07-28/atlas/11_S_LA_2026_7_28"),
    )
    parser.add_argument(
        "--mesma-dir",
        default="",
        help="Folder of per-tile MESMA outputs, e.g. new_data/2026-07-28/mesma",
    )
    parser.add_argument("--html", default=str(Path(__file__).resolve().parent / "postfire_landscape_atlas1.html"))
    parser.add_argument("--max-dim", type=int, default=512)
    parser.add_argument(
        "--date",
        default="",
        help="If set (YYYY-MM-DD), build an atlas for every *_fractions cube under new_data/<date>/mesma/",
    )
    parser.add_argument(
        "--tile",
        default="",
        help="Scene folder name or substring. Comma-separated for several, e.g. 11_S_KR,11_S_KV. Default: all tiles.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Build at most N fractions cubes (0 = no cap). Use 5 for a trial run.",
    )
    args = parser.parse_args()

    if args.date or args.mesma_dir:
        date = args.date or "2026-07-28"
        found = discover_fractions(date, home, Path(args.mesma_dir) if args.mesma_dir else None)
        if args.tile.strip() and args.tile.strip().lower() != "all":
            needles = [
                n.strip().replace("-", "_").upper()
                for n in args.tile.split(",")
                if n.strip()
            ]
            found = [
                p
                for p in found
                if any(n in p.parent.name.upper() or n in p.name.upper() for n in needles)
            ]
        if args.limit and args.limit > 0:
            found = found[: args.limit]
        if not found:
            extra = f" matching --tile {args.tile}" if args.tile else ""
            sys.exit(
                f"No *_fractions files under new_data/{date}/mesma/{extra}\n"
                "Expected folders like 11_S_KR_2026_7_28/ with a file ending in _fractions."
            )
        print(f"Found {len(found)} fractions cube(s) for {date}:", flush=True)
        for frac in found:
            print(f"  {frac}", flush=True)
        script = str(Path(__file__).resolve())
        atlas_root = find_new_data(home) / date / "atlas"
        tile_folders = []
        for frac in found:
            tile_dir = frac.parent.name
            tile_folders.append(tile_dir)
            out = atlas_root / tile_dir
            print("\n====", frac, "====", flush=True)
            subprocess.check_call(
                [
                    sys.executable,
                    script,
                    "--fractions",
                    str(frac),
                    "--tsf-dir",
                    args.tsf_dir,
                    "--out",
                    str(out),
                    "--html",
                    args.html,
                    "--max-dim",
                    str(args.max_dim),
                ]
            )
        existing = [p.name for p in sorted(atlas_root.iterdir()) if p.is_dir()] if atlas_root.exists() else tile_folders
        write_atlas_index(atlas_root, existing)
        return

    fractions = Path(args.fractions)
    if not fractions.exists():
        sys.exit(f"Fractions file not found:\n  {fractions}\nLaunch a VICE Jupyter session so /data-store is mounted.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print("Fractions:", fractions, flush=True)
    print("Output:   ", out, flush=True)

    meta = parse_hdr(fractions)
    print("ENVI", meta, flush=True)
    mm = memmap_envi(fractions, meta)
    r0, r1, c0, c1, n_valid = valid_bbox(mm)
    print(f"Valid pixels {n_valid:,}  rows {r0}:{r1}  cols {c0}:{c1}", flush=True)

    crop = np.array(mm[:, r0:r1, c0:c1], dtype=np.float32)
    valid = crop[6] > 1e-6
    H, W = crop.shape[1], crop.shape[2]
    scale = max(H, W) / args.max_dim
    out_rows = max(32, int(round(H / scale)))
    out_cols = max(32, int(round(W / scale)))
    print(f"Atlas grid {out_cols} x {out_rows}", flush=True)

    agg, valid_cells = aggregate(crop, valid, out_rows, out_cols)
    print(f"Valid cells {int(valid_cells.sum()):,} / {out_rows * out_cols}", flush=True)

    xmin = meta["origin_x"] + c0 * meta["res"]
    xmax = meta["origin_x"] + c1 * meta["res"]
    ymax = meta["origin_y"] - r0 * meta["res"]
    ymin = meta["origin_y"] - r1 * meta["res"]
    ul = utm_to_ll(xmin, ymax, meta["zone"])
    lr = utm_to_ll(xmax, ymin, meta["zone"])

    # Native RGB preview (soil, pv, npv)
    vis_h = min(H, 900)
    vis_w = max(1, int(round(W * vis_h / H)))
    yy = np.linspace(0, H - 1, vis_h).astype(int)
    xx = np.linspace(0, W - 1, vis_w).astype(int)
    soil = np.clip(crop[4][np.ix_(yy, xx)], 0, 1)
    pv = np.clip(crop[2][np.ix_(yy, xx)], 0, 1)
    npv = np.clip(crop[1][np.ix_(yy, xx)], 0, 1)
    rgb = np.dstack([soil, pv, npv])
    rgb = np.power(rgb, 0.85)
    rgb[~valid[np.ix_(yy, xx)]] = 0
    png_path = out / "mesma_original.png"
    save_png(png_path, rgb * 255)
    print("Wrote", png_path, flush=True)

    tsf_pack = None
    tsf_path = find_tsf_raster(Path(args.tsf_dir))
    if tsf_path:
        print("TSF raster:", tsf_path, flush=True)
        cell_w = (xmax - xmin) / out_cols
        cell_h = (ymax - ymin) / out_rows
        cols = np.arange(out_cols)
        rows = np.arange(out_rows)
        xs = xmin + (cols + 0.5) * cell_w
        ys = ymax - (rows + 0.5) * cell_h
        xxg, yyg = np.meshgrid(xs, ys)
        sampled = sample_tsf(tsf_path, xxg, yyg, zone=meta["zone"])
        if sampled is not None:
            tsf_pack = pack_days(sampled, valid_cells)
            print("Joined time-since-fire into the atlas grid.", flush=True)
    else:
        print("No TSF GeoTIFF in Time_since_fire (HTML-only is OK). Fractions-only atlas.", flush=True)

    tile, date = tile_label(fractions.parent.name or fractions.name)

    payload = {
        "tile": tile,
        "sceneDate": date,
        "source": fractions.name,
        "rows": out_rows,
        "cols": out_cols,
        "cellMetersX": round((xmax - xmin) / out_cols, 2),
        "cellMetersY": round((ymax - ymin) / out_rows, 2),
        "utm": {"zone": meta["zone"], "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax},
        "lonlat": {
            "west": min(ul[0], lr[0]),
            "east": max(ul[0], lr[0]),
            "south": min(ul[1], lr[1]),
            "north": max(ul[1], lr[1]),
        },
        "nativeValidPixels": int(n_valid),
        "nativeBBoxPixels": [int(c0), int(r0), int(c1), int(r1)],
        "scale": 1000,
        "bands": {
            "pv": pack(agg["pv_n"], valid_cells),
            "npv": pack(agg["npv_n"], valid_cells),
            "soil": pack(agg["soil_n"], valid_cells),
            "rock": pack(agg["rock_n"], valid_cells),
            "water": pack(agg["water_n"], valid_cells),
            "shade": pack(agg["shade"], valid_cells),
            "none": pack(agg["none_n"], valid_cells),
        },
    }
    if tsf_pack is not None:
        payload["bands"]["tsf"] = tsf_pack

    js_path = out / "mesma_grid.js"
    js_path.write_text("window.MESMA_GRID=" + json.dumps(payload, separators=(",", ":")) + ";\n")
    print("Wrote", js_path, f"({js_path.stat().st_size/1e6:.1f} MB)", flush=True)

    html_src = Path(args.html)
    if html_src.exists():
        shutil.copy2(html_src, out / "postfire_landscape_atlas1.html")
        print("Wrote", out / "postfire_landscape_atlas1.html", "(grid stays in mesma_grid.js)", flush=True)
    else:
        print("HTML not found at", html_src, "— copy postfire_landscape_atlas1.html into the out folder.", flush=True)

    if out.parent.name == "atlas":
        folders = [p.name for p in sorted(out.parent.iterdir()) if p.is_dir()]
        write_atlas_index(out.parent, folders)

    print("\nDone. Open this folder in the Discovery Environment and download:")
    print(f"  {out}")
    print("Files: postfire_landscape_atlas1.html, mesma_grid.js, mesma_original.png, atlas_tiles.js")
    print("Or open atlas/index.html to pick a scene. Do not download the 6 GB fractions file.")


def join_tsf_into_atlas_grids(atlas_root: Path, tsf_dir: Path) -> int:
    """Add TSF days into existing mesma_grid.js files without re-reading MESMA cubes."""
    tsf_path = find_tsf_raster(Path(tsf_dir))
    if tsf_path is None:
        raise FileNotFoundError(f"No TSF GeoTIFF under {tsf_dir}")
    n = 0
    for folder in sorted(p for p in Path(atlas_root).iterdir() if p.is_dir()):
        js_path = folder / "mesma_grid.js"
        if not js_path.exists():
            continue
        raw = js_path.read_text()
        marker = "window.MESMA_GRID="
        if marker not in raw:
            continue
        g = json.loads(raw.split(marker, 1)[1].strip().rstrip(";"))
        utm = g.get("utm") or {}
        rows, cols = int(g["rows"]), int(g["cols"])
        if not all(k in utm for k in ("xmin", "xmax", "ymin", "ymax")):
            print("  skip (no UTM)", folder.name, flush=True)
            continue
        xmin, xmax, ymin, ymax = utm["xmin"], utm["xmax"], utm["ymin"], utm["ymax"]
        xs = xmin + (np.arange(cols) + 0.5) * ((xmax - xmin) / cols)
        ys = ymax - (np.arange(rows) + 0.5) * ((ymax - ymin) / rows)
        xx, yy = np.meshgrid(xs, ys)
        sampled = sample_tsf(tsf_path, xx, yy, zone=int(utm.get("zone") or 11))
        if sampled is None:
            print("  skip (sample failed)", folder.name, flush=True)
            continue
        pv = np.array(g["bands"]["pv"], dtype=np.int32).reshape(rows, cols)
        valid = pv >= 0
        g["bands"]["tsf"] = pack_days(sampled, valid)
        js_path.write_text(marker + json.dumps(g, separators=(",", ":")) + ";\n")
        n += 1
        print(f"  joined TSF days → {folder.name}", flush=True)
    print(f"Joined TSF days into {n} tile grid(s) from {tsf_path}", flush=True)
    return n


DATE_LABELS = {
    "2026-07-28": "July 2026",
    "2026-10-16": "October 2026",
}


def collect_date_entries(new_data: Path, date: str) -> list[dict]:
    atlas = new_data / date / "atlas"
    if not atlas.exists():
        return []
    tiles = []
    js = atlas / "atlas_tiles.js"
    if js.exists():
        try:
            raw = js.read_text().split("=", 1)[1].strip().rstrip(";")
            tiles = json.loads(raw)
        except Exception:
            tiles = []
    if not tiles:
        for folder in sorted(p.name for p in atlas.iterdir() if p.is_dir()):
            html = atlas / folder / "postfire_landscape_atlas1.html"
            if html.exists():
                label, parsed = tile_label(folder)
                tiles.append({"id": folder, "label": label, "date": parsed or date})
    out = []
    for t in tiles:
        if t.get("id") == "test":
            continue
        label, parsed = tile_label(t.get("id") or "")
        d = t.get("date") or parsed or date
        out.append(
            {
                "id": t["id"],
                "label": t.get("label") or label,
                "date": d,
                "href": f"{date}/atlas/{t['id']}/postfire_landscape_atlas1.html?v={int(time.time())}",
                "ring": t.get("ring"),
                "center": t.get("center"),
            }
        )
    return out


def find_atlas_html(html_src: Path | None = None) -> Path | None:
    cands = []
    if html_src:
        cands.append(Path(html_src))
    cands.extend(
        [
            Path.cwd() / "postfire_landscape_atlas1.html",
            Path(__file__).resolve().parent / "postfire_landscape_atlas1.html",
        ]
    )
    for cand in cands:
        if cand.exists() and cand.stat().st_size > 1000:
            return cand
    return None


def copy_atlas_html(
    new_data: Path,
    html_src: Path | None = None,
    dates: list[str] | None = None,
) -> int:
    """Copy postfire_landscape_atlas1.html into every existing atlas scene folder."""
    src = find_atlas_html(html_src)
    if src is None:
        print(
            "No postfire_landscape_atlas1.html next to this notebook — "
            "upload it, then re-run so scene pages update.",
            flush=True,
        )
        return 0
    if dates is None:
        dates = sorted(
            p.name
            for p in Path(new_data).iterdir()
            if p.is_dir() and (p / "atlas").exists() and re.match(r"\d{4}-\d{2}-\d{2}$", p.name)
        )
    text = src.read_text()
    n = 0
    for date in dates:
        atlas = Path(new_data) / date / "atlas"
        if not atlas.exists():
            continue
        for folder in atlas.iterdir():
            if folder.is_dir() and (folder / "mesma_grid.js").exists():
                (folder / "postfire_landscape_atlas1.html").write_text(text)
                n += 1
    print(
        f"Copied {src.name} ({src.stat().st_size/1024:.0f} KB) into {n} scene page(s)",
        flush=True,
    )
    return n


def write_multi_date_index(new_data: Path, dates: list[str] | None = None) -> Path:
    """Write new_data/index.html with a July/October dropdown over both atlases."""
    if dates is None:
        dates = sorted(
            p.name
            for p in new_data.iterdir()
            if p.is_dir() and (p / "atlas").exists() and re.match(r"\d{4}-\d{2}-\d{2}$", p.name)
        )
    copy_atlas_html(new_data, dates=dates)
    entries = []
    for date in dates:
        entries.extend(collect_date_entries(new_data, date))
    if not entries:
        raise FileNotFoundError(f"No atlas scenes under {new_data}/<date>/atlas/")
    dest = new_data / "index.html"
    dest.write_text(_multi_date_picker_html(entries))
    print(f"Wrote {dest} ({len(entries)} scenes, dates {sorted({e['date'] for e in entries})})", flush=True)
    return dest


def _multi_date_picker_html(entries: list[dict]) -> str:
    tiles_json = json.dumps(entries, separators=(",", ":"))
    labels = json.dumps(DATE_LABELS, separators=(",", ":"))
    aoi = json.dumps(
        [[32.53, -120.708008], [38.395343, -113.975687]],
        separators=(",", ":"),
    )
    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"/><meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Postfire recovery atlas — choose a scene</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Bitter:wght@600;700&family=IBM+Plex+Sans:wght@400;500&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
  :root{{--page:#E9DFC1;--card:#F7F1DD;--ink:#221F17;--ink-soft:#5C5645;}}
  *{{box-sizing:border-box;}}
  body{{margin:0;background:var(--page);color:var(--ink);font-family:'IBM Plex Sans',system-ui,sans-serif;}}
  .wrap{{max-width:1180px;margin:0 auto;padding:28px 18px 60px;}}
  h1{{font-family:'Bitter',serif;font-size:30px;margin:0 0 8px;}}
  .lede{{color:var(--ink-soft);line-height:1.55;max-width:720px;margin:0 0 16px;}}
  .side{{display:flex;flex-direction:column;gap:8px;min-height:0;}}
  .date-box{{background:var(--card);border:1.5px solid var(--ink);border-radius:8px;padding:10px;}}
  .date-box label{{display:block;font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-soft);margin:0 0 6px;}}
  .date-box select{{width:100%;font:600 15px 'IBM Plex Sans',sans-serif;padding:8px 10px;border:1.5px solid var(--ink);border-radius:6px;background:#fff;}}
  .stage{{display:grid;grid-template-columns:minmax(0,1fr) 220px;gap:14px;align-items:stretch;}}
  @media (max-width:820px){{.stage{{grid-template-columns:1fr;}}}}
  #pickerMap{{width:100%;aspect-ratio:1.15;min-height:520px;border:2px solid var(--ink);border-radius:8px;background:#dce6d4;}}
  .leaflet-control-zoom,.leaflet-control-attribution{{display:none !important;}}
  .tile-lab{{background:none !important;border:none !important;}}
  .tile-lab .mark{{display:flex;align-items:center;gap:3px;transform:translate(-4px,-6px);}}
  .tile-lab .dot{{width:7px;height:7px;background:#1a3a7a;border:1.5px solid #fff;border-radius:50%;flex:none;}}
  .tile-lab .name{{font:600 10px 'IBM Plex Mono',monospace;color:#12306a;background:rgba(247,241,221,.92);padding:0 3px;line-height:14px;white-space:nowrap;}}
  .list{{background:var(--card);border:1.5px solid var(--ink);border-radius:8px;padding:10px;overflow:auto;max-height:min(72vh,760px);}}
  .list a{{display:block;padding:7px 8px;border-radius:6px;text-decoration:none;color:var(--ink);font-size:13px;}}
  .list a:hover,.list a.on{{background:#EFE5C7;}}
  .list .id{{font-family:'Bitter',serif;font-weight:700;}}
  .list .sub{{display:block;font-family:'IBM Plex Mono',monospace;font-size:10px;color:#8C826A;}}
  .tip{{background:#fff;border:1.5px solid var(--ink);padding:6px 8px;font:12px 'IBM Plex Sans',sans-serif;}}
</style></head>
<body><div class="wrap">
<h1>Postfire recovery atlas</h1>
<p class="lede">Use the dropdown on the right for July or October, then click a Sentinel-2 square. Both dates stay on disk.</p>
<div class="stage">
  <div id="pickerMap"></div>
  <div class="side">
    <div class="date-box">
      <label for="dateSel">Scene date</label>
      <select id="dateSel"></select>
    </div>
    <nav class="list" id="tileList"></nav>
  </div>
</div>
</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const AOI = {aoi};
const ALL = {tiles_json};
const DATE_LABELS = {labels};
const dates = [...new Set(ALL.map(t => t.date))].sort();
let DATE = dates.includes('2026-07-28') ? '2026-07-28' : dates[0];
const sel = document.getElementById('dateSel');
dates.forEach(d => {{
  const o = document.createElement('option');
  o.value = d;
  o.textContent = DATE_LABELS[d] || d;
  if (d === DATE) o.selected = true;
  sel.appendChild(o);
}});
const map = L.map('pickerMap', {{
  zoomControl:false, dragging:false, scrollWheelZoom:false, doubleClickZoom:false,
  boxZoom:false, keyboard:false, touchZoom:false, attributionControl:false
}});
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{{z}}/{{y}}/{{x}}', {{attribution:'Tiles &copy; Esri'}}).addTo(map);
L.rectangle(AOI, {{color:'#d62728', weight:2.5, fill:false, interactive:false}}).addTo(map);
const layerGroup = L.layerGroup().addTo(map);
const list = document.getElementById('tileList');
let layers = {{}};
let listLinks = {{}};
let current = null;
function tiles(){{ return ALL.filter(t => t.date === DATE); }}
function hrefOf(t){{ return t.href || ('./' + t.id + '/postfire_landscape_atlas1.html'); }}
function centerOf(t){{
  if (t.center && t.center.length===2) return t.center;
  const ring = t.ring || [];
  if (!ring.length) return null;
  return [ring.reduce((s,p)=>s+p[0],0)/ring.length, ring.reduce((s,p)=>s+p[1],0)/ring.length];
}}
function viewBounds(){{
  const b = L.latLngBounds(AOI);
  tiles().forEach(t => {{ if (t.ring) t.ring.forEach(p => b.extend(p)); }});
  return b;
}}
function lockAOI(){{
  map.setMinZoom(0); map.setMaxZoom(18);
  map.fitBounds(viewBounds(), {{padding:[8,8], animate:false}});
  const z = map.getZoom();
  map.setMinZoom(z); map.setMaxZoom(z);
}}
function nearestTile(latlng){{
  let best=null, bestD=1e15;
  tiles().forEach(t => {{
    const c = centerOf(t);
    if (!c) return;
    const d = map.distance(latlng, c);
    if (d < bestD) {{ bestD=d; best=t; }}
  }});
  return (best && bestD < 70000) ? best : null;
}}
function setCurrent(t){{
  if (current && layers[current.id]) {{
    layers[current.id].setStyle({{weight:1.5, color:'#2450a0', fillOpacity:0.01}});
    if (listLinks[current.id]) listLinks[current.id].classList.remove('on');
  }}
  current = t;
  if (t && layers[t.id]) {{
    layers[t.id].setStyle({{weight:2.5, color:'#12306a', fillOpacity:0.16}});
    layers[t.id].bringToFront();
    if (listLinks[t.id]) {{
      listLinks[t.id].classList.add('on');
      listLinks[t.id].scrollIntoView({{block:'nearest'}});
    }}
  }}
}}
function render(){{
  layerGroup.clearLayers();
  list.innerHTML = '';
  layers = {{}}; listLinks = {{}}; current = null;
  tiles().forEach(t => {{
    const href = hrefOf(t);
    if (t.ring && t.ring.length) {{
      const poly = L.polygon(t.ring, {{color:'#2450a0', weight:1.5, fillColor:'#2450a0', fillOpacity:0.01, interactive:false}});
      poly.addTo(layerGroup);
      layers[t.id] = poly;
    }}
    const c = centerOf(t);
    if (c) {{
      const lab = L.marker(c, {{
        icon: L.divIcon({{className:'tile-lab', html:'<span class="mark"><span class="dot"></span><span class="name">'+t.label+'</span></span>', iconSize:[0,0]}}),
        keyboard:false, zIndexOffset:600
      }});
      lab.on('click', () => {{ location.href = href; }});
      lab.addTo(layerGroup);
    }}
    const a = document.createElement('a');
    a.href = href;
    a.innerHTML = '<span class="id">' + t.label + '</span><span class="sub">' + t.id + '</span>';
    a.onmouseenter = () => setCurrent(t);
    list.appendChild(a);
    listLinks[t.id] = a;
  }});
  lockAOI();
}}
sel.onchange = () => {{ DATE = sel.value; render(); }};
render();
window.addEventListener('resize', lockAOI);
const tip = L.tooltip({{sticky:true, className:'tip'}});
map.on('mousemove', e => {{
  const t = nearestTile(e.latlng);
  setCurrent(t);
  if (t) {{ tip.setLatLng(e.latlng).setContent(t.label); map.openTooltip(tip); }}
  else map.closeTooltip(tip);
}});
map.on('click', e => {{
  const t = nearestTile(e.latlng);
  if (t) location.href = hrefOf(t);
}});
map.on('mouseout', () => {{ setCurrent(null); map.closeTooltip(tip); }});
</script></body></html>
"""


if __name__ == "__main__":
    main()
