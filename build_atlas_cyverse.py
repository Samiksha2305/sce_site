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


def utm_to_ll(x: float, y: float, zone: int = 11, northern: bool = True):
    """WGS84 UTM → lon/lat. Used for tile squares on the scene-picker map."""
    a = 6378137.0
    f = 1 / 298.257223563
    k0 = 0.9996
    e = math.sqrt(f * (2 - f))
    e1 = (1 - math.sqrt(1 - e * e)) / (1 + math.sqrt(1 - e * e))
    easting = x - 500000.0
    northing = y if northern else y - 10_000_000.0
    lon0 = math.radians(-183.0 + 6.0 * zone)
    m = northing / k0
    mu = m / (a * (1 - e**2 / 4 - 3 * e**4 / 64 - 5 * e**6 / 256))
    phi1 = (
        mu
        + (3 * e1 / 2 - 27 * e1**3 / 32) * math.sin(2 * mu)
        + (21 * e1**2 / 16 - 55 * e1**4 / 32) * math.sin(4 * mu)
        + (151 * e1**3 / 96) * math.sin(6 * mu)
    )
    sinp, cosp, tanp = math.sin(phi1), math.cos(phi1), math.tan(phi1)
    ep2 = e**2 / (1 - e**2)
    n1 = a / math.sqrt(1 - e**2 * sinp**2)
    t1 = tanp**2
    c1 = ep2 * cosp**2
    r1 = a * (1 - e**2) / (1 - e**2 * sinp**2) ** 1.5
    d = easting / (n1 * k0)
    lat = phi1 - (n1 * tanp / r1) * (
        d**2 / 2
        - (5 + 3 * t1 + 10 * c1 - 4 * c1**2 - 9 * ep2) * d**4 / 24
        + (61 + 90 * t1 + 298 * c1 + 45 * t1**2 - 252 * ep2 - 3 * c1**2) * d**6 / 720
    )
    lon = lon0 + (
        d
        - (1 + 2 * t1 + c1) * d**3 / 6
        + (5 - 2 * c1 + 28 * t1 - 3 * c1**2 + 8 * ep2 + 24 * t1**2) * d**5 / 120
    ) / cosp
    return math.degrees(lon), math.degrees(lat)


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


# Same rectangle as download_from_copernicus_final.ipynb (the matplotlib figure).
# Southern_California_AOI.geojson (−122.80, 32.50)–(-114.75, 37.10) is a larger
# planning envelope; using it leaves empty ocean and clips the official tiles.
AOI_BOUNDS = {
    "west": -120.708008,
    "south": 32.53,
    "east": -113.975687,
    "north": 38.395343,
}

# Official ESA S2A_OPER_GIP_TILPAR footprints ([lat, lon] for Leaflet).
# Same KML as SCE-CU/Mesma/download/download_from_copernicus_final.ipynb
S2_KML_FOOTPRINTS = {
    "10SGB": {"ring": [[33.420926,-120.849393],[33.394855,-119.669903],[32.406061,-119.706744],[32.431169,-120.873216],[33.420926,-120.849393]], "center": [32.913494,-120.27156]},
    "11SKR": {"ring": [[32.496814,-120.192952],[32.521047,-119.02522],[31.53107,-119.003545],[31.507745,-120.158821],[32.496814,-120.192952]], "center": [32.014396,-119.598248]},
    "11SKS": {"ring": [[33.39762,-120.225533],[33.422698,-119.045911],[32.432875,-119.023246],[32.408724,-120.189844],[33.39762,-120.225533]], "center": [32.915711,-119.62439]},
    "11SKT": {"ring": [[34.298266,-120.259596],[34.324206,-119.067545],[33.33454,-119.043847],[33.309546,-120.222283],[34.298266,-120.259596]], "center": [33.816876,-119.651722]},
    "11SKU": {"ring": [[35.198208,-120.295184],[35.225028,-119.090147],[34.235522,-119.065374],[34.209667,-120.256178],[35.198208,-120.295184]], "center": [34.717347,-119.680279]},
    "11SKV": {"ring": [[36.098523,-120.332407],[36.126243,-119.11379],[35.136899,-119.087891],[35.110165,-120.291632],[36.098523,-120.332407]], "center": [35.618204,-119.710149]},
    "11SLA": {"ring": [[37.025276,-119.24838],[37.042247,-118.014303],[36.052541,-118.001452],[36.036168,-119.219917],[37.025276,-119.24838]], "center": [36.539207,-118.624916]},
    "11SLB": {"ring": [[37.925582,-119.275528],[37.943111,-118.02656],[36.953559,-118.013125],[36.936642,-119.245772],[37.925582,-119.275528]], "center": [37.439877,-118.644327]},
    "11SLR": {"ring": [[32.519328,-119.129284],[32.533687,-117.960535],[31.543236,-117.950248],[31.529415,-119.106497],[32.519328,-119.129284]], "center": [32.031551,-118.539766]},
    "11SLS": {"ring": [[33.420919,-119.151037],[33.435779,-117.970355],[32.445473,-117.959598],[32.431162,-119.127209],[33.420919,-119.151037]], "center": [32.93347,-118.555318]},
    "11SLT": {"ring": [[34.322366,-119.17378],[34.337737,-117.980622],[33.347578,-117.969375],[33.332767,-119.148867],[34.322366,-119.17378]], "center": [33.835252,-118.571578]},
    "11SLU": {"ring": [[35.223125,-119.197542],[35.239018,-117.99135],[34.249008,-117.979592],[34.233687,-119.171497],[35.223125,-119.197542]], "center": [34.736352,-118.588567]},
    "11SLV": {"ring": [[36.124276,-119.222396],[36.140702,-118.002571],[35.150843,-117.990279],[35.135002,-119.19517],[36.124276,-119.222396]], "center": [35.637852,-118.606338]},
    "11SMA": {"ring": [[37.04125,-118.124935],[37.046532,-116.890241],[36.056675,-116.891632],[36.051579,-118.110684],[37.04125,-118.124935]], "center": [36.549055,-117.507588]},
    "11SMB": {"ring": [[37.942081,-118.138529],[37.947537,-116.888914],[36.95783,-116.890368],[36.952565,-118.12363],[37.942081,-118.138529]], "center": [37.450051,-117.513722]},
    "11SMR": {"ring": [[32.532843,-118.065306],[32.537312,-116.89606],[31.546725,-116.897173],[31.542424,-118.053897],[32.532843,-118.065306]], "center": [32.039868,-117.480683]},
    "11SMS": {"ring": [[33.434905,-118.076197],[33.439531,-116.894997],[32.449086,-116.896161],[32.444632,-118.064267],[33.434905,-118.076197]], "center": [32.942081,-117.485597]},
    "11SMT": {"ring": [[34.336833,-118.087583],[34.341617,-116.893886],[33.351317,-116.895103],[33.346707,-118.07511],[34.336833,-118.087583]], "center": [33.844162,-117.490735]},
    "11SMU": {"ring": [[35.238084,-118.09948],[35.24303,-116.892725],[34.252876,-116.893998],[34.248107,-118.08644],[35.238084,-118.09948]], "center": [34.745569,-117.496103]},
    "11SMV": {"ring": [[36.139737,-118.111925],[36.144849,-116.89151],[35.154843,-116.892841],[35.149912,-118.098293],[36.139737,-118.111925]], "center": [35.647381,-117.501718]},
    "11SNA": {"ring": [[37.046583,-117.000225],[37.040161,-115.765562],[36.050528,-115.7812],[36.056724,-117.000222],[37.046583,-117.000225]], "center": [36.548556,-116.382893]},
    "11SNB": {"ring": [[37.94759,-117.000228],[37.940956,-115.750646],[36.951479,-115.766995],[36.957881,-117.000225],[37.94759,-117.000228]], "center": [37.449535,-116.375437]},
    "11SNR": {"ring": [[32.537355,-117.000213],[32.531922,-115.830993],[31.541537,-115.843511],[31.546767,-117.000211],[32.537355,-117.000213]], "center": [32.039446,-116.415603]},
    "11SNS": {"ring": [[33.439575,-117.000215],[33.433952,-115.819042],[32.443714,-115.832133],[32.449129,-117.000213],[33.439575,-117.000215]], "center": [32.941644,-116.409629]},
    "11SNT": {"ring": [[34.341663,-117.000217],[34.335847,-115.806548],[33.345757,-115.820235],[33.351361,-117.000215],[34.341663,-117.000217]], "center": [33.84371,-116.403383]},
    "11SNU": {"ring": [[35.243078,-117.00022],[35.237064,-115.793493],[34.247124,-115.807802],[34.252922,-117.000217],[35.243078,-117.00022]], "center": [34.745101,-116.396857]},
    "11SNV": {"ring": [[36.144898,-117.000222],[36.138683,-115.779838],[35.148896,-115.794796],[35.15489,-117.00022],[36.144898,-117.000222]], "center": [35.646897,-116.39003]},
    "11SPA": {"ring": [[37.041254,-115.875514],[37.023138,-114.641532],[36.034105,-114.671386],[36.051583,-115.88976],[37.041254,-115.875514]], "center": [36.53768,-115.265646]},
    "11SPB": {"ring": [[37.942085,-115.861926],[37.923373,-114.613058],[36.934511,-114.644268],[36.952569,-115.87682],[37.942085,-115.861926]], "center": [37.438298,-115.244939]},
    "11SPR": {"ring": [[32.532847,-115.93512],[32.517519,-114.766447],[31.527674,-114.790348],[31.542427,-115.946524],[32.532847,-115.93512]], "center": [32.03026,-115.356486]},
    "11SPS": {"ring": [[33.434909,-115.924234],[33.419047,-114.743632],[32.429359,-114.768624],[32.444635,-115.936159],[33.434909,-115.924234]], "center": [32.932134,-115.339895]},
    "11SPT": {"ring": [[34.336837,-115.912852],[34.320429,-114.719777],[33.330901,-114.745908],[33.346711,-115.92532],[34.336837,-115.912852]], "center": [33.833869,-115.322549]},
    "11SPU": {"ring": [[35.238088,-115.900959],[35.221123,-114.694855],[34.231757,-114.722171],[34.248111,-115.913994],[35.238088,-115.900959]], "center": [34.734922,-115.304424]},
    "11SPV": {"ring": [[36.139741,-115.888519],[36.122207,-114.668785],[35.133006,-114.697342],[35.149916,-115.902146],[36.139741,-115.888519]], "center": [35.636374,-115.285466]},
    "11SQA": {"ring": [[37.025284,-114.752069],[36.995511,-113.519414],[36.007451,-113.563414],[36.036176,-114.780526],[37.025284,-114.752069]], "center": [36.516368,-114.14997]},
    "11SQB": {"ring": [[37.925591,-114.724927],[37.894839,-113.477449],[36.906972,-113.523446],[36.93665,-114.754677],[37.925591,-114.724927]], "center": [37.416281,-114.116063]},
    "11SQR": {"ring": [[32.519335,-114.871141],[32.494142,-113.703535],[31.505174,-113.738769],[31.529422,-114.893924],[32.519335,-114.871141]], "center": [32.012254,-114.29873]},
    "11SQS": {"ring": [[33.420926,-114.849393],[33.394855,-113.669903],[32.406061,-113.706744],[32.431169,-114.873216],[33.420926,-114.849393]], "center": [32.913494,-114.27156]},
    "11SQT": {"ring": [[34.322373,-114.826654],[34.295406,-113.634741],[33.30679,-113.673258],[33.332775,-114.851562],[34.322373,-114.826654]], "center": [33.814582,-114.243152]},
    "11SQU": {"ring": [[35.223133,-114.802898],[35.195251,-113.598005],[34.206817,-113.63827],[34.233695,-114.828937],[35.223133,-114.802898]], "center": [34.714975,-114.213471]},
    "11SQV": {"ring": [[36.124284,-114.778048],[36.095467,-113.559581],[35.107218,-113.601672],[35.13501,-114.805269],[36.124284,-114.778048]], "center": [35.615751,-114.182425]},
}

# MGRS 100 km column letters: zone%3 == 1, 2, 0 (fallback if a tile is not in the KML table)
_MGRS_COLS = {1: "ABCDEFGH", 2: "JKLMNPQR", 0: "STUVWXYZ"}
_MGRS_ROWS = "ABCDEFGHJKLMNPQRSTUV"
_S2_REF = {"code": "11SKV", "x": 199980.0, "y": 4000020.0, "size": 109800.0}


def _s2_code(folder: str) -> str:
    label, _ = tile_label(folder)
    return label.replace("_", "")


def _mgrs_row_steps(row: str, zone: int, ref_row: str = "V") -> int:
    """100 km row steps from 11SKV. Even UTM zones shift MGRS row letters by 5."""
    ri = _MGRS_ROWS.index(row)
    if zone % 2 == 0:
        ri = (ri - 5) % 20
    rr = _MGRS_ROWS.index(ref_row)
    d = ri - rr
    if d < -10:
        d += 20
    elif d > 10:
        d -= 20
    return d


def s2_tile_corners_lonlat(code: str) -> list[list[float]] | None:
    """Official ESA KML ring, else a UTM-square fallback."""
    code = code.replace("_", "")
    official = S2_KML_FOOTPRINTS.get(code)
    if official:
        return official["ring"]
    if len(code) < 5 or not code[:2].isdigit():
        return None
    zone = int(code[:2])
    col, row = code[3], code[4]
    cols = _MGRS_COLS[zone % 3]
    if col not in cols or row not in _MGRS_ROWS:
        return None
    sw_e = 100000.0 * (cols.index(col) + 1)
    sw_n = 3_900_000.0 + _mgrs_row_steps(row, zone) * 100000.0
    xmin = sw_e - 20.0
    ymax = sw_n + 100020.0
    size = _S2_REF["size"]
    ring = []
    for xx, yy in ((xmin, ymax), (xmin + size, ymax), (xmin + size, ymax - size), (xmin, ymax - size)):
        lon, lat = utm_to_ll(xx, yy, zone=zone)
        ring.append([round(lat, 5), round(lon, 5)])
    return ring


def lonlat_from_grid_js(js_path: Path) -> list[list[float]] | None:
    raw = js_path.read_bytes()[:8000].decode("utf-8", "ignore")
    m = re.search(
        r'"lonlat":\{"west":(?P<w>[-\d.]+),"east":(?P<e>[-\d.]+),"south":(?P<s>[-\d.]+),"north":(?P<n>[-\d.]+)\}',
        raw,
    )
    if not m:
        return None
    west, east, south, north = (float(m.group(k)) for k in ("w", "e", "s", "n"))
    return [[south, west], [south, east], [north, east], [north, west]]


def lonlat_from_hdr(fractions: Path) -> list[list[float]] | None:
    try:
        meta = parse_hdr(fractions)
    except Exception:
        return None
    if not meta.get("samples") or not meta.get("lines"):
        return None
    xmin, ymax = meta["origin_x"], meta["origin_y"]
    xmax = xmin + meta["samples"] * meta["res"]
    ymin = ymax - meta["lines"] * meta["res"]
    ring = []
    for xx, yy in ((xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)):
        lon, lat = utm_to_ll(xx, yy, zone=int(meta["zone"]))
        ring.append([round(lat, 5), round(lon, 5)])
    return ring


def find_tile_fractions(tile_dir: Path) -> Path | None:
    if not tile_dir.exists():
        return None
    for p in tile_dir.iterdir():
        if p.is_file() and p.name.endswith("_fractions") and p.suffix == "":
            return p
    return None


def tile_map_geom(atlas_tile: Path, mesma_tile: Path | None, folder: str) -> tuple[list[list[float]] | None, list[float] | None]:
    """Official ESA tile polygon + center (same footprints as the Copernicus KML plot)."""
    code = _s2_code(folder)
    official = S2_KML_FOOTPRINTS.get(code)
    if official:
        return official["ring"], official.get("center")
    ring = s2_tile_corners_lonlat(code)
    if ring:
        return ring, None
    if mesma_tile is not None:
        frac = find_tile_fractions(mesma_tile)
        if frac is not None:
            ring = lonlat_from_hdr(frac)
            if ring:
                return ring, None
    js = atlas_tile / "mesma_grid.js"
    if js.exists():
        return lonlat_from_grid_js(js), None
    return None, None


def tile_map_ring(atlas_tile: Path, mesma_tile: Path | None, folder: str) -> list[list[float]] | None:
    ring, _ = tile_map_geom(atlas_tile, mesma_tile, folder)
    return ring


def scene_picker_html(entries: list[dict]) -> str:
    tiles_json = json.dumps(entries, separators=(",", ":"))
    aoi = json.dumps(
        [[AOI_BOUNDS["south"], AOI_BOUNDS["west"]], [AOI_BOUNDS["north"], AOI_BOUNDS["east"]]],
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
  :root{{--page:#E9DFC1;--card:#F7F1DD;--ink:#221F17;--ink-soft:#5C5645;--line:#CDBF93;--forest:#2F5233;}}
  *{{box-sizing:border-box;}}
  body{{margin:0;background:var(--page);color:var(--ink);font-family:'IBM Plex Sans',system-ui,sans-serif;}}
  .wrap{{max-width:1180px;margin:0 auto;padding:28px 18px 60px;}}
  h1{{font-family:'Bitter',serif;font-size:30px;margin:0 0 8px;}}
  .lede{{color:var(--ink-soft);line-height:1.55;max-width:640px;margin:0 0 16px;}}
  .stage{{display:grid;grid-template-columns:minmax(0,1fr) 220px;gap:14px;align-items:stretch;}}
  @media (max-width:820px){{.stage{{grid-template-columns:1fr;}}}}
  #pickerMap{{width:100%;aspect-ratio:1.15;min-height:520px;border:2px solid var(--ink);border-radius:8px;background:#dce6d4;}}
  .leaflet-control-zoom,.leaflet-control-attribution{{display:none !important;}}
  .leaflet-container{{cursor:pointer;}}
  .tile-lab{{background:none !important;border:none !important;}}
  .tile-lab .mark{{display:flex;align-items:center;gap:3px;transform:translate(-4px,-6px);}}
  .tile-lab .dot{{width:7px;height:7px;background:#1a3a7a;border:1.5px solid #fff;border-radius:50%;flex:none;}}
  .tile-lab .name{{font:600 10px 'IBM Plex Mono',monospace;color:#12306a;background:rgba(247,241,221,.92);
    padding:0 3px;line-height:14px;white-space:nowrap;}}
  .list{{background:var(--card);border:1.5px solid var(--ink);border-radius:8px;padding:10px;overflow:auto;max-height:min(72vh,760px);}}
  .list a{{display:block;padding:7px 8px;border-radius:6px;text-decoration:none;color:var(--ink);font-size:13px;}}
  .list a:hover,.list a.on{{background:#EFE5C7;}}
  .list .id{{font-family:'Bitter',serif;font-weight:700;}}
  .list .sub{{display:block;font-family:'IBM Plex Mono',monospace;font-size:10px;color:#8C826A;}}
  .tip{{background:#fff;border:1.5px solid var(--ink);padding:6px 8px;font:12px 'IBM Plex Sans',sans-serif;}}
</style></head>
<body><div class="wrap">
<h1>Postfire recovery atlas</h1>
<p class="lede">Click a Sentinel-2 square to open that scene. Names sit on the official ESA tile centers (same TILE_ID as the Copernicus KML). Blue outlines overlap on purpose — that is how the tiles are defined.</p>
<div class="stage">
  <div id="pickerMap"></div>
  <nav class="list" id="tileList"></nav>
</div>
</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const AOI = {aoi};
const TILES = {tiles_json};
const map = L.map('pickerMap', {{
  zoomControl:false, dragging:false, scrollWheelZoom:false, doubleClickZoom:false,
  boxZoom:false, keyboard:false, touchZoom:false, attributionControl:false
}});
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
  attribution: 'Tiles &copy; Esri'
}}).addTo(map);
function viewBounds(){{
  const b = L.latLngBounds(AOI);
  TILES.forEach(t => {{
    if (t.ring && t.ring.length) t.ring.forEach(p => b.extend(p));
  }});
  return b;
}}
function lockAOI(){{
  map.fitBounds(viewBounds(), {{padding:[8,8], animate:false}});
  const z = map.getZoom();
  map.setMinZoom(z);
  map.setMaxZoom(z);
}}
lockAOI();
window.addEventListener('resize', lockAOI);
L.rectangle(AOI, {{color:'#d62728', weight:2.5, fill:false, interactive:false}}).addTo(map);
const layers = {{}};
const list = document.getElementById('tileList');
const listLinks = {{}};
function centerOf(t){{
  if (t.center && t.center.length===2) return t.center;
  const ring = t.ring || [];
  if (!ring.length) return null;
  return [ring.reduce((s,p)=>s+p[0],0)/ring.length, ring.reduce((s,p)=>s+p[1],0)/ring.length];
}}
function nearestTile(latlng){{
  let best=null, bestD=1e15;
  TILES.forEach(t => {{
    const c = centerOf(t);
    if (!c) return;
    const d = map.distance(latlng, c);
    if (d < bestD) {{ bestD=d; best=t; }}
  }});
  return (best && bestD < 70000) ? best : null;
}}
let current = null;
function setCurrent(t){{
  if (current === t) return;
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
TILES.forEach(t => {{
  const href = './' + t.id + '/postfire_landscape_atlas1.html';
  const ring = t.ring;
  const c = centerOf(t);
  if (ring && ring.length) {{
    const poly = L.polygon(ring, {{
      color:'#2450a0', weight:1.5, fillColor:'#2450a0', fillOpacity:0.01, interactive:false
    }});
    poly.addTo(map);
    layers[t.id] = poly;
  }}
  if (c) {{
    const lab = L.marker(c, {{
      icon: L.divIcon({{
        className:'tile-lab',
        html:'<span class="mark"><span class="dot"></span><span class="name">'+t.label+'</span></span>',
        iconSize:[0,0]
      }}),
      keyboard:false, zIndexOffset:600
    }});
    lab.on('click', () => {{ location.href = href; }});
    lab.addTo(map);
  }}
  const a = document.createElement('a');
  a.href = href;
  a.innerHTML = '<span class="id">' + t.label + '</span><span class="sub">' + t.id + '</span>';
  a.onmouseenter = () => setCurrent(t);
  list.appendChild(a);
  listLinks[t.id] = a;
}});
const tip = L.tooltip({{sticky:true, className:'tip'}});
map.on('mousemove', e => {{
  const t = nearestTile(e.latlng);
  setCurrent(t);
  if (t) {{
    tip.setLatLng(e.latlng).setContent(t.label);
    map.openTooltip(tip);
  }} else {{
    map.closeTooltip(tip);
  }}
}});
map.on('click', e => {{
  const t = nearestTile(e.latlng);
  if (t) location.href = './' + t.id + '/postfire_landscape_atlas1.html';
}});
map.on('mouseout', () => {{ setCurrent(null); map.closeTooltip(tip); }});
</script>
</body></html>
"""


def write_atlas_index(
    atlas_root: Path,
    tile_folders: list[str],
    mesma_dir: Path | None = None,
) -> None:
    """Write atlas/index.html as a clickable tile map over the SoCal AOI."""
    atlas_root.mkdir(parents=True, exist_ok=True)
    if mesma_dir is None:
        guess = find_new_data() / "2026-07-28" / "mesma"
        mesma_dir = guess if guess.exists() else None
    entries = []
    n_mapped = 0
    for folder in tile_folders:
        if folder == "test":
            continue
        html = atlas_root / folder / "postfire_landscape_atlas1.html"
        if not html.exists():
            continue
        label, date = tile_label(folder)
        mesma_tile = (mesma_dir / folder) if mesma_dir is not None else None
        ring, center = tile_map_geom(atlas_root / folder, mesma_tile, folder)
        if ring:
            n_mapped += 1
        entries.append({"id": folder, "label": label, "date": date, "ring": ring, "center": center})
    js = "window.ATLAS_TILES=" + json.dumps(entries, separators=(",", ":")) + ";\n"
    (atlas_root / "atlas_tiles.js").write_text(js)
    for folder in tile_folders:
        dest = atlas_root / folder
        if dest.is_dir():
            (dest / "atlas_tiles.js").write_text(js)
    (atlas_root / "index.html").write_text(scene_picker_html(entries))
    print(
        f"Wrote clickable tile map: {atlas_root / 'index.html'} "
        f"({len(entries)} scene(s), {n_mapped} with map squares)",
        flush=True,
    )
    copy_tsf_html(atlas_root)


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
        write_atlas_index(atlas_root, existing, mesma_dir=find_new_data(home) / date / "mesma")
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
    vis_h = min(H, 1600)
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
        html = html_src.read_text(encoding="utf-8", errors="replace")
        html = html.replace(
            '<script src="mesma_grid.js"></script>',
            "<script>" + js_path.read_text(encoding="utf-8") + "</script>",
            1,
        )
        tiles_js = out.parent / "atlas_tiles.js" if out.parent.name == "atlas" else out / "atlas_tiles.js"
        tiles = tiles_js.read_text(encoding="utf-8") if tiles_js.exists() else "window.ATLAS_TILES=[];\n"
        html = html.replace(
            '<script src="atlas_tiles.js" onerror="window.ATLAS_TILES=[]"></script>',
            "<script>" + tiles + "</script>",
            1,
        )
        (out / "postfire_landscape_atlas1.html").write_text(html, encoding="utf-8")
        print("Wrote self-contained HTML to", out, flush=True)
    else:
        print("HTML not found at", html_src, "— copy postfire_landscape_atlas1.html into the out folder.", flush=True)

    if out.parent.name == "atlas":
        folders = [p.name for p in sorted(out.parent.iterdir()) if p.is_dir()]
        write_atlas_index(out.parent, folders, mesma_dir=fractions.parent.parent)

    print("\nDone. Open this folder in the Discovery Environment and download:")
    print(f"  {out}")
    print("Files: postfire_landscape_atlas1.html, mesma_grid.js, mesma_original.png, atlas_tiles.js")
    print("Or open atlas/index.html to pick a scene. Do not download the 6 GB fractions file.")


if __name__ == "__main__":
    main()
