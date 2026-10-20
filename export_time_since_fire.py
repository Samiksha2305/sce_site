#!/usr/bin/env python3
"""Export MODIS time-since-fire as a GeoTIFF onto the CyVerse data store.

Uses the same MCD64A1 logic as SCE-CU/TimeSinceFire, but writes a raster
the atlas can join (the existing Time_since_fire folder is mostly HTML).

Run in VICE Jupyter after Earth Engine auth:
  python export_time_since_fire.py --end-date 2026-07-28
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import ee


def find_home() -> Path:
    for p in (
        Path("/data-store/iplant/home/samiksha23"),
        Path("/home/jovyan/data-store/iplant/home/samiksha23"),
        Path("/iplant/home/samiksha23"),
    ):
        if p.exists():
            return p
    return Path("/data-store/iplant/home/samiksha23")


def initialize_ee(project_id: str) -> None:
    try:
        ee.Initialize(project=project_id)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project_id)


def compute_date_ranges(start_date: str, end_date: str):
    start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    ranges = []
    year = start.year
    while year <= end.year:
        range_start = start_date if year == start.year else f"{year}-01-01"
        range_end = end_date if year == end.year else f"{year}-12-31"
        ranges.append([range_start, range_end])
        year += 1
    return ranges


def process_modis_data(start_date: str, end_date: str, aoi):
    reference_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    reference_date_millis = int(reference_date.timestamp() * 1000)
    date_ranges = compute_date_ranges(start_date, end_date)

    def yearly_burn(range_start, range_end):
        burned = (
            ee.ImageCollection("MODIS/061/MCD64A1")
            .filter(ee.Filter.date(range_start, range_end))
            .filterBounds(aoi)
            .select("BurnDate")
            .reduce(ee.Reducer.max())
            .rename("latest_burn_date")
            .clip(aoi)
        )
        start_of_year_millis = ee.Date(range_start).millis()
        return burned.expression(
            "burn_date == 0 ? 0 : startOfYear + (burn_date - 1) * 86400000",
            {"burn_date": burned, "startOfYear": start_of_year_millis},
        ).rename("yearly_burn_date").toFloat()

    most_recent = ee.Image(0).rename("most_recent_burn_date").clip(aoi).toFloat()
    for range_start, range_end in date_ranges:
        yb = yearly_burn(range_start, range_end)
        most_recent = most_recent.where(yb.gt(most_recent), yb)

    days = most_recent.expression(
        "burn_date == 0 ? 0 : round((reference_date - burn_date) / (1000 * 60 * 60 * 24))",
        {"reference_date": reference_date_millis, "burn_date": most_recent},
    ).rename("days_since_last_burn").toFloat()
    return days


def export_static_map(image, region, out_dir: Path, end_date: str) -> Path:
    """Download a PNG once so the atlas does not depend on expiring EE tiles."""
    import urllib.request

    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "southern_california_burn_map.png"
    html = out_dir / "southern_california_burn_map.html"
    vis = image.updateMask(image.gt(0))
    url = vis.getThumbURL(
        {
            "min": 1,
            "max": 7300,
            "palette": ["yellow", "#ffd000", "#ff9f00", "#ff6b00", "#f03b20", "#b10026"],
            "region": region,
            "dimensions": 1600,
            "format": "png",
        }
    )
    print("Downloading static burn PNG ...")
    urllib.request.urlretrieve(url, png)
    html.write_text(
        f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"/><meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Days since last burn</title>
<style>
  body{{margin:0;background:#E9DFC1;color:#221F17;font-family:system-ui,sans-serif;}}
  .wrap{{padding:12px 14px 20px;}}
  h1{{font-size:18px;margin:0 0 8px;}}
  p{{font-size:13px;color:#5C5645;margin:0 0 12px;line-height:1.45;}}
  img{{width:100%;height:auto;display:block;border:1px solid #CDBF93;background:#fff;}}
  .leg{{margin-top:12px;display:flex;flex-wrap:wrap;gap:8px 14px;font-size:12px;}}
  .sw{{display:inline-block;width:12px;height:12px;border:1px solid #221F17;margin-right:6px;vertical-align:middle;}}
</style></head><body><div class="wrap">
<h1>Days since last burn — Southern California</h1>
<p>Fixed image from MODIS MCD64A1 through {end_date}. This file does not use live Earth Engine tiles.</p>
<img src="southern_california_burn_map.png" alt="Days since last burn"/>
<div class="leg">
  <span><i class="sw" style="background:yellow"></i>0–1 year</span>
  <span><i class="sw" style="background:#ffd000"></i>1–2 years</span>
  <span><i class="sw" style="background:#ff9f00"></i>2–5 years</span>
  <span><i class="sw" style="background:#ff6b00"></i>5–10 years</span>
  <span><i class="sw" style="background:#f03b20"></i>10–20 years</span>
  <span><i class="sw" style="background:#b10026"></i>&gt;20 years</span>
</div>
</div></body></html>
""",
        encoding="utf-8",
    )
    print(f"Wrote static map {png} ({png.stat().st_size/1e6:.2f} MB) and {html.name}")
    return html
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import geemap

        print("Exporting with geemap.ee_export_image ...")
        geemap.ee_export_image(
            image,
            filename=str(out_path),
            scale=scale,
            region=region,
            crs="EPSG:4326",
            file_per_band=False,
        )
        return
    except Exception as exc:
        print("geemap export failed, trying getDownloadURL:", exc)

    url = image.getDownloadURL(
        {
            "name": out_path.stem,
            "scale": scale,
            "crs": "EPSG:4326",
            "region": region,
            "format": "GEO_TIFF",
        }
    )
    import urllib.request

    print("Downloading", url[:80], "...")
    urllib.request.urlretrieve(url, out_path)


def main():
    home = find_home()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end-date", default="2026-07-28")
    parser.add_argument("--start-date", default="2000-01-01")
    parser.add_argument("--project", default="timesincefire")
    parser.add_argument("--scale", type=int, default=500)
    parser.add_argument(
        "--out",
        default="",
        help="Output GeoTIFF path. Default: Time_since_fire/days_since_last_burn_<end-date>.tif",
    )
    args = parser.parse_args()

    out = Path(args.out) if args.out else (
        home
        / "SCE-CU/sentinel2_data/new_data/Time_since_fire"
        / f"days_since_last_burn_{args.end_date.replace('-', '')}.tif"
    )

    print("Earth Engine project:", args.project)
    initialize_ee(args.project)

    aoi = ee.Geometry.BBox(-122.80, 32.50, -114.75, 37.10)
    print(f"Building TSF through {args.end_date} (MODIS MCD64A1) ...")
    days = process_modis_data(args.start_date, args.end_date, aoi)
    print("Writing", out)
    export_geotiff(days, aoi, out, scale=args.scale)
    export_static_map(days, aoi, out.parent, args.end_date)
    if out.exists():
        print(f"Done: {out} ({out.stat().st_size/1e6:.1f} MB)")
    else:
        raise SystemExit("Export finished but file was not written.")


if __name__ == "__main__":
    main()
