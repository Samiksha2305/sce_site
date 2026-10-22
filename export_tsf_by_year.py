#!/usr/bin/env python3
"""Time-since-fire at ignition for Southern California fires, 2015–2026.

For each fire year Y:

1. Keep only burns in year Y (MODIS pixels + MTBS named perimeters).
2. Assign an exact burn date: MODIS MCD64A1 BurnDate (day of year) or
   MTBS Ig_Date when the feature is a named fire.
3. Compute TSF as of 31 Dec (Y-1) from the FULL MODIS history
   (2000-01-01 through that December date). Do not build TSF from the
   year-Y file alone — those polygons had not burned yet.
4. Add (fire_date − 31 Dec Y-1) so the result is TSF at ignition
   (days since the previous fire when this one started).

Writes under Time_since_fire/by_year_2015_2026/:

  {year}_modis_burn_date.tif      pixel fire date as year*1000+DOY (2020158 = 2020-06-06)
  {year}_tsf_as_of_dec_prev.tif   TSF on 31 Dec (Y-1), masked to year Y
  {year}_days_from_dec_to_fire.tif
  {year}_tsf_at_ignition.tif
  {year}_modis_perimeters.geojson
  {year}_mtbs_perimeters.geojson  named fires with Ig_Date + TSF fields
  fires.geojson                   all years combined (MTBS + MODIS)
  tsf_at_ignition_2015_2026.tif   mosaic (later year wins on overlap)

Run on CyVerse VICE after Earth Engine auth:

  python export_tsf_by_year.py
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import urllib.request
from pathlib import Path

import ee

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_time_since_fire import find_home, initialize_ee


AOI_WEST, AOI_SOUTH, AOI_EAST, AOI_NORTH = -122.80, 32.50, -114.75, 37.10
HISTORY_START = "2000-01-01"
MTBS_ID = "USFS/GTAC/MTBS/burned_area_boundaries/v1"
MS_PER_DAY = 86400000


def aoi_box() -> ee.Geometry:
    return ee.Geometry.BBox(AOI_WEST, AOI_SOUTH, AOI_EAST, AOI_NORTH)


def year_ranges(start: str, end: str) -> list[list[str]]:
    first = datetime.datetime.strptime(start, "%Y-%m-%d")
    last = datetime.datetime.strptime(end, "%Y-%m-%d")
    ranges = []
    year = first.year
    while year <= last.year:
        ranges.append(
            [
                start if year == first.year else f"{year}-01-01",
                end if year == last.year else f"{year}-12-31",
            ]
        )
        year += 1
    return ranges


def yearly_burn_millis(range_start: str, range_end: str, aoi: ee.Geometry) -> ee.Image:
    burned = (
        ee.ImageCollection("MODIS/061/MCD64A1")
        .filterDate(range_start, range_end)
        .filterBounds(aoi)
        .select("BurnDate")
        .reduce(ee.Reducer.max())
        .rename("doy")
        .clip(aoi)
    )
    start_millis = ee.Date(range_start).millis()
    return burned.expression(
        "doy > 0 ? startOfYear + (doy - 1) * 86400000 : 0",
        {"doy": burned, "startOfYear": start_millis},
    ).rename("burn_millis").toFloat()


def most_recent_burn_millis(start: str, end: str, aoi: ee.Geometry) -> ee.Image:
    latest = ee.Image(0).rename("burn_millis").clip(aoi).toFloat()
    for range_start, range_end in year_ranges(start, end):
        yearly = yearly_burn_millis(range_start, range_end, aoi)
        latest = latest.where(yearly.gt(latest), yearly)
    return latest


def days_since(burn_millis: ee.Image, as_of: str) -> ee.Image:
    as_of_ms = ee.Date(as_of).millis()
    return burn_millis.expression(
        "burn == 0 ? 0 : round((asOf - burn) / 86400000)",
        {"burn": burn_millis, "asOf": as_of_ms},
    ).rename("days").toFloat()


def fire_date_yyyymmdd(year: int, aoi: ee.Geometry) -> tuple[ee.Image, ee.Image]:
    """MODIS BurnDate (1–366) → year*1000+DOY and burn millis for year Y."""
    doy = (
        ee.ImageCollection("MODIS/061/MCD64A1")
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .filterBounds(aoi)
        .select("BurnDate")
        .reduce(ee.Reducer.max())
        .rename("doy")
        .clip(aoi)
    )
    jan1 = ee.Date(f"{year}-01-01")
    # day-of-year 1 = Jan 1. Convert to month/day with a calendar image.
    millis = doy.expression(
        "doy > 0 ? jan1 + (doy - 1) * 86400000 : 0",
        {"doy": doy, "jan1": jan1.millis()},
    )
    # YYYYMMDD via unix date: use ee.Image.from millis is not built-in;
    # encode as year * 1000 + doy (e.g. 2020158 = 2020-06-06).
    coded = doy.expression(
        "doy > 0 ? year * 1000 + doy : 0",
        {"doy": doy, "year": year},
    ).rename("burn_yyyydoy").toFloat()
    return coded, millis.rename("burn_millis").toFloat()


def export_image(image: ee.Image, out_path: Path, region: ee.Geometry, scale: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import geemap

        print(f"  exporting {out_path.name} ...", flush=True)
        geemap.ee_export_image(
            image,
            filename=str(out_path),
            scale=scale,
            region=region,
            crs="EPSG:4326",
            file_per_band=False,
        )
        if out_path.exists():
            return
    except Exception as exc:
        print(f"  geemap export failed ({exc}); trying getDownloadURL", flush=True)

    url = image.getDownloadURL(
        {
            "name": out_path.stem,
            "scale": scale,
            "crs": "EPSG:4326",
            "region": region,
            "format": "GEO_TIFF",
        }
    )
    print(f"  downloading {out_path.name} ...", flush=True)
    urllib.request.urlretrieve(url, out_path)


def export_geojson(fc: ee.FeatureCollection, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = int(fc.size().getInfo() or 0)
    if n == 0:
        out_path.write_text(json.dumps({"type": "FeatureCollection", "features": []}))
        return 0
    try:
        import geemap

        geemap.ee_export_vector(fc, filename=str(out_path), file_format="GEOJSON")
        if out_path.exists():
            return n
    except Exception as exc:
        print(f"  geemap vector export failed ({exc}); trying getDownloadURL", flush=True)
    url = fc.getDownloadURL("GEOJSON")
    urllib.request.urlretrieve(url, out_path)
    return n


def mtbs_year(year: int, aoi: ee.Geometry) -> ee.FeatureCollection:
    start = ee.Date(f"{year}-01-01").millis()
    stop = ee.Date(f"{year + 1}-01-01").millis()
    return (
        ee.FeatureCollection(MTBS_ID)
        .filterBounds(aoi)
        .filter(ee.Filter.gte("Ig_Date", start))
        .filter(ee.Filter.lt("Ig_Date", stop))
    )


def attach_tsf_to_fires(
    fires: ee.FeatureCollection,
    tsf_dec: ee.Image,
    as_of: str,
    year: int,
    scale: int,
) -> ee.FeatureCollection:
    as_of_ms = ee.Date(as_of).millis()

    def add_fields(feat: ee.Feature) -> ee.Feature:
        ig = ee.Number(feat.get("Ig_Date"))
        offset = ig.subtract(as_of_ms).divide(MS_PER_DAY).round()
        stats = tsf_dec.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=feat.geometry(),
            scale=scale,
            maxPixels=1e7,
            bestEffort=True,
        )
        tsf_mean = ee.Number(ee.Algorithms.If(stats.get("days"), stats.get("days"), 0))
        return feat.set(
            {
                "burn_year": year,
                "burn_date": ee.Date(ig).format("YYYY-MM-dd"),
                "days_from_dec_prev": offset,
                "tsf_as_of_dec_prev": tsf_mean,
                "tsf_at_ignition": tsf_mean.add(offset),
                "as_of_date": as_of,
            }
        )

    return fires.map(add_fields)


def process_year(year: int, aoi: ee.Geometry, out_dir: Path, scale: int) -> tuple[ee.FeatureCollection, ee.Image, ee.FeatureCollection]:
    as_of = f"{year - 1}-12-31"
    print(f"\n=== {year}  TSF as of {as_of} ===", flush=True)

    prior = most_recent_burn_millis(HISTORY_START, as_of, aoi)
    tsf_dec = days_since(prior, as_of).rename("days")
    coded, fire_millis = fire_date_yyyymmdd(year, aoi)
    year_mask = fire_millis.gt(0)
    offset = fire_millis.expression(
        "fire == 0 ? 0 : round((fire - dec) / 86400000)",
        {"fire": fire_millis, "dec": ee.Date(as_of).millis()},
    ).rename("days_from_dec").toFloat()
    tsf_ign = (
        tsf_dec.add(offset)
        .updateMask(year_mask.And(prior.gt(0)))
        .rename("tsf_at_ignition")
        .toFloat()
    )
    tsf_dec_masked = tsf_dec.updateMask(year_mask)
    offset_masked = offset.updateMask(year_mask)
    coded_masked = coded.updateMask(year_mask)

    export_image(coded_masked, out_dir / f"{year}_modis_burn_date.tif", aoi, scale)
    export_image(tsf_dec_masked, out_dir / f"{year}_tsf_as_of_dec_prev.tif", aoi, scale)
    export_image(offset_masked, out_dir / f"{year}_days_from_dec_to_fire.tif", aoi, scale)
    export_image(tsf_ign, out_dir / f"{year}_tsf_at_ignition.tif", aoi, scale)

    vectors = year_mask.selfMask().reduceToVectors(
        geometry=aoi,
        scale=scale,
        geometryType="polygon",
        eightConnected=True,
        labelProperty="burned",
        maxPixels=1e9,
        bestEffort=True,
    ).map(
        lambda f: f.set(
            {
                "burn_year": year,
                "as_of_date": as_of,
                "date_source": "MODIS_MCD64A1_BurnDate",
            }
        )
    )
    n_modis = export_geojson(vectors, out_dir / f"{year}_modis_perimeters.geojson")
    print(f"  MODIS year-{year} polygons: {n_modis}", flush=True)

    mtbs = attach_tsf_to_fires(mtbs_year(year, aoi), tsf_dec, as_of, year, scale)
    n_mtbs = export_geojson(mtbs, out_dir / f"{year}_mtbs_perimeters.geojson")
    print(f"  MTBS named fires: {n_mtbs}", flush=True)
    return mtbs, tsf_ign, vectors


def mosaic_years(year_images: list[ee.Image]) -> ee.Image:
    """Later year wins where fires overlap (images must be in year order)."""
    return ee.ImageCollection(year_images).mosaic().rename("tsf_at_ignition").toFloat()


def write_readme(out_dir: Path, years: list[int]) -> None:
    text = f"""# TSF at ignition, {years[0]}–{years[-1]}

Each year-Y folder pair uses this formula:

    TSF_at_ignition = TSF_as_of_31_Dec_(Y-1) + (fire_date − 31 Dec Y-1)

TSF_as_of_31_Dec_(Y-1) is computed from the full MODIS MCD64A1 record
(2000 through that December date), then masked to year-Y burns.

`{{year}}_modis_burn_date.tif` stores year*1000 + day-of-year
(e.g. 2020158 = 2020-06-06). MTBS GeoJSON has ISO `burn_date` from Ig_Date.

Pixels with no earlier MODIS burn (first recorded fire) are nodata in
`tsf_at_ignition` so a missing history is not treated as 0 days.
"""
    (out_dir / "README.txt").write_text(text)


def main() -> None:
    home = find_home()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--project", default="timesincefire")
    parser.add_argument("--scale", type=int, default=500)
    parser.add_argument(
        "--out",
        default="",
        help="Output folder. Default: Time_since_fire/by_year_2015_2026",
    )
    args = parser.parse_args()

    years = list(range(args.start_year, args.end_year + 1))
    out = Path(args.out) if args.out else (
        home / "SCE-CU/sentinel2_data/new_data/Time_since_fire" / "by_year_2015_2026"
    )
    out.mkdir(parents=True, exist_ok=True)

    print("Earth Engine project:", args.project, flush=True)
    initialize_ee(args.project)
    aoi = aoi_box()
    print(f"Years {years[0]}–{years[-1]}  →  {out}", flush=True)

    mtbs_all = ee.FeatureCollection([])
    modis_all = ee.FeatureCollection([])
    ign_images: list[ee.Image] = []
    for year in years:
        mtbs, ign, vectors = process_year(year, aoi, out, args.scale)
        mtbs_all = mtbs_all.merge(mtbs)
        modis_all = modis_all.merge(vectors)
        ign_images.append(ign)

    combined = mtbs_all.merge(modis_all)
    n_all = export_geojson(combined, out / f"fires_{years[0]}_{years[-1]}.geojson")
    print(f"\nCombined fires: {n_all}", flush=True)
    raw = out / f"fires_{years[0]}_{years[-1]}.geojson"
    keep = (
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
    slim = []
    seen = set()
    extras = [
        raw,
        out / "fires_web.geojson",
        out.parent / "by_year_2015_2020" / "fires_2015_2020.geojson",
        out.parent / "by_year_2015_2020" / "fires_web.geojson",
    ]
    for src in extras:
        if not src.exists():
            continue
        for feat in (json.loads(src.read_text()).get("features") or []):
            props = feat.get("properties") or {}
            geom = feat.get("geometry")
            if not geom:
                continue
            key = (
                props.get("Incid_Name"),
                props.get("burn_year"),
                props.get("burn_date"),
                props.get("date_source"),
            )
            if key in seen:
                continue
            seen.add(key)
            slim.append(
                {
                    "type": "Feature",
                    "properties": {k: props[k] for k in keep if k in props},
                    "geometry": geom,
                }
            )
    web = out / "fires.geojson"
    web.write_text(json.dumps({"type": "FeatureCollection", "features": slim}, separators=(",", ":")))
    (out / "fires_web.geojson").write_text(web.read_text())
    print(f"Wrote web GeoJSON: {web} ({len(slim)} fires)", flush=True)
    mosaic = mosaic_years(ign_images)
    export_image(mosaic, out / f"tsf_at_ignition_{years[0]}_{years[-1]}.tif", aoi, args.scale)
    write_readme(out, years)
    print(f"Done. Files in {out}", flush=True)


if __name__ == "__main__":
    main()
