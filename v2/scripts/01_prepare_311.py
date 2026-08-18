"""Reduce the Austin 311 export to noise complaints, aggregated to H3 res 9.

The export is close to a gigabyte, so it is read in chunks. Socrata column
names change between exports, hence the detection step; override it by passing
--type-col / --lat-col / --lon-col if the guess is wrong.
"""

import argparse
import re

import h3
import numpy as np
import pandas as pd

from housisense import config

NOISE_PATTERN = r"(noise|loud)"
CHUNK = 200_000

# Travis County envelope, used to drop null-island and out-of-area rows.
LAT_RANGE = (30.00, 30.70)
LON_RANGE = (-98.20, -97.40)

CANDIDATES = {
    "type": [r"sr.*type.*desc", r"complaint.*type", r"sr_type", r"type.*desc",
             r"\btype\b", r"description"],
    "lat": [r"^lat", r"latitude", r"\by_?coord"],
    "lon": [r"^lon", r"^lng", r"longitude", r"\bx_?coord"],
    "date": [r"created.*date", r"sr_created", r"open.*date", r"\bdate\b"],
    "location": [r"location", r"\bpoint\b", r"geocode"],
}

LATLON_PATTERN = re.compile(r"(-?\d+\.\d+)[,\s]+(-?\d+\.\d+)")


def detect(columns, key):
    lowered = {c: c.lower() for c in columns}
    for pattern in CANDIDATES[key]:
        for original, lower in lowered.items():
            if re.search(pattern, lower):
                return original
    return None


def parse_location(series):
    """Pull coordinates out of a combined '(30.26, -97.74)' column."""
    lat, lon = [], []
    for value in series.astype(str):
        match = LATLON_PATTERN.search(value)
        if match:
            lat.append(float(match.group(1)))
            lon.append(float(match.group(2)))
        else:
            lat.append(np.nan)
            lon.append(np.nan)
    return pd.Series(lat), pd.Series(lon)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(config.RAW_311_CSV))
    ap.add_argument("--type-col")
    ap.add_argument("--lat-col")
    ap.add_argument("--lon-col")
    ap.add_argument("--date-col")
    args = ap.parse_args()

    config.require(args.input)
    config.OUT.mkdir(parents=True, exist_ok=True)

    header = pd.read_csv(args.input, nrows=5, dtype=str, low_memory=False)
    columns = list(header.columns)
    type_col = args.type_col or detect(columns, "type")
    lat_col = args.lat_col or detect(columns, "lat")
    lon_col = args.lon_col or detect(columns, "lon")
    date_col = args.date_col or detect(columns, "date")
    loc_col = detect(columns, "location")

    print(f"type column: {type_col}")
    print(f"coordinates: {lat_col} / {lon_col}" if lat_col and lon_col
          else f"coordinates: parsed from {loc_col}")

    if type_col is None:
        raise SystemExit("No service-request type column found; pass --type-col.")
    if (lat_col is None or lon_col is None) and loc_col is None:
        raise SystemExit("No coordinates found; pass --lat-col and --lon-col.")

    pattern = re.compile(NOISE_PATTERN, re.IGNORECASE)
    kept = []
    scanned = matched = 0

    for chunk in pd.read_csv(args.input, chunksize=CHUNK, dtype=str, low_memory=False):
        scanned += len(chunk)
        subset = chunk[chunk[type_col].astype(str).str.contains(pattern, na=False)].copy()
        matched += len(subset)
        if subset.empty:
            continue

        if lat_col and lon_col:
            subset["latitude"] = pd.to_numeric(subset[lat_col], errors="coerce")
            subset["longitude"] = pd.to_numeric(subset[lon_col], errors="coerce")
        else:
            lat, lon = parse_location(subset[loc_col])
            subset["latitude"], subset["longitude"] = lat.values, lon.values

        subset["sr_type"] = subset[type_col]
        subset["created_date"] = subset[date_col] if date_col else ""
        kept.append(subset[["latitude", "longitude", "sr_type", "created_date"]])

    if not kept:
        raise SystemExit("No rows matched the noise filter; check the type column.")

    df = pd.concat(kept, ignore_index=True)
    print(f"\nscanned {scanned:,} rows, matched {matched:,}")
    print("service-request types kept:")
    print(df["sr_type"].value_counts().head(20).to_string())

    df = df.dropna(subset=["latitude", "longitude"])
    inside = (df.latitude.between(*LAT_RANGE) & df.longitude.between(*LON_RANGE))
    print(f"\n{len(df[inside]):,} complaints inside the study area "
          f"({(~inside).sum():,} dropped)")
    df = df[inside].copy()

    df["h3"] = [h3.latlng_to_cell(a, o, config.H3_RES)
                for a, o in zip(df.latitude, df.longitude)]
    df.to_csv(config.NOISE_POINTS, index=False)

    cells = df.groupby("h3").size().rename("noise_311").reset_index()
    cells.to_csv(config.NOISE_H3, index=False)
    print(f"\nwrote {config.NOISE_POINTS.name} ({len(df):,} rows) "
          f"and {config.NOISE_H3.name} ({len(cells):,} cells)")


if __name__ == "__main__":
    main()
