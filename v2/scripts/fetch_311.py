"""Download Austin 311 noise complaints through the SODA API.

An alternative to step 1 for anyone who does not want the full export. The
filter runs server side, so this pulls tens of thousands of rows instead of the
several million in the bulk download. Results are written in the same format
step 1 produces, so step 6 can be run straight afterwards.

An app token is not required but raises the rate limit; set SOCRATA_APP_TOKEN
if you have one.
"""

import argparse
import os
import time

import h3
import pandas as pd
import requests

from housisense import config

DATASET = "xwdj-i9he"  # Austin 311 Public Data
ENDPOINT = f"https://data.austintexas.gov/resource/{DATASET}.json"
PAGE = 50_000

LAT_RANGE = (30.00, 30.70)
LON_RANGE = (-98.20, -97.40)


def fetch_page(offset, token):
    params = {
        "$select": "sr_type_desc,sr_created_date,latitude,longitude",
        "$where": "lower(sr_type_desc) like '%noise%' or lower(sr_type_desc) like '%loud%'",
        "$limit": PAGE,
        "$offset": offset,
    }
    headers = {"X-App-Token": token} if token else {}
    response = requests.get(ENDPOINT, params=params, headers=headers, timeout=120)
    response.raise_for_status()
    return response.json()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=500_000)
    args = ap.parse_args()

    token = os.getenv("SOCRATA_APP_TOKEN")
    config.OUT.mkdir(parents=True, exist_ok=True)

    rows, offset = [], 0
    while offset < args.max_rows:
        page = fetch_page(offset, token)
        if not page:
            break
        rows.extend(page)
        offset += PAGE
        print(f"  {len(rows):,} rows")
        time.sleep(0.5)

    if not rows:
        raise SystemExit("The API returned no rows; the dataset schema may have changed.")

    df = pd.DataFrame(rows).rename(columns={
        "sr_type_desc": "sr_type", "sr_created_date": "created_date"})
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])
    df = df[df.latitude.between(*LAT_RANGE) & df.longitude.between(*LON_RANGE)].copy()

    df["h3"] = [h3.latlng_to_cell(a, o, config.H3_RES)
                for a, o in zip(df.latitude, df.longitude)]
    df[["latitude", "longitude", "sr_type", "created_date", "h3"]].to_csv(
        config.NOISE_POINTS, index=False)

    cells = df.groupby("h3").size().rename("noise_311").reset_index()
    cells.to_csv(config.NOISE_H3, index=False)
    print(f"{len(df):,} complaints across {len(cells):,} cells")


if __name__ == "__main__":
    main()
