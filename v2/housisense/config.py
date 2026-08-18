import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.getenv("HOUSISENSE_DATA", ROOT / "data"))
OUT = Path(os.getenv("HOUSISENSE_OUT", ROOT / "out"))
FIGS = Path(os.getenv("HOUSISENSE_FIGS", ROOT / "figures"))

DB_URL = os.getenv(
    "HOUSISENSE_DB_URL",
    "postgresql+psycopg2://postgres:postgres@localhost:5432/housisense",
)

EMBED_MODEL = os.getenv("HOUSISENSE_EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2")
EMBED_DIM = 768
GEN_MODEL = os.getenv("HOUSISENSE_GEN_MODEL", "HuggingFaceH4/zephyr-7b-beta")

SRID = 4326
METER_CRS = 32614  # UTM 14N covers Travis County
H3_RES = 9

# Source layers. The large ones are not tracked in git; see scripts/fetch_data.py.
LISTINGS_CSV = DATA / "listings.csv"
REVIEWS_CSV = DATA / "reviews_with_sentiment.csv"
RAW_311_CSV = DATA / "Austin_311_Public_Data.csv"
PARK_SHP = DATA / "park.shp"
STREETS_SHP = DATA / "streets.shp"
TRANSIT_SHP = DATA / "CapMetro_Stops.shp"
NEIGHBORHOOD_SHP = DATA / "neighborhoods.shp"
CRIME_GEOJSON = DATA / "austin_crime_by_census.geojson"
POI_CSV = DATA / "poi.csv"

# Derived tables written by scripts/ and consumed downstream.
LISTING_FEATURES = OUT / "listings_features.csv"
NOISE_POINTS = OUT / "noise_311_points.csv"
NOISE_H3 = OUT / "noise_311_h3.csv"
LISTING_QUIETNESS = OUT / "listing_quietness.csv"
CELL_QUIETNESS = OUT / "cell_quietness.csv"
VALIDATION_REPORT = OUT / "validation_311.txt"


def require(*paths):
    """Exit with a readable message instead of a stack trace on missing inputs."""
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise SystemExit(
            "Missing input files:\n  " + "\n  ".join(missing)
            + "\nSee README.md for how to obtain them."
        )
