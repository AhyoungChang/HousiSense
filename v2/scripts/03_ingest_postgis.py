"""Load listings and the raw geometry layers into PostGIS.

Geometry is stored in EPSG:4326 and cast to ::geography at query time, so
ST_DWithin and ST_Distance return metres without reprojection. This is what
lets "near X" be resolved with a query-dependent epsilon instead of the fixed
250 m threshold used in version 1.
"""

import argparse
import warnings

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from housisense import config
from housisense.db import engine, run

warnings.filterwarnings("ignore", category=UserWarning)

LISTING_COLUMNS = [
    "id", "name", "price", "room_type", "accommodates", "bedrooms",
    "latitude", "longitude", "neighname",
    "dist_capmetro_km", "dist_park_km", "dist_amenity_km",
    "dist_shop_km", "dist_highway_km",
    "n_amenity_400", "n_amenity_800", "n_shop_400", "n_shop_800",
    "acc_amenity", "acc_shop", "crime_block", "geometry",
]


def to_wgs84(gdf):
    if gdf.crs is None:
        return gdf.set_crs(epsg=config.SRID, allow_override=True)
    return gdf.to_crs(epsg=config.SRID) if gdf.crs.to_epsg() != config.SRID else gdf


def points_from_csv(path, lat="latitude", lon="longitude"):
    df = pd.read_csv(path).dropna(subset=[lat, lon])
    return gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df[lon], df[lat])],
        crs=f"EPSG:{config.SRID}",
    )


def load_listings():
    base = pd.read_csv(config.LISTINGS_CSV, low_memory=False)
    base["id"] = base["id"].astype(str)

    features = pd.read_csv(config.LISTING_FEATURES)
    features["id"] = features["id"].astype(str)
    feature_cols = ["id"] + [
        c for c in features.columns
        if c.startswith(("dist_", "crime_", "n_", "acc_")) or c == "neighname"
    ]

    merged = (base.merge(features[feature_cols], on="id", how="left")
                  .dropna(subset=["latitude", "longitude"]))
    gdf = gpd.GeoDataFrame(
        merged,
        geometry=[Point(xy) for xy in zip(merged["longitude"], merged["latitude"])],
        crs=f"EPSG:{config.SRID}",
    )
    return gdf[[c for c in LISTING_COLUMNS if c in gdf.columns]]


def build_layers():
    return {
        "listings": load_listings(),
        "parks": to_wgs84(gpd.read_file(config.PARK_SHP))[["objectid", "park_type", "geometry"]],
        "streets": to_wgs84(gpd.read_file(config.STREETS_SHP))[
            ["objectid", "road_class", "full_stree", "geometry"]],
        "transit_stops": to_wgs84(gpd.read_file(config.TRANSIT_SHP))[
            ["STOP_ID", "STOP_NAME", "geometry"]],
        "pois": points_from_csv(config.POI_CSV)[["name", "category", "type", "geometry"]],
        "neighborhoods": to_wgs84(gpd.read_file(config.NEIGHBORHOOD_SHP))[
            ["neighname", "geometry"]],
        "crime_blocks": to_wgs84(gpd.read_file(config.CRIME_GEOJSON))[
            ["geoid", "avg_annual_crime", "geometry"]],
    }


def demo():
    """Show that only the epsilon parameter changes between reach definitions."""
    sql = """
        SELECT COUNT(DISTINCT l.id) AS n
        FROM listings l
        JOIN parks p ON ST_DWithin(l.geometry::geography, p.geometry::geography, %(eps)s)
    """
    print("\nlistings within reach of a park, by epsilon:")
    for eps in (400, 1000, 2000):
        n = pd.read_sql(sql, engine(), params={"eps": eps}).iloc[0, 0]
        print(f"  {eps:>5} m  {n:>6,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true",
                    help="after loading, count candidates at three epsilon values")
    args = ap.parse_args()

    config.require(config.LISTINGS_CSV, config.LISTING_FEATURES, config.PARK_SHP,
                   config.STREETS_SHP, config.TRANSIT_SHP, config.POI_CSV,
                   config.NEIGHBORHOOD_SHP, config.CRIME_GEOJSON)

    run("CREATE EXTENSION IF NOT EXISTS postgis;")

    for name, gdf in build_layers().items():
        print(f"{name:<14} {len(gdf):>7,} rows")
        gdf.to_postgis(name, engine(), if_exists="replace", index=False)
        run(f"CREATE INDEX IF NOT EXISTS {name}_gix ON {name} USING GIST (geometry);")

    run("CREATE INDEX IF NOT EXISTS listings_id_ix ON listings (id);")
    print("spatial indexes built")

    if args.demo:
        demo()


if __name__ == "__main__":
    main()
