"""Per-listing spatial features computed from each listing's own coordinates.

Version 1 inherited distances from the H3 cell centroid, which about 4.7
listings share on average. Against exact transit distance that gives a median
error of 60 m, a 90th percentile of 137 m and a maximum of 215 m, with 25.9%
of listings off by more than 100 m. At walking scale that is material, so
proximity is now measured per listing.

Densities that are defined over an area (crime, POI counts, quietness, 311)
stay on H3; only nearest-distance measures move to the listing level.
"""

import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Point

from housisense import config

warnings.filterwarnings("ignore", category=UserWarning)

METER_CRS = f"EPSG:{config.METER_CRS}"
HIGHWAY_CLASSES = [1, 2]


def load_listings():
    df = (pd.read_csv(config.LISTINGS_CSV, low_memory=False)
            .dropna(subset=["latitude", "longitude"])
            .copy())
    df["id"] = df["id"].astype(str)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df["longitude"], df["latitude"])],
        crs=f"EPSG:{config.SRID}",
    )
    return gdf.to_crs(METER_CRS)


def nearest_distance(listings, layer_path, out_col, where=None, decimals=3):
    """Distance in km from every listing to the closest feature in a layer.

    `where` is a (column, value_or_list) pair applied to the target layer first.
    Point layers go through a KD-tree; lines and polygons use sjoin_nearest so
    the distance is to the geometry itself rather than to its centroid.
    """
    target = gpd.read_file(layer_path)

    if where is not None:
        col, val = where
        values = val if isinstance(val, (list, tuple, set)) else [val]
        # Shapefile attributes may hold these as int, float or string.
        values = set(values) | {str(v) for v in values} | {
            float(v) for v in values if str(v).replace(".", "").isdigit()
        }
        target = target[target[col].isin(values)]

    target = target.to_crs(METER_CRS)
    if target.empty:
        listings[out_col] = np.nan
        return listings

    if (target.geom_type == "Point").all():
        tree = cKDTree(np.array([(p.x, p.y) for p in target.geometry]))
        distances, _ = tree.query(np.array([(p.x, p.y) for p in listings.geometry]), k=1)
        listings[out_col] = (distances / 1000.0).round(decimals)
    else:
        joined = gpd.sjoin_nearest(listings[["id", "geometry"]], target,
                                   how="left", distance_col="_d")
        nearest = joined.groupby("id")["_d"].min()
        listings[out_col] = (listings["id"].map(nearest) / 1000.0).round(decimals)

    return listings


def areal_value(listings, layer_path, value_col, out_col, agg="mean", decimals=2):
    """Assign each listing the value of the polygon it falls inside."""
    polygons = gpd.read_file(layer_path).to_crs(f"EPSG:{config.SRID}")[[value_col, "geometry"]]
    points = listings.to_crs(f"EPSG:{config.SRID}")[["id", "geometry"]]
    joined = gpd.sjoin(points, polygons, how="left", predicate="within")

    if agg == "first":
        values = joined.drop_duplicates("id").set_index("id")[value_col]
    else:
        values = joined.groupby("id")[value_col].agg(agg)

    mapped = listings["id"].map(values)
    listings[out_col] = mapped.round(decimals) if pd.api.types.is_numeric_dtype(mapped) else mapped
    return listings


def main():
    config.require(config.LISTINGS_CSV, config.TRANSIT_SHP, config.CRIME_GEOJSON,
                   config.PARK_SHP, config.STREETS_SHP, config.NEIGHBORHOOD_SHP)
    config.OUT.mkdir(parents=True, exist_ok=True)

    listings = load_listings()
    print(f"{len(listings):,} listings with coordinates")

    listings = nearest_distance(listings, config.TRANSIT_SHP, "dist_capmetro_km")
    listings = nearest_distance(listings, config.PARK_SHP, "dist_park_km")
    listings = nearest_distance(listings, config.STREETS_SHP, "dist_highway_km",
                                where=("road_class", HIGHWAY_CLASSES))
    listings = areal_value(listings, config.CRIME_GEOJSON, "avg_annual_crime", "crime_block")
    listings = areal_value(listings, config.NEIGHBORHOOD_SHP, "neighname", "neighname",
                           agg="first")

    # Watershed polygons tile the whole city, so nearest-distance to one is zero
    # everywhere and carries no signal. Blue-space proximity needs hydrography.

    keep = ["id", "latitude", "longitude", "neighname"] + [
        c for c in listings.columns if c.startswith(("dist_", "crime_", "n_", "acc_"))
    ]
    listings[keep].to_csv(config.LISTING_FEATURES, index=False)
    print(f"wrote {config.LISTING_FEATURES} with {len(keep) - 4} feature columns")


if __name__ == "__main__":
    main()
