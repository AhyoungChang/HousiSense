"""Compute the perceived quietness field at listing and cell level.

Writes two CSVs that are small enough to keep in the repository, and by default
also pushes the listing-level score into PostGIS as a soft objective. Pass
--no-db to run the scoring without a database, which is all that is needed to
reproduce the validation in step 6.
"""

import argparse

import pandas as pd

from housisense import config, quietness
from housisense.db import engine, run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-db", action="store_true",
                    help="write the CSVs only, skip the PostGIS update")
    args = ap.parse_args()

    config.require(config.LISTINGS_CSV, config.REVIEWS_CSV)
    config.OUT.mkdir(parents=True, exist_ok=True)

    listings = pd.read_csv(config.LISTINGS_CSV, low_memory=False)
    listing_to_cell = quietness.listing_cell_index(listings, config.H3_RES)

    reviews = pd.read_csv(config.REVIEWS_CSV, low_memory=False)
    reviews["listing_id"] = reviews["listing_id"].astype(str)
    print(f"scoring the quietness aspect over {len(reviews):,} reviews")

    scored = quietness.score_reviews(reviews)
    n_aspect = scored["q"].notna().sum()
    print(f"  aspect-bearing reviews: {n_aspect:,} "
          f"({100 * n_aspect / max(len(scored), 1):.1f}%)")

    listing_level = quietness.per_listing(scored, listing_to_cell)
    listing_level[["id", "quietness", "q_raw", "n_aspect", "h3"]].to_csv(
        config.LISTING_QUIETNESS, index=False)
    print(f"  {len(listing_level):,} listings -> {config.LISTING_QUIETNESS.name}")

    cell_level = quietness.per_cell(scored, listing_to_cell)
    cell_level.to_csv(config.CELL_QUIETNESS, index=False)
    print(f"  {len(cell_level):,} cells -> {config.CELL_QUIETNESS.name}")

    if args.no_db:
        return

    listing_level[["id", "quietness"]].to_sql(
        "quietness_stage", engine(), if_exists="replace", index=False)
    run("ALTER TABLE listings ADD COLUMN IF NOT EXISTS quietness double precision;")
    run("UPDATE listings l SET quietness = q.quietness "
        "FROM quietness_stage q WHERE l.id = q.id;")
    run("DROP TABLE quietness_stage;")
    print("listings.quietness updated")


if __name__ == "__main__":
    main()
