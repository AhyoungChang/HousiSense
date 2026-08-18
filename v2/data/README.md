# Data sources

Everything here is public. Files above roughly 20 MB are excluded from the
repository; the table says where to get them.

| File | Tracked | Source |
| --- | --- | --- |
| `listings.csv` | no, 36 MB | Inside Airbnb, Austin snapshot. Redistribution is restricted, so download it directly. |
| `reviews_with_sentiment.csv` | no, 354 MB | Inside Airbnb reviews for the same snapshot, with a sentiment column added. |
| `Austin_311_Public_Data.csv` | no, 935 MB | City of Austin open data portal, dataset `xwdj-i9he`. `scripts/fetch_311.py` pulls only the noise complaints if you would rather not download the whole export. |
| `streets.*` | no, 238 MB | City of Austin street centrelines. |
| `park.*` | yes | City of Austin park polygons. |
| `neighborhoods.*` | yes | City of Austin neighbourhood boundaries. |
| `CapMetro_Stops.*` | yes | Capital Metro stop locations, EPSG:2277, reprojected on load. |
| `austin_crime_by_census.geojson` | yes | Annual crime counts by census block group. |
| `poi.csv` | yes | Points of interest with `category` and `type` columns. |

Record the snapshot date of the Inside Airbnb extract you use, since the
listing set changes month to month and the quietness field is computed from it.

## Derived files

`out/` holds what the pipeline produces. The small ones are committed so the
validation can be re-run without the large inputs:

- `cell_quietness.csv`, `listing_quietness.csv` from step 5
- `noise_311_h3.csv` from step 1
- `noise_311_points.csv` and `listings_features.csv` are excluded; regenerate
  them with steps 1 and 2.
