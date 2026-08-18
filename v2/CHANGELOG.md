# Changelog

## 2.0.0

Reorganised into an installable package. Every script now imports paths and
connection settings from `housisense/config.py` rather than defining its own,
which removes several filename mismatches between stages.

Where the v1 files went:

| v1 | v2 |
| --- | --- |
| `llm.py` | `housisense/models.py` |
| `quietness_field_pilot.py` | `housisense/quietness.py` and `scripts/06_validate_311.py` |
| `export_quietness.py`, `build_perceptual.py` | `scripts/05_score_quietness.py` |
| `clean_311_noise.py` | `scripts/01_prepare_311.py` |
| `listings_exact_pipeline.py` | `scripts/02_listing_features.py` |
| `postgis_ingest.py` | `scripts/03_ingest_postgis.py` |
| `build_review_index.py` | `scripts/04_index_reviews.py` |
| `app_v2.py` | `housisense/search.py` and `app.py` |
| `housisense_maps.py` | `figures/process_maps.py` |

Fixed along the way:

- The feature table was written as `listings_exact_distances.csv` but read as
  `listings_features.csv`.
- The 311 stage wrote `out/noise_311_h3.csv` while validation expected
  `data/austin_311_noise.csv`.
- The figure script looked for `data/parks.shp`; the file is `park.shp`.
- Quietness scoring was implemented twice, once for CSV export and once for the
  database write, with the two copies free to drift apart.
- The database password and a local model directory were hardcoded. Both now
  come from the environment; see `.env.example`.
