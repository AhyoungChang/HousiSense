# HousiSense v2

Retrieval over housing listings that combines exact spatial constraints with a
perceived quietness field derived from guest reviews.

Version 1 answered spatial questions with fixed distance thresholds and
cell-centroid geometry. Version 2 changes three things:

- **Query-dependent proximity.** What counts as "near" is set per query and
  clamped to pedestrian-catchment priors, rather than fixed at 250 m for
  everything.
- **Listing-level geometry.** Distances are measured from each listing's own
  coordinates. Under the old cell-centroid scheme roughly 4.7 listings shared a
  centroid, giving a median transit-distance error of 60 m and a 90th
  percentile of 137 m.
- **Perception as a retrievable field.** Aspect-based sentiment over
  noise-related sentences produces a quietness score in [-1, 1], smoothed
  toward the neighbourhood mean, and used as a soft ranking objective.

The quietness field is validated against City of Austin 311 noise complaints,
an independent municipal record that never enters the model.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -r requirements.txt

cp .env.example .env    # then edit and export the variables
```

PostGIS with pgvector, via Docker:

```bash
docker run --name housisense-pg -e POSTGRES_PASSWORD=CHANGE_ME \
    -e POSTGRES_DB=housisense -p 5432:5432 -d pgvector/pgvector:pg16
```

## Pipeline

Run in order. Steps 1, 2, 5 and 6 need no database.

| Step | Script | Produces |
| --- | --- | --- |
| 1 | `scripts/01_prepare_311.py` | `out/noise_311_points.csv`, `out/noise_311_h3.csv` |
| 2 | `scripts/02_listing_features.py` | `out/listings_features.csv` |
| 3 | `scripts/03_ingest_postgis.py` | PostGIS tables and GIST indexes |
| 4 | `scripts/04_index_reviews.py` | `review_embeddings` with an ivfflat index |
| 5 | `scripts/05_score_quietness.py` | `out/listing_quietness.csv`, `out/cell_quietness.csv` |
| 6 | `scripts/06_validate_311.py` | `out/validation_311.txt` |

Then:

```bash
streamlit run app.py
python figures/process_maps.py
```

`scripts/fetch_311.py` is an alternative to step 1 that pulls only noise
complaints through the Socrata API, avoiding the full export.

## Reproducing the validation without the large files

Steps 5 and 6 are the ones the paper's external validation rests on, and the
inputs they need are the two review-derived CSVs plus the 311 counts. If you
only want that result:

```bash
python scripts/fetch_311.py
python scripts/05_score_quietness.py --no-db
python scripts/06_validate_311.py
```

The committed `out/cell_quietness.csv` and `out/noise_311_h3.csv` let you skip
straight to step 6.

## Data

Files under about 20 MB are committed. The rest are not; see `data/README.md`
for each source and how to obtain it. `figures/process_maps.py --synthetic`
runs on generated data if you want to exercise the figure code first.

## Layout

```
housisense/       library code: config, models, quietness scorer, retrieval
scripts/          numbered pipeline steps
figures/          figure generation
notebooks/        exploratory analysis
data/             source layers
out/              derived tables
```

## Citation

Chang, A., & Jiao, J. HousiSense: A spatial-cognitive RAG agent for explainable
housing reasoning. Manuscript in preparation.
