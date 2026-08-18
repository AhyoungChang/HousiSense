"""Embed guest reviews into a pgvector table for semantic ranking.

Creates review_embeddings(listing_id, txt, embedding), which the retrieval
pipeline searches with the cosine distance operator over the spatial candidate
set rather than over the whole corpus.
"""

from sqlalchemy import text

from housisense import config
from housisense.db import engine, run
from housisense.models import embed_batch, to_pgvector

import pandas as pd

BATCH = 256
MAX_CHARS = 1000  # reviews are truncated before embedding
QUOTE_CHARS = 500  # how much of each review is kept for display


def create_table():
    run("CREATE EXTENSION IF NOT EXISTS vector;")
    run("DROP TABLE IF EXISTS review_embeddings;")
    run(f"""
        CREATE TABLE review_embeddings (
            id serial PRIMARY KEY,
            listing_id text,
            txt text,
            embedding vector({config.EMBED_DIM})
        );
    """)


def load_reviews():
    df = pd.read_csv(config.REVIEWS_CSV, low_memory=False)
    df["listing_id"] = df["listing_id"].astype(str)
    df = df[df["comments"].notna()]
    df = df[df["comments"].astype(str).str.strip().astype(bool)]
    return df


def main():
    config.require(config.REVIEWS_CSV)
    create_table()

    reviews = load_reviews()
    texts = reviews["comments"].astype(str).str.slice(0, MAX_CHARS).tolist()
    ids = reviews["listing_id"].tolist()
    print(f"embedding {len(texts):,} reviews")

    insert = text("INSERT INTO review_embeddings (listing_id, txt, embedding) "
                  "VALUES (:lid, :txt, CAST(:emb AS vector))")

    with engine().begin() as con:
        for start in range(0, len(texts), BATCH):
            chunk = texts[start:start + BATCH]
            chunk_ids = ids[start:start + BATCH]
            vectors = embed_batch(chunk, batch_size=BATCH)
            con.execute(insert, [
                {"lid": chunk_ids[i], "txt": chunk[i][:QUOTE_CHARS],
                 "emb": to_pgvector(vectors[i])}
                for i in range(len(chunk))
            ])
            print(f"  {min(start + BATCH, len(texts)):,}/{len(texts):,}")

    run("CREATE INDEX ON review_embeddings USING ivfflat "
        "(embedding vector_cosine_ops) WITH (lists = 100);")
    run("CREATE INDEX review_lid_ix ON review_embeddings (listing_id);")
    run("ANALYZE review_embeddings;")
    print("index built")


if __name__ == "__main__":
    main()
