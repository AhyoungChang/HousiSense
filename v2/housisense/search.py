"""Retrieval pipeline over the PostGIS + pgvector store.

A query is parsed into a plan by the language model, turned into a hard spatial
filter in SQL, scored on soft objectives, and finally explained. The model is
used only at the two ends; everything between is deterministic SQL, so the
candidate set is reproducible. Hard constraints use a query-dependent epsilon
rather than a fixed threshold, clamped to pedestrian-catchment priors.
"""

import pandas as pd
from sqlalchemy import text

from .db import engine
from .models import complete, complete_json, embed, to_pgvector

REF_TABLES = ["parks", "transit_stops", "pois", "neighborhoods"]

# Plausible ranges for "near X" in metres, by kind of destination.
EPS_PRIOR = {
    "park": (300, 1500),
    "transit": (200, 900),
    "shop": (200, 800),
    "amenity": (200, 1000),
    "campus": (800, 4000),
    "default": (300, 1500),
}

SOFT_FIELDS = ["quietness", "n_amenity_400", "n_amenity_800", "n_shop_800", "acc_amenity"]

# Columns each reference table may be filtered on. Anything else is dropped,
# which keeps generated SQL both valid and injection-free.
ALLOWED_FILTER = {
    "pois": {"category", "type"},
    "neighborhoods": {"neighname"},
    "parks": {"park_type"},
    "transit_stops": set(),
}

CANDIDATE_LIMIT = 500

PARSE_PROMPT = """You convert a housing query into a JSON plan. Output ONLY JSON.

Reference tables for hard spatial constraints: {refs}
  - parks        : any park (no filter)
  - transit_stops: any bus/transit stop (no filter)
  - neighborhoods: filter {{"neighname": "<name>"}}
  - pois         : filter {{"category": "amenity"}} or {{"category": "shop"}}
                   for a specific kind use {{"type": "restaurant"}} / "cafe" / "bar" / "grocery"
Soft preference fields (on listings): {soft}
  quietness in [-1,1] (+ = quieter); n_*_400/800 = count within band; acc_* = gravity access.

Rules:
- Each HARD constraint = a place that must be reachable: ref table, optional
  "filter" (an object of column->value, ONLY the columns listed above), op
  "dwithin", kind in {kinds}, and eps_m (meters) = how far "near" means for THIS
  query and persona (a car owner's "near" is larger than a no-car student's).
- SOFT = qualitative preferences mapped to those fields with a signed weight
  (+ prefer high, - prefer low): quiet -> {{"quietness":1.0}};
  lively/walkable -> {{"n_amenity_400":1.0}}.
- semantic_query = the free-text intent for review matching.

JSON schema:
{{"hard":[{{"ref":"pois","filter":{{"type":"cafe"}},"op":"dwithin","eps_m":700,"kind":"amenity"}}],
  "soft":{{"quietness":1.0}}, "semantic_query":"..."}}

Query: "{q}"
JSON:"""

EXPLAIN_PROMPT = """Explain in ONE or TWO sentences why this listing fits the request.
Ground every claim in the numbers and the guest quote; do not invent facts.

Request: "{q}"
Listing: {name} | price ${price} | {neigh}
Numbers: quietness={quietness}, amenities_within_400m={n_amenity_400},
         transit_km={dist_capmetro_km}, within the {eps}m buffer requested.
Guest quote: "{quote}"

Explanation:"""

QUIET_WORDS = ["quiet", "light sleeper", "sleep", "peaceful", "calm", "noise", "noisy"]
LIVELY_WORDS = ["lively", "vibrant", "walkable", "lots of", "cafes", "restaurants",
                "amenities", "nightlife", "busy", "plenty"]


def keyword_plan(query):
    """Fallback plan for when the model does not return usable JSON."""
    q = query.lower()
    hard = []

    if "park" in q:
        hard.append({"ref": "parks", "filter": {}, "op": "dwithin", "eps_m": 900, "kind": "park"})
    if any(w in q for w in ["transit", "bus", "metro", "station", "stop", "train"]):
        hard.append({"ref": "transit_stops", "filter": {}, "op": "dwithin",
                     "eps_m": 600, "kind": "transit"})
    if any(w in q for w in ["shop", "store", "grocery", "mall"]):
        hard.append({"ref": "pois", "filter": {"category": "shop"}, "op": "dwithin",
                     "eps_m": 600, "kind": "shop"})

    for keyword, poi_type in [("restaurant", "restaurant"), ("cafe", "cafe"),
                              ("coffee", "cafe"), ("bar", "bar")]:
        if keyword in q:
            hard.append({"ref": "pois", "filter": {"type": poi_type}, "op": "dwithin",
                         "eps_m": 700, "kind": "amenity"})
            break
    else:
        if any(w in q for w in ["amenity", "amenities", "eat", "dining"]):
            hard.append({"ref": "pois", "filter": {"category": "amenity"}, "op": "dwithin",
                         "eps_m": 700, "kind": "amenity"})

    if not hard:
        hard.append({"ref": "parks", "filter": {}, "op": "dwithin", "eps_m": 900, "kind": "park"})

    soft = {}
    if any(w in q for w in QUIET_WORDS):
        soft["quietness"] = 1.0
    if any(w in q for w in LIVELY_WORDS):
        soft["n_amenity_400"] = 1.0

    return {"hard": hard, "soft": soft, "semantic_query": query}


def _sanitize(plan, query):
    for constraint in plan["hard"]:
        low, high = EPS_PRIOR.get(constraint.get("kind", "default"), EPS_PRIOR["default"])
        eps = int(constraint.get("eps_m", low))
        constraint["eps_m"] = int(min(max(eps, low), high))

        if constraint.get("ref") not in REF_TABLES:
            constraint["ref"] = "parks"

        # The model often puts a specific POI kind in "category"; move it to "type".
        if constraint["ref"] == "pois":
            filt = constraint.get("filter") or {}
            category = filt.get("category")
            if category and str(category).lower() not in ("amenity", "shop"):
                filt.pop("category")
                filt["type"] = category
                constraint["filter"] = filt

    plan["soft"] = {k: float(v) for k, v in plan.get("soft", {}).items() if k in SOFT_FIELDS}
    if str(plan.get("semantic_query", "")).strip() in ("", "..."):
        plan["semantic_query"] = query
    return plan


def parse_query(query):
    try:
        plan = complete_json(PARSE_PROMPT.format(
            refs=REF_TABLES, soft=SOFT_FIELDS, kinds=list(EPS_PRIOR), q=query))
        plan.setdefault("hard", [])
        plan.setdefault("soft", {})
        plan.setdefault("semantic_query", query)
        if not isinstance(plan["hard"], list) or not plan["hard"]:
            raise ValueError("plan has no hard constraints")
    except Exception as exc:
        print(f"falling back to keyword plan: {exc}")
        plan = keyword_plan(query)
    return _sanitize(plan, query)


def _count(joins, wheres, params):
    clause = ("WHERE " + " AND ".join(wheres)) if wheres else ""
    sql = text(f"SELECT COUNT(*) FROM listings l {' '.join(joins)} {clause}")
    return int(pd.read_sql(sql, engine(), params=params).iloc[0, 0])


def spatial_candidates(plan):
    """Apply hard constraints one at a time, dropping any that would empty the set.

    Returns the candidate set and the constraints that had to be relaxed, so the
    interface can tell the user which part of their request could not be met.
    """
    joins, wheres, params, relaxed = [], [], {}, []

    for i, constraint in enumerate(plan["hard"]):
        ref = constraint["ref"]
        join = (f"JOIN {ref} r{i} ON ST_DWithin("
                f"l.geometry::geography, r{i}.geometry::geography, :eps{i})")
        clause, local = [], {f"eps{i}": constraint["eps_m"]}

        for col, val in (constraint.get("filter") or {}).items():
            if col in ALLOWED_FILTER.get(ref, set()) and val not in (None, "", "TRUE"):
                key = f"p{i}_{len(clause)}"
                clause.append(f"r{i}.{col} ILIKE :{key}")
                local[key] = f"%{val}%"

        if _count(joins + [join], wheres + clause, {**params, **local}) > 0:
            joins.append(join)
            wheres += clause
            params.update(local)
        else:
            relaxed.append({"ref": ref, "filter": constraint.get("filter"),
                            "eps_m": constraint["eps_m"]})

    if not joins:
        return pd.DataFrame(), relaxed

    clause = ("WHERE " + " AND ".join(wheres)) if wheres else ""
    sql = text(f"""
        SELECT DISTINCT l.id, l.name, l.price, l.neighname,
               l.quietness, l.n_amenity_400, l.n_amenity_800, l.n_shop_800,
               l.acc_amenity, l.dist_capmetro_km, l.crime_block,
               l.latitude, l.longitude
        FROM listings l {' '.join(joins)}
        {clause}
        LIMIT {CANDIDATE_LIMIT}
    """)
    return pd.read_sql(sql, engine(), params=params), relaxed


def semantic_scores(listing_ids, qvec):
    """Best cosine similarity between the query and any review of each listing."""
    if not listing_ids:
        return pd.Series(dtype=float)
    sql = text("SELECT listing_id, MAX(1 - (embedding <=> CAST(:qvec AS vector))) AS sem "
               "FROM review_embeddings WHERE listing_id = ANY(:ids) GROUP BY listing_id")
    df = pd.read_sql(sql, engine(),
                     params={"qvec": to_pgvector(qvec), "ids": list(listing_ids)})
    return df.set_index("listing_id")["sem"]


def best_quote(listing_id, qvec):
    sql = text("SELECT txt FROM review_embeddings WHERE listing_id = :lid "
               "ORDER BY embedding <=> CAST(:qvec AS vector) LIMIT 1")
    rows = pd.read_sql(sql, engine(),
                       params={"lid": str(listing_id), "qvec": to_pgvector(qvec)})
    return rows["txt"].iloc[0] if len(rows) else ""


def _zscore(series):
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return (s - s.mean()) / (s.std() + 1e-9)


def rank(candidates, plan, qvec, top_n=10):
    """Combine semantic similarity with the plan's weighted soft objectives."""
    if candidates.empty:
        return candidates

    ranked = candidates.copy()
    ranked["sem"] = ranked["id"].map(semantic_scores(ranked["id"].tolist(), qvec)).fillna(0.0)

    score = _zscore(ranked["sem"])
    for col, weight in plan["soft"].items():
        if col in ranked:
            score = score + weight * _zscore(ranked[col])

    ranked["score"] = score
    return ranked.sort_values("score", ascending=False).head(top_n)


def explain(row, plan, query, qvec):
    eps = plan["hard"][0]["eps_m"] if plan["hard"] else "n/a"
    amenities = int(row.get("n_amenity_400", 0) or 0)
    try:
        quote = best_quote(row["id"], qvec)
        return complete(EXPLAIN_PROMPT.format(
            q=query, name=row["name"], price=row["price"], neigh=row.get("neighname", ""),
            quietness=row.get("quietness"), n_amenity_400=amenities,
            dist_capmetro_km=row["dist_capmetro_km"], eps=eps, quote=quote[:200]),
            max_new_tokens=120)
    except Exception as exc:
        print(f"explanation model unavailable ({exc}); using the numeric template")
        return (f"quietness {row.get('quietness')}, {amenities} amenities within 400 m, "
                f"transit {row['dist_capmetro_km']} km, inside the {eps} m buffer requested.")


def run(query, top_n=10):
    plan = parse_query(query)
    candidates, relaxed = spatial_candidates(plan)
    qvec = embed(plan.get("semantic_query", query))
    ranked = rank(candidates, plan, qvec, top_n=top_n)
    explanations = [explain(row, plan, query, qvec) for _, row in ranked.iterrows()]
    return plan, ranked, explanations, relaxed
