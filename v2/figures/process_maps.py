"""Two figures showing how a query is resolved on the map.

Figure A follows the hard spatial filter: every listing, then those within
500 m of a restaurant, then the intersection with a 300 m buffer around
downtown. Figure B takes the surviving candidates and separates them on the
competing soft axes before combining them.

The geometry mirrors the PostGIS pipeline in geopandas, so the figures can be
regenerated without a database: ST_DWithin(x, eps) becomes buffer(eps) followed
by intersects().

    python figures/process_maps.py
    python figures/process_maps.py --basemap     # needs contextily and internet
    python figures/process_maps.py --synthetic   # no input files required
"""

import argparse
import os
import sys
import warnings

import geopandas as gpd
import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from shapely.geometry import box  # noqa: E402

from housisense import config  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

LAYERS = {
    "pois": config.POI_CSV,
    "neighborhoods": config.NEIGHBORHOOD_SHP,
    "streets": config.STREETS_SHP,
    "parks": config.PARK_SHP,
    "listings": config.LISTING_FEATURES,
}

COLS = {
    "listing_id": "id",
    "amenities": "n_amenity_400",
    "quietness": "quietness",
    "poi_type": "type",
    "neigh_field": "neighname",
}

QUERY = {
    "poi_type": "restaurant",
    "neighborhood": "downtown",
    "eps_restaurant_m": 500,
    "eps_downtown_m": 300,
}

WEIGHTS = {"quietness": 0.5, "amenity": 0.5}
TOP_N = 5

# Where no measured quietness column exists, distance to the nearest highway is
# a transparent stand-in: farther is quieter, and it is independent of amenity
# density, so the trade-off in figure B stays real rather than circular.
QUIETNESS_PROXY = {
    "col": "dist_highway_km",
    "higher_is_quieter": True,
    "label": "quietness (proxy: distance to highway)",
}

SOURCE_CRS = 4326
METRIC_CRS = config.METER_CRS
WEB_MERCATOR = 3857

INK, SUB, PAGE = "#1c2430", "#5b6573", "#ffffff"
C_FILTER = "#2f6e8f"
C_AMENITY = "#b06a2c"
C_DOWNTOWN = "#7a5230"
C_POI = "#b9685f"
GREY_POINT = "#c2c9d4"
PARK_FILL = "#e7efe2"
STREET_LINE = "#d9dee6"
NEIGH_EDGE = "#9aa6b6"

SEQ_QUIET = LinearSegmentedColormap.from_list("quiet", ["#eaf3ef", "#2f8f7d", "#16463c"])
SEQ_AMENITY = LinearSegmentedColormap.from_list("amenity", ["#f7ede1", "#d08a3f", "#7a4413"])
SEQ_COMBINED = LinearSegmentedColormap.from_list("combined", ["#eef1f6", "#5a8f9e", "#243b6b"])


def set_fonts():
    for name in ["LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf"]:
        path = f"/usr/share/fonts/truetype/liberation/{name}"
        if os.path.exists(path):
            fm.fontManager.addfont(path)
    if any(f.name == "Liberation Sans" for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = "Liberation Sans"
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42


def find_column(df, candidates):
    lowered = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def read_layer(path, label, required=False):
    """Read a vector file or a CSV of points or WKT, returned in metric CRS."""
    if not path or not os.path.exists(path):
        if required:
            sys.exit(f"Required layer missing: {label} ({path})")
        return None

    if str(path).lower().endswith(".csv"):
        import pandas as pd

        df = pd.read_csv(path)
        wkt_col = find_column(df, ["geometry", "wkt", "geom", "the_geom"])
        if wkt_col:
            from shapely import wkt

            gdf = gpd.GeoDataFrame(df.drop(columns=[wkt_col]),
                                   geometry=df[wkt_col].apply(wkt.loads),
                                   crs=f"EPSG:{SOURCE_CRS}")
        else:
            lon = find_column(df, ["longitude", "lon", "lng", "x"])
            lat = find_column(df, ["latitude", "lat", "y"])
            if lon is None or lat is None:
                sys.exit(f"{label}: CSV needs coordinates or a WKT column, "
                         f"found {list(df.columns)}")
            gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon], df[lat]),
                                   crs=f"EPSG:{SOURCE_CRS}")
        return gdf.to_crs(epsg=METRIC_CRS)

    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=SOURCE_CRS)
    return gdf.to_crs(epsg=METRIC_CRS)


def load_layers():
    return {
        "pois": read_layer(LAYERS["pois"], "pois", required=True),
        "neighborhoods": read_layer(LAYERS["neighborhoods"], "neighborhoods", required=True),
        "streets": read_layer(LAYERS["streets"], "streets"),
        "parks": read_layer(LAYERS["parks"], "parks"),
        "listings": read_layer(LAYERS["listings"], "listings", required=True),
    }


def apply_filter(layers):
    pois, neighborhoods, listings = layers["pois"], layers["neighborhoods"], layers["listings"]

    restaurants = pois[pois[COLS["poi_type"]].astype(str).str.lower()
                       == QUERY["poi_type"].lower()]
    downtown = neighborhoods[neighborhoods[COLS["neigh_field"]].astype(str).str.lower()
                             == QUERY["neighborhood"].lower()]
    if restaurants.empty:
        sys.exit(f"No POI of type {QUERY['poi_type']!r} in the layer.")
    if downtown.empty:
        sys.exit(f"No neighborhood named {QUERY['neighborhood']!r} in the layer.")

    restaurant_buffer = restaurants.buffer(QUERY["eps_restaurant_m"]).union_all()
    downtown_buffer = downtown.buffer(QUERY["eps_downtown_m"]).union_all()

    listings = listings.copy()
    listings["in_restaurant"] = listings.intersects(restaurant_buffer)
    listings["in_downtown"] = listings.intersects(downtown_buffer)
    listings["candidate"] = listings["in_restaurant"] & listings["in_downtown"]

    counts = {
        "all": len(listings),
        "in_restaurant": int(listings["in_restaurant"].sum()),
        "candidate": int(listings["candidate"].sum()),
    }
    return listings, restaurants, restaurant_buffer, downtown_buffer, counts


def normalize(series):
    values = series.astype(float)
    low, high = values.min(), values.max()
    return (values - low) / (high - low) if high > low else values * 0 + 0.5


def score_candidates(candidates):
    """Normalise whatever soft axes exist and combine them by weight."""
    scored = candidates.copy()

    quiet_col = COLS["quietness"] if COLS["quietness"] in scored.columns else find_column(
        scored, ["quietness", "quiet", "quietness_score", "calm"])
    amenity_col = COLS["amenities"] if COLS["amenities"] in scored.columns else find_column(
        scored, ["n_amenity_400", "n_amenity_800", "amenity_density", "amenity_count"])

    quiet_label, is_proxy = "quietness", False
    if quiet_col is None and QUIETNESS_PROXY["col"] in scored.columns:
        base = normalize(scored[QUIETNESS_PROXY["col"]])
        scored["q_norm"] = base if QUIETNESS_PROXY["higher_is_quieter"] else 1 - base
        quiet_col, quiet_label, is_proxy = QUIETNESS_PROXY["col"], QUIETNESS_PROXY["label"], True
    elif quiet_col is not None:
        scored["q_norm"] = normalize(scored[quiet_col])

    if amenity_col is not None:
        scored["a_norm"] = normalize(scored[amenity_col])

    axes = []
    if "q_norm" in scored:
        axes.append(("q_norm", WEIGHTS["quietness"]))
    if "a_norm" in scored:
        axes.append(("a_norm", WEIGHTS["amenity"]))

    meta = {
        "has_quiet": "q_norm" in scored,
        "has_amenity": "a_norm" in scored,
        "quiet_col": quiet_col,
        "amenity_col": amenity_col,
        "quiet_label": quiet_label,
        "amenity_label": "amenity density",
        "is_proxy": is_proxy,
    }
    if not axes:
        return scored, meta

    total_weight = sum(w for _, w in axes)
    scored["soft"] = sum((w / total_weight) * scored[col] for col, w in axes)
    scored = scored.sort_values("soft", ascending=False)
    scored["rank"] = np.arange(1, len(scored) + 1)
    return scored, meta


def project(gdf, basemap):
    return gdf.to_crs(epsg=WEB_MERCATOR) if basemap else gdf


def frame(geometry, basemap, margin=0.18):
    series = gpd.GeoSeries([geometry], crs=f"EPSG:{METRIC_CRS}")
    minx, miny, maxx, maxy = series.total_bounds
    mx, my = margin * (maxx - minx), margin * (maxy - miny)
    padded = box(minx - mx, miny - my, maxx + mx, maxy + my)
    return tuple(project(gpd.GeoSeries([padded], crs=f"EPSG:{METRIC_CRS}"), basemap).total_bounds)


def draw_context(ax, layers, extent, basemap):
    if layers.get("parks") is not None:
        project(layers["parks"], basemap).clip(extent).plot(ax=ax, color=PARK_FILL, zorder=1, lw=0)
    if layers.get("streets") is not None:
        project(layers["streets"], basemap).clip(extent).plot(
            ax=ax, color=STREET_LINE, lw=0.5, zorder=2)
    project(layers["neighborhoods"], basemap).boundary.plot(
        ax=ax, color=NEIGH_EDGE, lw=0.6, zorder=3, alpha=0.7)


def style_axes(ax, extent):
    ax.set_xlim(extent[0], extent[2])
    ax.set_ylim(extent[1], extent[3])
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#cdd4de")
        spine.set_linewidth(0.8)
    ax.set_aspect("equal")


def add_basemap(ax):
    try:
        import contextily as cx

        cx.add_basemap(ax, source=cx.providers.CartoDB.PositronNoLabels,
                       crs=f"EPSG:{WEB_MERCATOR}", attribution_size=5)
    except Exception as exc:
        print(f"basemap skipped: {exc}")


def scalebar(ax, extent, basemap, length_m=1000):
    if basemap:
        return  # web mercator distorts metres at this latitude
    x0, y0, x1, y1 = extent
    x = x0 + 0.06 * (x1 - x0)
    y = y0 + 0.06 * (y1 - y0)
    ax.plot([x, x + length_m], [y, y], color=INK, lw=2.4, zorder=10, solid_capstyle="butt")
    ax.text(x + length_m / 2, y + 0.012 * (y1 - y0), f"{length_m / 1000:g} km",
            ha="center", va="bottom", fontsize=7, color=INK, zorder=10)


def figure_spatial_filter(layers, filtered, basemap=False):
    listings, restaurants, restaurant_buffer, downtown_buffer, counts = filtered
    extent = frame(downtown_buffer, basemap)

    plotted = project(listings, basemap)
    poi_points = project(restaurants, basemap)
    rest_ring = project(gpd.GeoSeries([restaurant_buffer], crs=f"EPSG:{METRIC_CRS}"), basemap)
    downtown_ring = project(gpd.GeoSeries([downtown_buffer], crs=f"EPSG:{METRIC_CRS}"), basemap)

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.9), dpi=200)
    fig.patch.set_facecolor(PAGE)

    panels = [
        ("1 · All listings", f"{counts['all']:,}", SUB),
        ("2 · Within 500 m of restaurants", f"{counts['in_restaurant']:,}", C_FILTER),
        ("3 · And within 300 m of downtown", f"{counts['candidate']:,}  ·  candidate set", C_FILTER),
    ]

    for i, ax in enumerate(axes):
        draw_context(ax, layers, extent, basemap)
        poi_points.cx[extent[0]:extent[2], extent[1]:extent[3]].plot(
            ax=ax, color=C_POI, markersize=2, alpha=0.45, zorder=4)

        if i >= 1:
            rest_ring.plot(ax=ax, color=C_FILTER, alpha=0.10, zorder=3, lw=0)
            rest_ring.boundary.plot(ax=ax, color=C_FILTER, lw=0.7, alpha=0.6, zorder=3)
        if i >= 2:
            downtown_ring.plot(ax=ax, color=C_DOWNTOWN, alpha=0.10, zorder=3, lw=0)
            downtown_ring.boundary.plot(ax=ax, color=C_DOWNTOWN, lw=0.9, alpha=0.7, zorder=3)

        if i == 0:
            plotted.plot(ax=ax, color=SUB, markersize=3, alpha=0.45, zorder=5, lw=0)
        else:
            flag = "in_restaurant" if i == 1 else "candidate"
            plotted[~plotted[flag]].plot(ax=ax, color=GREY_POINT, markersize=2.5,
                                         alpha=0.5, zorder=5, lw=0)
            plotted[plotted[flag]].plot(
                ax=ax, color=C_FILTER, markersize=4 if i == 1 else 9,
                edgecolor="none" if i == 1 else "white", lw=0 if i == 1 else 0.4,
                zorder=6 if i == 1 else 7)

        if basemap:
            add_basemap(ax)
        style_axes(ax, extent)
        scalebar(ax, extent, basemap)

        title, count, colour = panels[i]
        ax.set_title(title, fontsize=11, fontweight="bold", color=INK, loc="left", pad=8)
        ax.text(0.985, 0.975, count, transform=ax.transAxes, ha="right", va="top",
                fontsize=10, fontweight="bold", color=colour,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=colour, lw=1.0, alpha=0.9))

    legend = [
        Line2D([], [], marker="o", ls="", mfc=C_FILTER, mec="white", ms=7, label="kept"),
        Line2D([], [], marker="o", ls="", mfc=GREY_POINT, mec="none", ms=6, label="filtered out"),
        Line2D([], [], marker="o", ls="", mfc=C_POI, mec="none", ms=5, label="restaurant"),
        Patch(fc=C_FILTER, alpha=0.18, ec=C_FILTER, label="500 m buffer"),
        Patch(fc=C_DOWNTOWN, alpha=0.18, ec=C_DOWNTOWN, label="300 m downtown buffer"),
    ]
    fig.legend(handles=legend, ncol=5, loc="lower center", frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Spatial filter: dynamic-\u03b5 buffers shrink the candidate set",
                 x=0.012, ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.012, 0.93,
             "query: \u201ca place near lots of restaurants in downtown Austin\u201d"
             "   ·   ST_DWithin in PostGIS",
             ha="left", fontsize=9.5, color=SUB)
    fig.tight_layout(rect=[0, 0.04, 1, 0.92])
    return fig


def figure_soft_scoring(layers, downtown_buffer, scored, meta, basemap=False):
    if scored is None or scored.empty or not (meta["has_quiet"] or meta["has_amenity"]):
        print("No usable soft columns; figure B skipped.")
        return None

    extent = frame(downtown_buffer, basemap)
    points = project(scored, basemap)

    panels = []
    if meta["has_quiet"]:
        panels.append((meta["quiet_label"], "q_norm", SEQ_QUIET, "quieter \u2192"))
    if meta["has_amenity"]:
        panels.append((meta["amenity_label"], "a_norm", SEQ_AMENITY, "more amenities \u2192"))

    weight_note = []
    if meta["has_quiet"]:
        weight_note.append(f"quietness {WEIGHTS['quietness']}")
    if meta["has_amenity"]:
        weight_note.append(f"amenity {WEIGHTS['amenity']}")
    panels.append(("combined soft score", "soft", SEQ_COMBINED,
                   "weighted (" + ", ".join(weight_note) + ")"))

    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 5.9), dpi=200)
    if len(panels) == 1:
        axes = [axes]
    fig.patch.set_facecolor(PAGE)

    norm = Normalize(vmin=0, vmax=1)
    for i, (ax, (title, column, cmap, caption)) in enumerate(zip(axes, panels)):
        draw_context(ax, layers, extent, basemap)
        points.plot(ax=ax, column=column, cmap=cmap, norm=norm, markersize=22,
                    edgecolor="white", lw=0.3, zorder=6)

        if column == "soft":
            top = points[points["rank"] <= TOP_N]
            top.plot(ax=ax, facecolor="none", edgecolor=INK, lw=1.4, markersize=80, zorder=7)
            for _, row in top.iterrows():
                ax.annotate(str(int(row["rank"])), (row.geometry.x, row.geometry.y),
                            textcoords="offset points", xytext=(6, 5),
                            fontsize=8, fontweight="bold", color=INK, zorder=8)

        if basemap:
            add_basemap(ax)
        style_axes(ax, extent)
        scalebar(ax, extent, basemap)
        ax.set_title(f"{chr(65 + i)} · {title}", fontsize=11, fontweight="bold",
                     color=INK, loc="left", pad=8)

        mappable = ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
        bar = fig.colorbar(mappable, ax=ax, fraction=0.045, pad=0.02, orientation="horizontal")
        bar.outline.set_linewidth(0.5)
        bar.set_label(caption, fontsize=8, color=SUB)
        bar.ax.tick_params(labelsize=7, length=2)

    fig.suptitle("Soft scoring: candidates differentiated on competing axes",
                 x=0.012, ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.012, 0.93,
             f"{len(scored)} candidates from the spatial filter   ·   "
             f"normalised features, top {TOP_N} ringed",
             ha="left", fontsize=9.5, color=SUB)
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    return fig


def synthetic_layers():
    """Austin-shaped random data so the figure code can be exercised alone."""
    from shapely.geometry import LineString, Point, Polygon

    rng = np.random.default_rng(11)
    cx, cy = -97.743, 30.267

    def square(x, y, r):
        return Polygon([(x - r, y - r), (x + r, y - r), (x + r, y + r), (x - r, y + r)])

    neighborhoods = gpd.GeoDataFrame(
        {COLS["neigh_field"]: ["downtown", "east", "south"]},
        geometry=[square(cx, cy, 0.012), square(cx + 0.03, cy, 0.012),
                  square(cx, cy - 0.03, 0.012)],
        crs="EPSG:4326")

    xs = np.r_[rng.normal(cx, 0.006, 120), rng.normal(cx + 0.02, 0.01, 40),
               rng.normal(cx, 0.03, 60)]
    ys = np.r_[rng.normal(cy, 0.006, 120), rng.normal(cy - 0.02, 0.01, 40),
               rng.normal(cy, 0.03, 60)]
    types = ["restaurant"] * 160 + ["cafe"] * 30 + ["park"] * 30
    pois = gpd.GeoDataFrame({COLS["poi_type"]: types},
                            geometry=[Point(a, b) for a, b in zip(xs, ys)], crs="EPSG:4326")

    grid = [LineString([(x, cy - 0.05), (x, cy + 0.05)])
            for x in np.linspace(cx - 0.05, cx + 0.05, 12)]
    grid += [LineString([(cx - 0.05, y), (cx + 0.05, y)])
             for y in np.linspace(cy - 0.05, cy + 0.05, 12)]
    streets = gpd.GeoDataFrame(geometry=grid, crs="EPSG:4326")

    parks = gpd.GeoDataFrame(
        geometry=[square(cx - 0.02, cy + 0.015, 0.004), square(cx + 0.01, cy - 0.01, 0.005)],
        crs="EPSG:4326")

    n = 1500
    listings = gpd.GeoDataFrame({
        COLS["listing_id"]: np.arange(n),
        "name": [f"Listing {i}" for i in range(n)],
        "price": rng.integers(70, 160, n),
        COLS["amenities"]: rng.poisson(6, n),
        "dist_capmetro_km": np.round(rng.uniform(0.05, 1.5, n), 3),
        COLS["quietness"]: np.round(rng.uniform(-1, 1, n), 3),
        COLS["neigh_field"]: "n/a",
    }, geometry=[Point(a, b) for a, b in zip(rng.normal(cx, 0.03, n),
                                             rng.normal(cy, 0.03, n))], crs="EPSG:4326")

    return {name: gdf.to_crs(epsg=METRIC_CRS) for name, gdf in {
        "pois": pois, "neighborhoods": neighborhoods, "streets": streets,
        "parks": parks, "listings": listings}.items()}


def save(fig, outdir, stem):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{outdir}/{stem}.{ext}", dpi=300, facecolor=PAGE, bbox_inches="tight")
    print(f"wrote {stem}.png/.pdf/.svg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basemap", action="store_true", help="add a web street-map background")
    ap.add_argument("--synthetic", action="store_true", help="run on generated data")
    ap.add_argument("--outdir", default=str(config.FIGS))
    args = ap.parse_args()

    set_fonts()
    os.makedirs(args.outdir, exist_ok=True)
    layers = synthetic_layers() if args.synthetic else load_layers()

    filtered = apply_filter(layers)
    listings, _, _, downtown_buffer, counts = filtered
    print(f"all={counts['all']:,}  near restaurants={counts['in_restaurant']:,}  "
          f"candidates={counts['candidate']:,}")

    save(figure_spatial_filter(layers, filtered, basemap=args.basemap),
         args.outdir, "figA_spatial_filter")

    if not counts["candidate"]:
        return

    scored, meta = score_candidates(listings[listings["candidate"]])
    used = [c for c in (meta["quiet_col"], meta["amenity_col"]) if c]
    print(f"soft axes: {used if used else 'none found'}")

    figure_b = figure_soft_scoring(layers, downtown_buffer, scored, meta, basemap=args.basemap)
    if figure_b is not None:
        save(figure_b, args.outdir, "figB_soft_scoring")


if __name__ == "__main__":
    main()
