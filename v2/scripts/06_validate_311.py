"""Validate the perceived quietness field against 311 noise complaints.

The field is built from guest reviews and the complaints are an independent
municipal record, so agreement between them is external evidence rather than a
consistency check. Two quantities are reported: the correlation itself, which
should be negative because positive quietness means quieter, and the gain in
explained variance once quietness is added to geometry-derived covariates. The
second is the part that matters for the argument, since it asks whether
perception carries information the built environment alone does not.
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from housisense import config

MIN_ASPECT_REVIEWS = 3
DEFAULT_COVARIATES = ["dist_highway", "road_length_km", "poi_amenity", "poi_shop"]


def load():
    config.require(config.CELL_QUIETNESS, config.NOISE_H3)
    cells = pd.read_csv(config.CELL_QUIETNESS)
    noise = pd.read_csv(config.NOISE_H3)
    merged = cells.merge(noise, on="h3", how="left")
    merged["noise_311"] = merged["noise_311"].fillna(0)
    return merged


def correlations(df, lines):
    lines.append(f"Cells with at least {MIN_ASPECT_REVIEWS} aspect reviews: {len(df)}")
    rho, p_rho = spearmanr(df["quietness"], df["noise_311"])
    r, p_r = pearsonr(df["quietness"], np.log1p(df["noise_311"]))
    lines.append(f"Spearman, quietness vs complaint count : rho = {rho:.3f} (p = {p_rho:.2e})")
    lines.append(f"Pearson, quietness vs log(count + 1)   : r   = {r:.3f} (p = {p_r:.2e})")


def incremental_r2(df, covariate_path, lines):
    import statsmodels.api as sm

    covariates = pd.read_csv(covariate_path)
    key = "h3_index" if "h3_index" in covariates.columns else "h3"
    available = [c for c in DEFAULT_COVARIATES if c in covariates.columns]
    if not available:
        lines.append(f"\nNo usable covariates in {covariate_path}; skipped.")
        return

    merged = df.merge(covariates[[key] + available], left_on="h3", right_on=key,
                      how="left").dropna(subset=available)
    y = np.log1p(merged["noise_311"])
    baseline = sm.OLS(y, sm.add_constant(merged[available].fillna(0))).fit()
    full = sm.OLS(y, sm.add_constant(merged[available + ["quietness"]].fillna(0))).fit()

    lines.append("")
    lines.append(f"Incremental variance explained (n = {len(merged)}, DV = log(count + 1))")
    lines.append(f"  covariates: {', '.join(available)}")
    lines.append(f"  R2 geometry only        : {baseline.rsquared:.3f}")
    lines.append(f"  R2 plus quietness       : {full.rsquared:.3f}")
    lines.append(f"  delta R2                : {full.rsquared - baseline.rsquared:.3f}")
    lines.append(f"  quietness coefficient   : {full.params['quietness']:.3f} "
                 f"(p = {full.pvalues['quietness']:.2e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--covariates",
                    help="CSV keyed on h3 with geometry-derived noise proxies")
    args = ap.parse_args()

    cells = load()
    usable = cells[cells["n_reviews_aspect"] >= MIN_ASPECT_REVIEWS].copy()

    lines = []
    correlations(usable, lines)
    if args.covariates:
        incremental_r2(usable, args.covariates, lines)

    report = "\n".join(lines)
    print(report)
    config.VALIDATION_REPORT.write_text(report + "\n")
    print(f"\nwrote {config.VALIDATION_REPORT}")


if __name__ == "__main__":
    main()
