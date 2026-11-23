"""Rebuild split manifests and descriptive checks; never fit an estimator."""
import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd

from preprocessing import FEATURES, INDICES, load_dataset, primary_split, region_transfer_splits


def prepare(workbook, output):
    output.mkdir(parents=True, exist_ok=True)
    data = load_dataset(workbook)
    X, y, meta = data.X, data.y, data.meta
    role = primary_split(meta)
    manifest = meta.join(role)
    manifest.to_csv(output / "split_manifest.csv")
    summary = manifest.join(y).groupby("split").agg(
        rows=("date", "size"), dates=("date", "nunique"), first_date=("date", "min"),
        last_date=("date", "max"), regions=("region", "nunique"), fire=("fire", "sum"))
    summary["no_fire"] = summary.rows - summary.fire
    summary.to_csv(output / "split_summary.csv")
    print(summary)
    raw = pd.read_excel(workbook, sheet_name="Clean data")
    audit = pd.Series({
        "rows": len(X), "missing_cells_full_sheet": raw.isna().sum().sum(),
        "exact_duplicate_rows": raw.duplicated().sum(),
        "duplicate_region_dates": meta.duplicated(["region", "date"]).sum(),
        "repeated_weather_rows_beyond_first": X.duplicated().sum(),
        "weather_combinations_with_both_labels": X.join(y).groupby(list(FEATURES)).fire.nunique().gt(1).sum(),
    })
    audit.rename("count").to_csv(output / "data_quality.csv")
    stats = X.describe().T.assign(dtype=X.dtypes.astype(str), missing=X.isna().sum())
    low = stats["25%"] - 1.5 * (stats["75%"] - stats["25%"])
    high = stats["75%"] + 1.5 * (stats["75%"] - stats["25%"])
    stats["iqr_flags"] = (X.lt(low) | X.gt(high)).sum()
    stats.to_csv(output / "weather_summary.csv")
    # Predictor-only diagnostics for the optional indices; never added to the pipeline.
    raw[list(INDICES)].corr().to_csv(output / "index_correlations.csv")
    overlap = []
    for earlier, later in combinations(["train", "validation", "test"], 2):
        a, b = role.eq(earlier), role.eq(later)
        matched = X[b].merge(X[a].drop_duplicates(), on=list(FEATURES))
        overlap.append(dict(earlier=earlier, later=later, later_rows_matching_weather=len(matched),
                            shared_regions=len(set(meta.region[a]) & set(meta.region[b])),
                            shared_dates=len(set(meta.date[a]) & set(meta.date[b]))))
    pd.DataFrame(overlap).to_csv(output / "overlap.csv", index=False)
    counts = []
    for region, train, valid in region_transfer_splits(meta, role):
        assert set(meta.loc[train, "region"]).isdisjoint(meta.loc[valid, "region"])
        assert meta.loc[train, "date"].max() < meta.loc[valid, "date"].min()
        counts.append(dict(region=region, train_rows=len(train), validation_rows=len(valid),
                           validation_fires=int(y.loc[valid].sum())))
    pd.DataFrame(counts).to_csv(output / "region_validation.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "audit")
    args = parser.parse_args()
    prepare(args.workbook, args.output)
