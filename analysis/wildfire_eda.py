"""Run: python wildfire_eda.py "wildfire_cleaned (1) (1)(1).xlsx" --output output
Dependencies: pandas, openpyxl, matplotlib, seaborn.
"""
import argparse
import numpy as np
import calendar
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Also works on machines without a graphical display.
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import seaborn as sns


FEATURES = ["temperature", "relative_humidity", "wind_speed", "rain",
            "FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]
SHORT = ["Temp", "RH", "Ws", "Rain", *FEATURES[4:], "Fire"]
LABELS = ["No Fire (0)", "Fire (1)"]
COLORS = ["#326985", "#c24c24"]


def save(fig, output, name, title, note):
    """Reserve figure space for the interpretation, also in exported images."""
    fig.suptitle(title, x=0.07, y=0.98, ha="left", weight="bold", fontsize=18)
    note += " Weather units are unverified." if name in {"03_boxplots", "05_humidity_temperature", "06_temperature", "07_wind_rain"} else ""
    fig.text(0.07, 0.025, textwrap.fill(note, 100), va="bottom", fontsize=10)
    fig.tight_layout(rect=(0.025, 0.17, 0.98, 0.92))
    fig.savefig(output / "figures" / f"{name}.png", dpi=170, facecolor="white")
    plt.close(fig)


def analyze(workbook, output):
    for folder in ["figures", "tables"]:
        (output / folder).mkdir(parents=True, exist_ok=True)
    data = pd.read_excel(workbook, sheet_name="Clean data")
    target = data["fire"]
    encoded = data["class_label"].str.strip().str.lower().map({"fire": 1, "not fire": 0})
    if target.isna().any() or not target.isin([0, 1]).all() or not encoded.eq(target).all():
        raise ValueError("Check the workbook's binary target and text labels.")
    columns = FEATURES + ["fire"]
    if data[columns + ["date"]].isna().any().any():
        raise ValueError("Missing dates or analysis values require review before plotting.")

    # Keep source and audit columns in the schema, outside the feature matrix.
    schema = pd.DataFrame({"dtype": data.dtypes.astype(str), "missing": data.isna().sum(),
                           "distinct": data.nunique()})
    counts = target.value_counts().reindex([0, 1], fill_value=0)
    shares = 100 * counts / len(data)
    summary = data[columns].describe().T
    for name, table in [("schema", schema), ("summary", summary),
                        ("class_counts", pd.DataFrame({"count": counts, "percent": shares}))]:
        table.to_csv(output / "tables" / f"{name}.csv")
        print(name, table.round(2), sep="\n")
    print("Duplicate region/date records:", data.duplicated(["region", "date"]).sum())
    sns.set_theme(style="whitegrid", context="notebook")

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(LABELS, counts, color=COLORS, width=0.55)
    ax.bar_label(bars, labels=[f"{n:,}  ({p:.1f}%)" for n, p in zip(counts, shares)], padding=6)
    ax.set(ylabel="Number of observations", ylim=(0, counts.max() * 1.18))
    save(fig, output, "01_classes", "Fire and non-fire observations",
         f"Fire accounts for {shares[1]:.1f}% of the {len(data):,} records. Both classes are well represented.")

    # Pearson r with a binary 0/1 target is a point-biserial correlation.
    corr = data[columns].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-1, vmax=1,
                square=True, xticklabels=SHORT, yticklabels=SHORT, annot_kws={"size": 9},
                cbar_kws={"shrink": 0.78, "label": "Pearson r"}, ax=ax)
    ax.tick_params(axis="y", labelrotation=0)
    save(fig, output, "02_correlations", "Weather, fire indices and the fire label",
         f"FWI (r = {corr.at['FWI','fire']:.2f}) and FFMC ({corr.at['FFMC','fire']:.2f}) track the fire label most closely. "
         f"Relative humidity has the strongest negative association ({corr.at['relative_humidity','fire']:.2f}).")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    med = data.groupby("fire")[FEATURES].median()
    for ax, col, title in zip(axes.flat, [*FEATURES[:3], "FFMC"],
                              ["Temperature", "Relative humidity", "Wind speed", "Fine Fuel Moisture Code"]):
        sns.boxplot(data=data, x="fire", y=col, hue="fire", order=[0, 1], hue_order=[0, 1],
                    palette=COLORS, dodge=False, legend=False, width=0.52, fliersize=3, ax=ax)
        ax.set_xticks([0, 1], LABELS)
        ax.set(title=title, xlabel="", ylabel="Recorded value" if col != "FFMC" else "Index value")
    save(fig, output, "03_boxplots", "How conditions differ by fire class",
         f"Fire records have higher median temperature ({med.at[1,'temperature']:g} vs {med.at[0,'temperature']:g}), "
         f"lower humidity ({med.at[1,'relative_humidity']:g} vs {med.at[0,'relative_humidity']:g}) and higher FFMC. "
         "Wind-speed distributions overlap substantially.")

    # Define an illustrative corner from the weather measurements alone.
    # Inclusive boundaries retain all ties in these integer-valued features.
    rh, temp = "relative_humidity", "temperature"
    dry, hot = data[rh].quantile(.25), data[temp].quantile(.75)
    corner = data.loc[(data[rh] <= dry) & (data[temp] >= hot), "fire"]
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(data=data.sample(frac=1, random_state=42), x=rh, y=temp, hue="fire",
                    style="fire", markers={0: "o", 1: "^"}, hue_order=[0, 1], style_order=[0, 1],
                    palette=COLORS, s=42, alpha=.57, linewidth=.3, ax=ax)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, LABELS)
    left, _ = ax.get_xlim()
    _, top = ax.get_ylim()
    ax.add_patch(Rectangle((left, hot), dry-left, top-hot, facecolor="#f0ad58",
                           edgecolor="#925518", alpha=.24, linestyle="--", zorder=0))
    ax.set(xlabel="Relative humidity (recorded value)", ylabel="Temperature (recorded value)")
    save(fig, output, "05_humidity_temperature", "Fire labels cluster in hot, dry conditions",
         f"Shading marks temperature ≥ {hot:g} and humidity ≤ {dry:g} (sample quartiles). "
         f"{corner.sum()} of {len(corner)} records there carry fire labels ({corner.mean():.1%}). "
         "Points can overlap. These cutoffs describe this sample.")


    tables = output / "tables"
    corr.to_csv(tables / "correlations.csv")
    data.groupby("fire")[FEATURES].describe().T.to_csv(tables / "statistics_by_class.csv")

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, color in enumerate(COLORS):
        values = data.loc[target.eq(label), "temperature"]
        bins = np.arange(data.temperature.min() - .5, data.temperature.max() + 1.5)
        ax.hist(values, bins=bins, weights=np.full(len(values), 100 / len(values)),
                histtype="step", linewidth=2.5, color=color, label=LABELS[label])
    ax.legend()
    ax.set(xlabel="Temperature (recorded value)", ylabel="Share within class (%)")
    save(fig, output, "06_temperature", "Temperature distribution by class",
         "Each bin spans one recorded unit. The two class distributions each sum to 100%.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, col, title in zip(axes, ["wind_speed", "rain"], ["Wind speed", "Rainfall"]):
        for label, color in enumerate(COLORS):
            values = data.loc[target.eq(label), col].sort_values()
            ax.step(values, np.arange(1, len(values)+1) * 100 / len(values),
                    where="post", color=color, lw=2.6, label=LABELS[label])
        ax.set(xlabel=title + " (recorded value)", ylabel="Records at or below value (%)", ylim=(-2, 103))
        ax.legend(loc="lower right")
    axes[1].set_xscale("symlog", linthresh=.1)
    axes[1].set_xticks([0, .1, 1, 5, 20], ["0", "0.1", "1", "5", "20"])
    save(fig, output, "07_wind_rain", "Wind speed and rainfall by class",
         "Cumulative distributions retain zero rainfall and rare extremes. Rainfall uses a compressed scale above 0.1.")

    region = data.groupby("region").fire.agg(records="size", fires="sum").sort_values("records")
    region["percent"] = 100 * region.fires / region.records
    region = region.sort_values("percent")
    region.to_csv(tables / "region.csv")
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(region.index, region.percent, color=COLORS[0], height=.65)
    ax.bar_label(bars, labels=[f"{row.percent:.1f}%  ({row.fires:.0f}/{row.records:.0f})"
                               for row in region.itertuples()], padding=5)
    ax.set(xlim=(0, 100), xlabel="Recorded region-days labeled Fire (%)")
    save(fig, output, "08_region", "Fire share by region",
         "Labels give fire records / observed records. Regions cover different years, limiting direct comparisons.")

    dated = data.assign(year=data.date.dt.year, month=data.date.dt.month)
    trusted = dated.loc[dated.date_status.ne("year_corrected_inferred")]
    def monthly(frame):
        table = frame.groupby(["year", "month"]).fire.agg(records="size", fires="sum")
        return table.assign(percent=100 * table.fires / table.records)
    time = monthly(trusted).join(monthly(dated), lsuffix="_primary", rsuffix="_including_inferred")
    missing = pd.read_excel(workbook, sheet_name="Missing dates")
    missing.to_csv(tables / "missing_dates.csv", index=False)
    time.to_csv(tables / "monthly_sensitivity.csv")
    dated.loc[~dated.index.isin(trusted.index), ["source_row", "region", "date", "fire", "date_status"]].to_csv(
        tables / "inferred_dates.csv", index=False)
    fig, axes = plt.subplots(1, len(time.index.levels[0]), figsize=(13, 6), sharey=True)
    for ax, (year, block) in zip(axes, time.groupby(level=0)):
        months = block.index.get_level_values("month")
        bars = ax.bar([calendar.month_abbr[m] for m in months], block.percent_primary, color=COLORS[1], width=.65)
        ax.bar_label(bars, labels=[f"{f}/{n}" for f, n in zip(block.fires_primary, block.records_primary)], fontsize=9)
        ax.set(title=f"{year}  ·  {trusted.loc[trusted.year.eq(year), 'region'].nunique()} region(s)", ylim=(0, 100))
    axes[0].set_ylabel("Observed region-days labeled Fire (%)")
    save(fig, output, "09_monthly", "Fire occurrence by year and month",
         "Labels show fire / observed region-days. Five absent August 2021 dates stay missing. Five inferred September 2021 dates are excluded; including them gives 74/210 versus 73/205.")

    # Retain extremes and export cell-level flags with their original source row.
    flags = []
    for col in FEATURES:
        q1, q3 = data[col].quantile([.25, .75])
        low, high = q1 - 1.5*(q3-q1), q3 + 1.5*(q3-q1)
        rows = data.loc[~data[col].between(low, high), ["source_row", "region", "date", "fire", col]]
        flags.append(rows.rename(columns={col:"value"}).assign(feature=col, lower=low, upper=high))
    pd.concat(flags, ignore_index=True).to_csv(tables / "outlier_flags.csv", index=False)
    return data, region, time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output"))
    args = parser.parse_args()
    analyze(args.workbook, args.output)
