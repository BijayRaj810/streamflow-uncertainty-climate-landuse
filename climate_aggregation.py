"""
climate_aggregation.py
=======================
Reusable monthly/yearly aggregation helpers for multi-station daily climate
records (precipitation, Tmax, Tmin). Use `how="sum"` for precipitation (daily
totals accumulate into monthly/annual totals) and `how="mean"` for temperature
(does not accumulate).
"""

import pandas as pd

MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _with_clean_dates(df, date_col):
    """Drop rows with no date (e.g. trailing blank rows from an Excel export)
    before deriving Year/Month -- otherwise a single NaT forces the whole
    Year column to float64, which is why the old output had years like
    "1985.0" instead of "1985"."""
    df = df.dropna(subset=[date_col]).copy()
    df[date_col] = pd.to_datetime(df[date_col])
    return df


def monthly_aggregate(df, date_col, station_cols, how="sum"):
    """One row per calendar year-month (e.g. "1985Jan"), aggregating
    `station_cols` with `how`."""
    df = _with_clean_dates(df, date_col)
    df["YYMM"] = df[date_col].dt.year.astype(str) + df[date_col].dt.strftime("%b")
    return (
        df.groupby("YYMM", sort=False)[list(station_cols)]
        .agg(how)
        .reset_index()
    )


def yearly_aggregate(df, date_col, station_cols, how="sum"):
    """One row per year, aggregating `station_cols` with `how`."""
    df = _with_clean_dates(df, date_col)
    df["Year"] = df[date_col].dt.year
    return (
        df.groupby("Year")[list(station_cols)]
        .agg(how)
        .reset_index()
    )


def monthly_climatology(df, date_col, station_cols, how="sum"):
    """Long-term average per calendar month (Jan-Dec): aggregate
    `station_cols` with `how` within each (Year, Month), then average that
    across all years. `how="sum"` gives "average total rainfall in a
    typical January"; `how="mean"` gives "average daily Tmax/Tmin in a
    typical January"."""
    df = _with_clean_dates(df, date_col)
    df["Year"] = df[date_col].dt.year
    df["Month"] = df[date_col].dt.strftime("%b")
    per_year_month = df.groupby(["Year", "Month"])[list(station_cols)].agg(how)
    climatology = per_year_month.groupby("Month")[list(station_cols)].mean().round(2)
    return climatology.reindex(MONTH_ORDER).reset_index()


def summary_stats(df, station_cols, transpose=False, add_sum=False):
    """describe() across `station_cols`. `transpose=True` puts stations as
    rows (matches the precipitation stats layout); `add_sum=True` inserts a
    `sum` column (count * mean) -- only meaningful when transposed."""
    stats = df[list(station_cols)].describe().round(2)
    if transpose:
        stats = stats.transpose()
        if add_sum:
            stats.insert(2, "sum", (stats["count"] * stats["mean"]).round(2))
    return stats
