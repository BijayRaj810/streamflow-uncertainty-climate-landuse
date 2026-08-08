"""
uq_toolkit.py
=============
Reusable toolkit for Uncertainty Quantification (UQ) and Source Attribution
of SWAT future-scenario streamflow ensembles.

Designed to be channel-agnostic: point it at any wide-format CSV that follows
the same layout used for this project (see `load_wide_csv` docstring) and it
will reproduce the full analysis described in the
"Future Scenario Uncertainty Quantification and Source Attribution Guide".

Typical usage
-------------
    from uq_toolkit import run_full_analysis

    run_full_analysis(
        csv_path="cha98_flo_out_wide52.csv",
        channel_name="cha98",
        outdir="../All_DATA/Uncertainty_Outputs/cha98"
    )

Author: generated for a SWAT+ streamflow uncertainty quantification workflow
"""

import os
import itertools
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

SEASONS = {
    "DJF": ["Dec", "Jan", "Feb"],
    "MAM": ["Mar", "Apr", "May"],
    "JJAS": ["Jun", "Jul", "Aug", "Sep"],
    "ON": ["Oct", "Nov"],
}

PERIOD_ORDER = ["near_future", "mid_future", "far_future"]


# --------------------------------------------------------------------------
# 1. DATA INGESTION / RESHAPING
# --------------------------------------------------------------------------

def load_wide_csv(csv_path, channel_name):
    """
    Reads a wide-format SWAT scenario-output CSV and reshapes it into a tidy
    long table.

    Expected file layout (as produced by the SWAT scenario export used in
    this project):
        Row 1:  "Scenario_Type" , Historical, LU_Only..., CC_Only..., LU_CC...
        Row 2:  "Period"        , historical, near_future, mid_future, ...
        Row 3:  "Emission"      , historical, 245/585 ...
        Row 4:  "Climate_Model" , historical, Cool-Dry/Cool-Wet/Hot-Dry/Hot-Wet ...
        Row 5+: Jan..Dec monthly flow values, one row per month.

    The first column holds the row labels (Scenario_Type/Period/Emission/
    Climate_Model/Month); every column after that is one simulation run.

    Returns
    -------
    pd.DataFrame with columns:
        Channel, Month, Scenario_Type, Period, SSP, Climate_Model, Run_ID, Flow
    """
    raw = pd.read_csv(csv_path, header=None, dtype=str)

    meta = raw.iloc[0:4, 1:].reset_index(drop=True).T
    meta.columns = ["Scenario_Type", "Period", "Emission", "Climate_Model"]
    meta = meta.reset_index(drop=True)
    meta["Run_ID"] = meta.index.astype(int)

    body = raw.iloc[4:, :].reset_index(drop=True)
    body.columns = ["Month"] + list(range(body.shape[1] - 1))
    body = body.melt(id_vars="Month", var_name="Run_ID", value_name="Flow")
    body["Run_ID"] = body["Run_ID"].astype(int)
    body["Flow"] = pd.to_numeric(body["Flow"], errors="coerce")

    tidy = body.merge(meta, on="Run_ID", how="left")
    tidy["Channel"] = channel_name
    tidy["SSP"] = tidy["Emission"].replace({"245": "SSP2-4.5", "585": "SSP5-8.5"})
    tidy["Month"] = pd.Categorical(tidy["Month"], categories=MONTH_ORDER, ordered=True)

    period_cat = ["historical"] + PERIOD_ORDER
    tidy["Period"] = pd.Categorical(tidy["Period"], categories=period_cat, ordered=True)

    cols = ["Channel", "Month", "Scenario_Type", "Period", "Emission", "SSP",
            "Climate_Model", "Run_ID", "Flow"]
    return tidy[cols].sort_values(["Month", "Scenario_Type", "Period"]).reset_index(drop=True)


def get_historical(tidy):
    """Returns a Series of historical flow indexed by Month."""
    hist = tidy.loc[tidy["Scenario_Type"] == "Historical", ["Month", "Flow"]]
    hist = hist.drop_duplicates(subset="Month").set_index("Month")["Flow"]
    return hist.reindex(MONTH_ORDER)


def get_future(tidy):
    """Returns only the future ensemble members (excludes Historical rows)."""
    return tidy.loc[tidy["Scenario_Type"] != "Historical"].copy()


# --------------------------------------------------------------------------
# 2. UNCERTAINTY QUANTIFICATION (descriptive statistics)
# --------------------------------------------------------------------------

def monthly_uncertainty_stats(tidy, group_cols=("Month",)):
    """
    Computes mean, median, min, max, std, CV, and 5/25/75/95th percentiles
    for the future ensemble, grouped by `group_cols` (default: Month only;
    pass e.g. ("Month", "Scenario_Type") or ("Month", "Period") to split
    further).
    """
    fut = get_future(tidy)

    def _stats(s):
        return pd.Series({
            "Mean": s.mean(),
            "Median": s.median(),
            "Min": s.min(),
            "Max": s.max(),
            "Std": s.std(),
            "CV_%": 100 * s.std() / s.mean() if s.mean() else np.nan,
            "P05": s.quantile(0.05),
            "P25": s.quantile(0.25),
            "P75": s.quantile(0.75),
            "P95": s.quantile(0.95),
            "N": s.count(),
        })

    out = fut.groupby(list(group_cols), observed=True)["Flow"].apply(_stats).unstack()
    return out.reset_index()


def seasonal_aggregate(tidy):
    """
    Aggregates monthly flow into seasonal totals (DJF, MAM, JJAS, ON) per
    Run_ID by averaging the months in each season. Returns a tidy frame with
    a "Season" column in place of "Month".
    """
    month_to_season = {m: s for s, months in SEASONS.items() for m in months}
    df = tidy.copy()
    df["Season"] = df["Month"].map(month_to_season)
    keep = ["Channel", "Season", "Scenario_Type", "Period", "Emission", "SSP",
            "Climate_Model", "Run_ID"]
    seasonal = df.groupby(keep, observed=True)["Flow"].mean().reset_index()
    seasonal["Season"] = pd.Categorical(seasonal["Season"],
                                         categories=list(SEASONS.keys()), ordered=True)
    return seasonal


def seasonal_uncertainty_stats(tidy):
    seasonal = seasonal_aggregate(tidy)
    fut = seasonal.loc[seasonal["Scenario_Type"] != "Historical"]

    def _stats(s):
        return pd.Series({
            "Mean": s.mean(), "Median": s.median(), "Min": s.min(), "Max": s.max(),
            "Std": s.std(), "CV_%": 100 * s.std() / s.mean() if s.mean() else np.nan,
            "P05": s.quantile(0.05), "P25": s.quantile(0.25),
            "P75": s.quantile(0.75), "P95": s.quantile(0.95), "N": s.count(),
        })

    out = fut.groupby("Season", observed=True)["Flow"].apply(_stats).unstack()
    return out.reset_index()


# --------------------------------------------------------------------------
# 3. SOURCE ATTRIBUTION
# --------------------------------------------------------------------------

def source_attribution(tidy):
    """
    Computes, for every Month x Period x SSP x Climate_Model combination
    present in both CC_Only and LU_CC:
        delta_LU       = Q_LU_only        - Q_historical
        delta_CC       = Q_CC_only        - Q_historical
        delta_Combined = Q_LU_CC          - Q_historical
        Interaction    = Q_LU_CC - Q_LU_only - Q_CC_only + Q_historical

    (Q_LU_only depends only on Month & Period, since the LU_Only scenario
    holds climate fixed at baseline; it is broadcast across SSP/Climate_Model.)

    Also computes percentage change vs. historical for LU, CC, Combined,
    and (implicitly) the interaction share.

    Returns a tidy DataFrame with one row per Month x Period x SSP x
    Climate_Model.
    """
    hist = get_historical(tidy)  # Series by Month

    lu = tidy.loc[tidy["Scenario_Type"] == "LU_Only",
                  ["Month", "Period", "Flow"]].rename(columns={"Flow": "Q_LU"})
    cc = tidy.loc[tidy["Scenario_Type"] == "CC_Only",
                  ["Month", "Period", "SSP", "Climate_Model", "Flow"]].rename(columns={"Flow": "Q_CC"})
    combo = tidy.loc[tidy["Scenario_Type"] == "LU_CC",
                     ["Month", "Period", "SSP", "Climate_Model", "Flow"]].rename(columns={"Flow": "Q_Combined"})

    merged = cc.merge(combo, on=["Month", "Period", "SSP", "Climate_Model"], how="inner")
    merged = merged.merge(lu, on=["Month", "Period"], how="left")
    hist_map = {str(k): v for k, v in hist.items()}
    merged["Q_Historical"] = merged["Month"].astype(str).map(hist_map).astype(float)
    merged["Q_LU"] = merged["Q_LU"].astype(float)
    merged["Q_CC"] = merged["Q_CC"].astype(float)
    merged["Q_Combined"] = merged["Q_Combined"].astype(float)

    merged["delta_LU"] = merged["Q_LU"] - merged["Q_Historical"]
    merged["delta_CC"] = merged["Q_CC"] - merged["Q_Historical"]
    merged["delta_Combined"] = merged["Q_Combined"] - merged["Q_Historical"]
    merged["Interaction"] = (merged["Q_Combined"] - merged["Q_LU"]
                              - merged["Q_CC"] + merged["Q_Historical"])

    merged["pct_LU"] = 100 * merged["delta_LU"] / merged["Q_Historical"]
    merged["pct_CC"] = 100 * merged["delta_CC"] / merged["Q_Historical"]
    merged["pct_Combined"] = 100 * merged["delta_Combined"] / merged["Q_Historical"]
    merged["pct_Interaction"] = 100 * merged["Interaction"] / merged["Q_Historical"]

    merged["Month"] = pd.Categorical(merged["Month"], categories=MONTH_ORDER, ordered=True)
    merged["Period"] = pd.Categorical(merged["Period"], categories=PERIOD_ORDER, ordered=True)
    return merged.sort_values(["Month", "Period", "SSP", "Climate_Model"]).reset_index(drop=True)


def monthly_pct_change_summary(attrib_df):
    """Mean monthly % change for LU, CC, Combined, Interaction (averaged
    across SSP/Climate_Model, separated by Month and Period)."""
    out = attrib_df.groupby(["Month", "Period"], observed=True)[
        ["pct_LU", "pct_CC", "pct_Combined", "pct_Interaction"]
    ].mean()
    out.columns = ["Land_Use_%", "Climate_%", "Combined_%", "Interaction_%"]
    return out.reset_index()


# --------------------------------------------------------------------------
# 4. VARIANCE-BASED ATTRIBUTION (3-way ANOVA: GCM x SSP x LULC)
# --------------------------------------------------------------------------

def three_way_variance_decomposition(tidy, method="omega2"):
    """
    For each Month x Period, performs a balanced 3-way ANOVA-style sum-of-
    squares decomposition of the change in Flow (Delta_Flow) using:
        Factor A: Climate_Model (GCM), 4 levels
        Factor B: SSP (Emission), 2 levels
        Factor C: LULC, 2 levels {CC_Only = baseline, LU_CC = future}
    following the design of Aryal et al. (2018) / Gaur et al. (2021).

    Uses statsmodels OLS with Type-II sums of squares when available;
    falls back to a manual balanced-design SS decomposition otherwise.

    Parameters
    ----------
    method : {"eta2", "omega2"}
        "eta2": classic variance fraction, SS_factor / SST. 
        "omega2" (default): unbiased (debiased) fraction.

    Returns a DataFrame with variance fractions (%), F-values (_F), 
    and p-values (_p) for all main effects and two-way interactions.
    """
    if method not in ("eta2", "omega2"):
        raise ValueError("method must be 'eta2' or 'omega2'")

    # 1. Filter to the fully crossed scenarios and establish the binary LULC factor
    df = tidy.loc[tidy["Scenario_Type"].isin(["CC_Only", "LU_CC"])].copy()
    df["LULC"] = df["Scenario_Type"].map({"CC_Only": "baseline", "LU_CC": "future"})

    # 2. Calculate Delta_Flow vs. Historical Baseline
    hist = get_historical(tidy)
    hist_map = {str(k): v for k, v in hist.items()}
    df["Hist_Flow"] = df["Month"].astype(str).map(hist_map).astype(float)
    df["Delta_Flow"] = df["Flow"] - df["Hist_Flow"]

    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
        use_statsmodels = True
    except ImportError:
        use_statsmodels = False
        if method == "omega2":
            raise ImportError("method='omega2' requires statsmodels to be installed.")

    records = []
    for (month, period), sub in df.groupby(["Month", "Period"], observed=True):
        if sub["Delta_Flow"].isna().all() or len(sub) < 2:
            continue
        
        sub = sub.rename(columns={"Climate_Model": "GCM"})

        if use_statsmodels:
            formula = "Delta_Flow ~ C(GCM) * C(SSP) + C(GCM) * C(LULC) + C(SSP) * C(LULC)"
            model = ols(formula, data=sub).fit()
            aov = sm.stats.anova_lm(model, typ=2)
            
            ss = aov["sum_sq"]
            dfs = aov["df"]
            f_vals = aov.get("F", {})
            p_vals = aov.get("PR(>F)", {})
            mse = model.mse_resid

            def get_val(table, key_parts, default=0.0):
                if not hasattr(table, "index"): return default
                for idx in table.index:
                    parts = set(p.strip() for p in idx.replace("C(", "").replace(")", "").split(":"))
                    if parts == set(key_parts):
                        return table[idx]
                return default

            # Sum of Squares
            ss_gcm = get_val(ss, ["GCM"])
            ss_ssp = get_val(ss, ["SSP"])
            ss_lulc = get_val(ss, ["LULC"])
            ss_gcm_ssp = get_val(ss, ["GCM", "SSP"])
            ss_gcm_lulc = get_val(ss, ["GCM", "LULC"])
            ss_ssp_lulc = get_val(ss, ["SSP", "LULC"])
            
            # F-values
            f_gcm = get_val(f_vals, ["GCM"], float("nan"))
            f_ssp = get_val(f_vals, ["SSP"], float("nan"))
            f_lulc = get_val(f_vals, ["LULC"], float("nan"))
            f_gcm_ssp = get_val(f_vals, ["GCM", "SSP"], float("nan"))
            f_gcm_lulc = get_val(f_vals, ["GCM", "LULC"], float("nan"))
            f_ssp_lulc = get_val(f_vals, ["SSP", "LULC"], float("nan"))
            
            # p-values
            p_gcm = get_val(p_vals, ["GCM"], float("nan"))
            p_ssp = get_val(p_vals, ["SSP"], float("nan"))
            p_lulc = get_val(p_vals, ["LULC"], float("nan"))
            p_gcm_ssp = get_val(p_vals, ["GCM", "SSP"], float("nan"))
            p_gcm_lulc = get_val(p_vals, ["GCM", "LULC"], float("nan"))
            p_ssp_lulc = get_val(p_vals, ["SSP", "LULC"], float("nan"))

            sst = ss.sum()

            if method == "omega2":
                df_gcm = get_val(dfs, ["GCM"])
                df_ssp = get_val(dfs, ["SSP"])
                df_lulc = get_val(dfs, ["LULC"])
                df_gcm_ssp = get_val(dfs, ["GCM", "SSP"])
                df_gcm_lulc = get_val(dfs, ["GCM", "LULC"])
                df_ssp_lulc = get_val(dfs, ["SSP", "LULC"])

                def omega2(ss_eff, df_eff):
                    if sst + mse == 0:
                        return 0.0
                    return max(0.0, (ss_eff - df_eff * mse) / (sst + mse))

                w_gcm = omega2(ss_gcm, df_gcm)
                w_ssp = omega2(ss_ssp, df_ssp)
                w_lulc = omega2(ss_lulc, df_lulc)
                w_gcm_ssp = omega2(ss_gcm_ssp, df_gcm_ssp)
                w_gcm_lulc = omega2(ss_gcm_lulc, df_gcm_lulc)
                w_ssp_lulc = omega2(ss_ssp_lulc, df_ssp_lulc)

                total_w = w_gcm + w_ssp + w_lulc + w_gcm_ssp + w_gcm_lulc + w_ssp_lulc
                
                records.append({
                    "Month": month, "Period": period,
                    "GCM_%": 100 * w_gcm,
                    "SSP_%": 100 * w_ssp,
                    "LULC_%": 100 * w_lulc,
                    "GCMxSSP_%": 100 * w_gcm_ssp,
                    "GCMxLULC_%": 100 * w_gcm_lulc,
                    "SSPxLULC_%": 100 * w_ssp_lulc,
                    "Residual_%": max(0.0, 100 * (1 - total_w)),
                    "GCM_F": f_gcm, "GCM_p": p_gcm,
                    "SSP_F": f_ssp, "SSP_p": p_ssp,
                    "LULC_F": f_lulc, "LULC_p": p_lulc,
                    "GCMxSSP_F": f_gcm_ssp, "GCMxSSP_p": p_gcm_ssp,
                    "GCMxLULC_F": f_gcm_lulc, "GCMxLULC_p": p_gcm_lulc,
                    "SSPxLULC_F": f_ssp_lulc, "SSPxLULC_p": p_ssp_lulc
                })
            else:
                records.append({
                    "Month": month, "Period": period,
                    "GCM_%": 100 * ss_gcm / sst if sst else 0,
                    "SSP_%": 100 * ss_ssp / sst if sst else 0,
                    "LULC_%": 100 * ss_lulc / sst if sst else 0,
                    "GCMxSSP_%": 100 * ss_gcm_ssp / sst if sst else 0,
                    "GCMxLULC_%": 100 * ss_gcm_lulc / sst if sst else 0,
                    "SSPxLULC_%": 100 * ss_ssp_lulc / sst if sst else 0,
                    "Residual_%": 100 * ss.get("Residual", 0.0) / sst if sst else 0,
                    "GCM_F": f_gcm, "GCM_p": p_gcm,
                    "SSP_F": f_ssp, "SSP_p": p_ssp,
                    "LULC_F": f_lulc, "LULC_p": p_lulc,
                    "GCMxSSP_F": f_gcm_ssp, "GCMxSSP_p": p_gcm_ssp,
                    "GCMxLULC_F": f_gcm_lulc, "GCMxLULC_p": p_gcm_lulc,
                    "SSPxLULC_F": f_ssp_lulc, "SSPxLULC_p": p_ssp_lulc
                })
            continue
        else:
            # Manual balanced 2x2x4 SS decomposition (stats omitted for brevity without statsmodels)
            grand_mean = sub["Delta_Flow"].mean()
            sst = ((sub["Delta_Flow"] - grand_mean) ** 2).sum()

            def main_ss(col):
                m = sub.groupby(col, observed=True)["Delta_Flow"].mean()
                n = sub.groupby(col, observed=True)["Delta_Flow"].count()
                return float(((m - grand_mean) ** 2 * n).sum())

            ss_gcm = main_ss("GCM")
            ss_ssp = main_ss("SSP")
            ss_lulc = main_ss("LULC")

            def two_way_ss(c1, c2, ss1, ss2):
                m = sub.groupby([c1, c2], observed=True)["Delta_Flow"].mean()
                n = sub.groupby([c1, c2], observed=True)["Delta_Flow"].count()
                ss_cells = float((((m - grand_mean) ** 2) * n).sum())
                return ss_cells - ss1 - ss2

            ss_gcm_ssp = two_way_ss("GCM", "SSP", ss_gcm, ss_ssp)
            ss_gcm_lulc = two_way_ss("GCM", "LULC", ss_gcm, ss_lulc)
            ss_ssp_lulc = two_way_ss("SSP", "LULC", ss_ssp, ss_lulc)

            explained = ss_gcm + ss_ssp + ss_lulc + ss_gcm_ssp + ss_gcm_lulc + ss_ssp_lulc
            ss_resid = max(sst - explained, 0.0)

            sst_total = explained + ss_resid
            if sst_total == 0: continue

            records.append({
                "Month": month, "Period": period,
                "GCM_%": 100 * ss_gcm / sst_total,
                "SSP_%": 100 * ss_ssp / sst_total,
                "LULC_%": 100 * ss_lulc / sst_total,
                "GCMxSSP_%": 100 * ss_gcm_ssp / sst_total,
                "GCMxLULC_%": 100 * ss_gcm_lulc / sst_total,
                "SSPxLULC_%": 100 * ss_ssp_lulc / sst_total,
                "Residual_%": 100 * ss_resid / sst_total,
                "GCM_F": float("nan"), "GCM_p": float("nan"),
                "SSP_F": float("nan"), "SSP_p": float("nan"),
                "LULC_F": float("nan"), "LULC_p": float("nan"),
                "GCMxSSP_F": float("nan"), "GCMxSSP_p": float("nan"),
                "GCMxLULC_F": float("nan"), "GCMxLULC_p": float("nan"),
                "SSPxLULC_F": float("nan"), "SSPxLULC_p": float("nan")
            })

    if not records:
        return pd.DataFrame() 

    out = pd.DataFrame.from_records(records)
    out["Month"] = pd.Categorical(out["Month"], categories=MONTH_ORDER, ordered=True)
    out["Period"] = pd.Categorical(out["Period"], categories=PERIOD_ORDER, ordered=True)
    return out.sort_values(["Period", "Month"]).reset_index(drop=True)
# --------------------------------------------------------------------------
# 5. PROBABILITY OF CHANGE / EXTREME SCENARIOS / FLOW DURATION CURVES
# --------------------------------------------------------------------------

def probability_of_change(tidy, group_cols=("Month",)):
    """
    For each group (default: Month), the proportion of future ensemble
    members whose Flow exceeds the historical value for that month.
    """
    hist = get_historical(tidy)
    fut = get_future(tidy).copy()
    hist_map = {str(k): v for k, v in hist.items()}
    fut["Hist"] = fut["Month"].astype(str).map(hist_map).astype(float)
    fut["Exceeds"] = fut["Flow"] > fut["Hist"]

    out = fut.groupby(list(group_cols), observed=True)["Exceeds"].mean().reset_index()
    out["Exceeds"] = 100 * out["Exceeds"]
    out = out.rename(columns={"Exceeds": "Prob_Exceed_Historical_%"})
    return out


def flow_duration_curve(series):
    """Returns (exceedance_probability, sorted_flow) for a 1-D flow series."""
    s = pd.Series(series).dropna().sort_values(ascending=False).reset_index(drop=True)
    n = len(s)
    exceedance = 100 * (np.arange(1, n + 1) / (n + 1))
    return exceedance, s.values


def extreme_scenario_identification(tidy, top_n=5):
    """
    Identifies the Period x SSP x Climate_Model combinations producing the
    largest increase and largest decrease in mean annual flow relative to
    historical, using the LU_CC (combined) scenario set as the "full"
    scenario ensemble.
    """
    hist_annual = get_historical(tidy).mean()
    combo = tidy.loc[tidy["Scenario_Type"] == "LU_CC"]
    annual = combo.groupby(["Period", "SSP", "Climate_Model"], observed=True)["Flow"].mean().reset_index()
    annual["pct_change_annual"] = 100 * (annual["Flow"] - hist_annual) / hist_annual
    top_increase = annual.sort_values("pct_change_annual", ascending=False).head(top_n)
    top_decrease = annual.sort_values("pct_change_annual", ascending=True).head(top_n)
    return top_increase.reset_index(drop=True), top_decrease.reset_index(drop=True)


# --------------------------------------------------------------------------
# 6. PLOTTING
# --------------------------------------------------------------------------

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_hydrograph_uncertainty_band(tidy, outdir, channel_name):
    # 1. Fetch historical baseline
    hist = get_historical(tidy)

    # 2. Dynamically calculate stats grouped by Month, Period, and Scenario_Type
    stats = monthly_uncertainty_stats(tidy, group_cols=("Month", "Period", "Scenario_Type"))

    # 3. Identify available periods
    periods = [p for p in PERIOD_ORDER if p in stats["Period"].unique()]
    if not periods:
        return

    # Set up a 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=False)
    
    # Flatten the 2x2 array into a 1D list [ax1, ax2, ax3, ax4]
    axes_flat = axes.flatten()

    x = np.arange(len(MONTH_ORDER))
    
    # Standardized color palette for ALL scenarios
    colors = {"LU_Only": "seagreen", "CC_Only": "darkorange", "LU_CC": "purple"}

    for idx, period in enumerate(periods):
        ax = axes_flat[idx]
        
        # Plot Historical baseline (zorder=5 ensures it stays on top of the climate bands)
        ax.plot(x, hist.values, color="black", lw=0.5, marker="s", markersize=2, 
                label="Historical Baseline", zorder=500)

        # Filter the stats table for the current period loop
        sub_stats = stats[stats["Period"] == period]

        for stype, color in colors.items():
            # Filter for the specific scenario type and align with standard calendar order
            df_stype = sub_stats[sub_stats["Scenario_Type"] == stype].set_index("Month").reindex(MONTH_ORDER)

            if df_stype.empty or df_stype["Mean"].isna().all():
                continue

            # Plot the Uncertainty Band (5th - 95th percentile)
            ax.fill_between(x, df_stype["P05"], df_stype["P95"], color=color, alpha=0.15, 
                            label=f"{stype} (5-95%)")

            # Force LU_Only to the absolute front (zorder=6) with a larger marker 
            # so it is not eclipsed by the historical baseline.
            z_idx = 6 if stype == "LU_Only" else 3
            m_size = 6 if stype == "LU_Only" else 4

            # Plot the Ensemble Mean
            ax.plot(x, df_stype["Mean"], color=color, lw=2, marker="o", markersize=m_size, 
                    label=f"{stype} Mean", zorder=z_idx)

        # Apply formatting to the populated subplots
        ax.set_xticks(x)
        ax.set_xticklabels(MONTH_ORDER, rotation=45)
        ax.set_title(period.replace("_", " ").title(), fontweight="regular")
        ax.grid(alpha=0.3)
        
        # Only add the Y-axis label to the left-most plots (indices 0 and 2)
        if idx % 2 == 0:
            ax.set_ylabel("Flow ($m^3/s$)", fontweight="regular")

    # Handle the empty remaining axes (specifically the 4th quadrant)
    for i in range(len(periods), 4):
        axes_flat[i].axis('off')

    # Extract deduplicated handles for a clean global legend
    handles, labels = axes_flat[0].get_legend_handles_labels()
    
    # Place the legend in the exact center of the 4th quadrant
    # ncol=1 allows the items to stack in a clean list within that empty space
    legend_ax = axes_flat[3]
    legend_ax.legend(handles, labels, loc="center", frameon=False, fontsize=11, ncol=1)

    # Bring the super title down slightly so it breathes well with the 2x2 grid
    fig.suptitle(f"{channel_name}: Streamflow Uncertainty Envelopes", 
                 y=0.96, fontsize=12, fontweight="regular")
    
    # Use tight_layout, but explicitly carve out space at the top so the super title doesn't overlap
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    
    # Save the figure
    fig.savefig(os.path.join(outdir, f"1_{channel_name}_scenario_envelopes.png"), 
                dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_monthly_boxplots(tidy, outdir, channel_name):
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines

    fut = get_future(tidy)
    data = [fut.loc[fut["Month"] == m, "Flow"].dropna().values for m in MONTH_ORDER]
    hist = get_historical(tidy)

    # Slightly taller figure to accommodate the new legend and multi-line title
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#faf9f6")
    ax.set_facecolor("#faf9f6")

    box_kwargs = dict(
        patch_artist=True,
        widths=0.5,
        boxprops=dict(facecolor="#aec7e8", edgecolor="#3573b9", linewidth=1.4, alpha=0.85),
        medianprops=dict(color="#1f5fa8", linewidth=2.2),
        whiskerprops=dict(color="#9aa0a6", linewidth=1.3),
        capprops=dict(color="#9aa0a6", linewidth=1.3),
        flierprops=dict(marker="o", markerfacecolor='black', markeredgecolor="none",
                         markersize=4, alpha=0.6),
    )
    
    try:
        # matplotlib >= 3.9
        ax.boxplot(data, tick_labels=MONTH_ORDER, showfliers=True, **box_kwargs)
    except TypeError:
        # matplotlib < 3.9
        ax.boxplot(data, labels=MONTH_ORDER, showfliers=True, **box_kwargs)

    # Plot Historical
    ax.plot(np.arange(1, 13), hist.values, color="black", lw=2.4, marker="s",
            markersize=6, label="Historical baseline", zorder=5)

    ax.set_ylabel("Flow ($m^3/s$)", fontsize=11, fontweight="regular")
    
    # Updated Title explicitly stating the massive aggregation scope
    ax.set_title(f"{channel_name}: Monthly Distribution of Future Ensemble Flows\n"
                 f"(Across all scenarios and periods: LU_Only, CC_Only, LU_CC)",
                 fontsize=12, fontweight="regular", pad=15)

    # --- NEW CUSTOM EXPLANATORY LEGEND ---
    hist_line = mlines.Line2D([], [], color="black", lw=2.4, marker="s", markersize=6, label="Historical Baseline")
    median_line = mlines.Line2D([], [], color="#1f5fa8", linewidth=2.2, label="Ensemble Median (50th %)")
    iqr_patch = mpatches.Patch(facecolor="#aec7e8", edgecolor="#3573b9", linewidth=1.4, alpha=0.85, label="Interquartile Range (25th - 75th %)")
    whisker_line = mlines.Line2D([], [], color="#9aa0a6", linewidth=1.5, label="Expected Range (Whiskers)")
    outlier_dot = mlines.Line2D([], [], color="w", marker="o", markerfacecolor="#e8918f", markersize=8, label="Extreme Outliers")

    # Placed in the upper left with a solid white background so it doesn't get lost in the grid lines
    ax.legend(handles=[hist_line, median_line, iqr_patch, whisker_line, outlier_dot], 
              loc="upper left", frameon=True, fontsize=10, 
              facecolor="#ffffff", edgecolor="lightgrey", framealpha=0.95)
    # -------------------------------------

    ax.grid(axis="y", color="lightgrey", alpha=0.5, linewidth=0.8)
    ax.set_axisbelow(True)
    
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
        
    ax.tick_params(axis="x", labelsize=11, colors="#555555")
    ax.tick_params(axis="y", labelsize=11, colors="#555555")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"2_{channel_name}_monthly_boxplots.png"),
                dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_pct_change_bars(pct_summary, outdir, channel_name):
    # Identify available periods
    periods = [p for p in PERIOD_ORDER if p in pct_summary["Period"].unique()]
    if not periods:
        return

    # Set up a 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=False)
    
    # Flatten the 2x2 array into a 1D list so we can loop through it easily [ax1, ax2, ax3, ax4]
    axes_flat = axes.flatten()

    x = np.arange(len(MONTH_ORDER))
    width = 0.2
    
    # Map the columns to a clean color palette
    columns = ["Land_Use_%", "Climate_%", "Combined_%", "Interaction_%"]
    colors = ["seagreen", "darkorange", "purple", "grey"]

    # Loop through the periods and plot on the first 3 axes
    for idx, period in enumerate(periods):
        ax = axes_flat[idx]
        sub = pct_summary[pct_summary["Period"] == period].set_index("Month").reindex(MONTH_ORDER)

        for i, (col, color) in enumerate(zip(columns, colors)):
            ax.bar(x + (i - 1.5) * width, sub[col], width, 
                   label=col.replace("_%", "").replace("_", " "), color=color)

        # Apply formatting to the populated subplots
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(MONTH_ORDER, rotation=45)
        ax.set_title(period.replace("_", " ").title(), fontweight="regular")
        ax.grid(axis="y", alpha=0.3)
        
        # Only add the Y-axis label to the left-most plots (indices 0 and 2)
        if idx % 2 == 0:
            ax.set_ylabel("% Change vs. Historical", fontweight="regular")

    # Handle the empty remaining axes (specifically the 4th quadrant)
    for i in range(len(periods), 4):
        axes_flat[i].axis('off')  # Hides the bounding box, ticks, and background

    # Extract deduplicated handles for the legend from the first populated plot
    handles, labels = axes_flat[0].get_legend_handles_labels()
    
    # Place the legend in the exact center of the 4th quadrant
    legend_ax = axes_flat[3]
    legend_ax.legend(handles, labels, loc="center", frameon=False, fontsize=12)

    # Bring the super title down slightly so it breathes well with the 2x2 grid
    fig.suptitle(f"{channel_name}: Monthly % Change by Source and Period", 
                 y=0.96, fontsize=12, fontweight="regular")
    
    # Use tight_layout, but explicitly carve out space at the top so the super title doesn't overlap
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    
    # Save the figure
    fig.savefig(os.path.join(outdir, f"3_{channel_name}_pct_change_bars.png"), 
                dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_combined_ssp_mean_gcm_spread(tidy, outdir, channel_name):
    # 1. Filter for the two target scenarios
    target_scenarios = ["CC_Only", "LU_CC"]
    df = tidy[tidy["Scenario_Type"].isin(target_scenarios)].copy()
    if df.empty:
        return

    # 2. Fetch historical baseline
    hist = get_historical(tidy)

    # 3. Aggregate to get the mean of SSPs and min/max spread across GCMs
    agg = df.groupby(["Scenario_Type", "Period", "SSP", "Month"], observed=True)["Flow"].agg(
        mean="mean", gcm_min="min", gcm_max="max"
    ).reset_index()

    # 4. Identify available periods
    periods = [p for p in PERIOD_ORDER if p in df["Period"].unique()]
    if not periods:
        return

    # 5. Set up the 2x3 subplot grid (2 rows for scenarios, 3 cols for periods)
    fig, axes = plt.subplots(2, len(periods), figsize=(5 * len(periods), 8), 
                             sharey=True, sharex=True)
    
    # Standardized color palette
    ssp_colors = {"SSP2-4.5": "tab:green", "SSP5-8.5": "tab:purple"}
    x = np.arange(len(MONTH_ORDER))

    # 6. Loop through rows (Scenarios) and columns (Periods)
    for row_idx, scenario in enumerate(target_scenarios):
        for col_idx, period in enumerate(periods):
            # Handle indexing gracefully depending on array shape
            ax = axes[row_idx, col_idx] if len(periods) > 1 else axes[row_idx]
            
            # Plot Historical baseline in every subplot
            ax.plot(x, hist.values, color="black", lw=1.8, linestyle="--", marker="s",
                    markersize=4, label="Historical Baseline", zorder=5)

            # Filter data for the specific grid cell
            sub = agg[(agg["Scenario_Type"] == scenario) & (agg["Period"] == period)]
            
            for ssp, color in ssp_colors.items():
                g = sub[sub["SSP"] == ssp].set_index("Month").reindex(MONTH_ORDER)
                if g["mean"].isna().all():
                    continue
                
                # Plot the GCM Spread Band
                ax.fill_between(x, g["gcm_min"], g["gcm_max"], color=color, alpha=0.2,
                                label=f"{ssp} (GCM Spread)")
                
                # Plot the SSP Mean Line
                ax.plot(x, g["mean"], color=color, lw=2.2, marker="o", markersize=4,
                        label=f"{ssp} Mean")
            
            # Formatting for the specific subplot
            ax.set_xticks(x)
            ax.set_xticklabels(MONTH_ORDER, rotation=45)
            ax.grid(alpha=0.3)
            
            # Add Period titles only to the top row
            if row_idx == 0:
                ax.set_title(period.replace("_", " ").title(), fontweight="regular")
                
            # Add Scenario labels to the Y-axis of the first column
            if col_idx == 0:
                display_name = "Climate Only" if scenario == "CC_Only" else "Climate + Land Use"
                ax.set_ylabel(f"{display_name}\nFlow ($m^3/s$)", fontweight="regular")

    # 7. Extract and deduplicate legend handles
    handles, labels = axes[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    # Place a clean, single-row global legend at the bottom
    fig.legend(by_label.values(), by_label.keys(), loc="lower center", 
               bbox_to_anchor=(0.5, -0.05), ncol=5, frameon=False, fontsize=11)

    # 8. Set the super title
    fig.suptitle(f"{channel_name}: Climate (SSP) Means and GCM Spread", 
                 y=1, fontsize=12, fontweight="regular")
    
    # 9. Let tight_layout handle the internal padding
    fig.tight_layout()
    
    # 10. Save the figure
    fig.savefig(os.path.join(outdir, f"4_{channel_name}_combined_ssp_mean_gcm_spread.png"), 
                dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_combined_gcm_mean_ssp_spread(tidy, outdir, channel_name):
    # 1. Filter for the two target scenarios
    target_scenarios = ["CC_Only", "LU_CC"]
    df = tidy[tidy["Scenario_Type"].isin(target_scenarios)].copy()
    if df.empty:
        return

    # 2. Fetch historical baseline
    hist = get_historical(tidy)

    # 3. Aggregate to get the mean of GCMs and min/max spread across SSPs
    agg = df.groupby(["Scenario_Type", "Period", "Climate_Model", "Month"], observed=True)["Flow"].agg(
        mean="mean", ssp_min="min", ssp_max="max"
    ).reset_index()

    # 4. Identify available periods
    periods = [p for p in PERIOD_ORDER if p in df["Period"].unique()]
    if not periods:
        return

    # 5. Set up the 2x3 subplot grid (2 rows for scenarios, 3 cols for periods)
    fig, axes = plt.subplots(2, len(periods), figsize=(5 * len(periods), 8), 
                             sharey=True, sharex=True)
    
    # Standardized color palette for the 4 GCMs
    gcm_colors = {
        "Cool-Dry": "tab:blue", 
        "Cool-Wet": "tab:cyan",
        "Hot-Dry": "tab:red", 
        "Hot-Wet": "tab:orange"
    }
    x = np.arange(len(MONTH_ORDER))

    # 6. Loop through rows (Scenarios) and columns (Periods)
    for row_idx, scenario in enumerate(target_scenarios):
        for col_idx, period in enumerate(periods):
            # Handle indexing gracefully depending on array shape
            ax = axes[row_idx, col_idx] if len(periods) > 1 else axes[row_idx]
            
            # Plot Historical baseline in every subplot
            ax.plot(x, hist.values, color="black", lw=1.8, linestyle="--", marker="s",
                    markersize=4, label="Historical Baseline", zorder=5)

            # Filter data for the specific grid cell
            sub = agg[(agg["Scenario_Type"] == scenario) & (agg["Period"] == period)]
            
            for gcm, color in gcm_colors.items():
                g = sub[sub["Climate_Model"] == gcm].set_index("Month").reindex(MONTH_ORDER)
                if g["mean"].isna().all():
                    continue
                
                # Plot the SSP Spread Band
                ax.fill_between(x, g["ssp_min"], g["ssp_max"], color=color, alpha=0.20,
                                label=f"{gcm} (SSP Spread)")
                
                # Plot the GCM Mean Line
                ax.plot(x, g["mean"], color=color, lw=2.2, marker="o", markersize=4,
                        label=f"{gcm} Mean")
            
            # Formatting for the specific subplot
            ax.set_xticks(x)
            ax.set_xticklabels(MONTH_ORDER, rotation=45)
            ax.grid(alpha=0.3)
            
            # Add Period titles only to the top row
            if row_idx == 0:
                ax.set_title(period.replace("_", " ").title(), fontweight="regular")
                
            # Add Scenario labels to the Y-axis of the first column
            if col_idx == 0:
                display_name = "Climate Only" if scenario == "CC_Only" else "Climate + Land Use"
                ax.set_ylabel(f"{display_name}\nFlow ($m^3/s$)", fontweight="regular")

    # 7. Extract and deduplicate legend handles
    handles, labels = axes[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    # Place a clean, two-row global legend at the bottom (to handle 9 items nicely)
    fig.legend(by_label.values(), by_label.keys(), loc="lower center", 
               bbox_to_anchor=(0.5, -0.08), ncol=5, frameon=False, fontsize=11)

    # 8. Set the super title
    fig.suptitle(f"{channel_name}: Climate Model (GCM) Means and SSP Spread", 
                 y=1, fontsize=12, fontweight="regular")
    
    # 9. Let tight_layout handle the internal padding
    fig.tight_layout()
    
    # 10. Save the figure
    fig.savefig(os.path.join(outdir, f"5_{channel_name}_combined_gcm_mean_ssp_spread.png"), 
                dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_heatmaps(stats, anova_table, outdir, channel_name):
    # Set up the 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # ---------------------------------------------------------
    # 1. Top-Left: Standard Deviation Heatmap
    # ---------------------------------------------------------
    piv_std = stats.set_index("Month").reindex(MONTH_ORDER)[["Std"]].T
    im0 = axes[0, 0].imshow(piv_std.values, cmap="Reds", aspect="auto")
    axes[0, 0].set_xticks(range(len(MONTH_ORDER)))
    axes[0, 0].set_xticklabels(MONTH_ORDER, rotation=45)
    axes[0, 0].set_yticks([0])
    axes[0, 0].set_yticklabels(["Std Dev"])
    axes[0, 0].set_title("Monthly Std. Deviation ($m^3/s$)")
    fig.colorbar(im0, ax=axes[0, 0], shrink=0.6)

    # ---------------------------------------------------------
    # Helper Function for ANOVA Heatmaps
    # ---------------------------------------------------------
    # Identify available periods and format them for the Y-axis labels
    periods = [p for p in PERIOD_ORDER if p in anova_table["Period"].unique()]
    period_labels = [p.replace("_", " ").title() for p in periods]

    def plot_var_heatmap(ax, factor_col, title):
        # Pivot the data: Rows = Periods, Columns = Months
        piv = anova_table.pivot(index="Period", columns="Month", values=factor_col).reindex(
            index=periods, columns=MONTH_ORDER)
        
        # Use vmin=0 and vmax=100 so the color scale is uniform across all three plots
        im = ax.imshow(piv.values, cmap="viridis", aspect="auto", vmin=0, vmax=100)
        
        # Formatting
        ax.set_xticks(range(len(MONTH_ORDER)))
        ax.set_xticklabels(MONTH_ORDER, rotation=45)
        ax.set_yticks(range(len(periods)))
        ax.set_yticklabels(period_labels)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.6)

    # ---------------------------------------------------------
    # 2. Top-Right: GCM Variance Contribution
    # ---------------------------------------------------------
    plot_var_heatmap(axes[0, 1], "GCM_%", "GCM Variance Contribution (%)")

    # ---------------------------------------------------------
    # 3. Bottom-Left: SSP Variance Contribution
    # ---------------------------------------------------------
    plot_var_heatmap(axes[1, 0], "SSP_%", "SSP Variance Contribution (%)")

    # ---------------------------------------------------------
    # 4. Bottom-Right: LULC Variance Contribution
    # ---------------------------------------------------------
    plot_var_heatmap(axes[1, 1], "LULC_%", "LULC Variance Contribution (%)")

    # Add a global super title
    fig.suptitle(f"{channel_name}: Streamflow Uncertainty Statistics and Variance Decomposition", 
                 fontsize=12, fontweight="regular", y=0.98)
    
    fig.tight_layout()
    # Add a slight top margin so the tight_layout doesn't overlap the super title
    fig.subplots_adjust(top=0.92)
    
    fig.savefig(os.path.join(outdir, f"7_{channel_name}_heatmaps.png"), dpi=200)
    plt.close(fig)


def plot_variance_contribution(anova_table, outdir, channel_name, period="near_future"):
    """
    Plots a publication-quality stacked bar chart of variance contributions.
    """
    _ensure_dir(outdir)
    
    # Filter and reindex by month to ensure strict calendar order
    sub = anova_table.loc[anova_table["Period"] == period].set_index("Month").reindex(MONTH_ORDER)
    
    factors = ["GCM_%", "SSP_%", "LULC_%", "GCMxSSP_%", "GCMxLULC_%", "SSPxLULC_%", "Residual_%"]
    labels = ["Climate Model (GCM)", "Emission Scenario (SSP)", "Land Use (LULC)", 
              "GCM x SSP", "GCM x LULC", "SSP x LULC", "Residuals (Internal Var.)"]
    
    # Use standard colorblind-safe palette
    colors = plt.colormaps.get_cmap("tab20c").colors[:len(factors)]

    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(MONTH_ORDER))

    for i, factor in enumerate(factors):
        values = sub[factor].values
        ax.bar(MONTH_ORDER, values, bottom=bottom, label=labels[i], color=colors[i], alpha=0.9)
        bottom += values

    # Styling and layout
    ax.set_ylabel("Variance Fraction (%)", fontsize=11, fontweight="regular")
    ax.set_xlabel("Month", fontsize=11, fontweight="regular")
    ax.set_title(f"{channel_name} Streamflow Uncertainty Attribution ({period.replace('_', ' ').title()})", 
                 fontsize=12, fontweight="regular", pad=15)
    
    # Move the legend outside the plot to prevent covering data points
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, frameon=True)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, f"8_{channel_name}_uncertainty_decomposition_{period}.png"), 
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved upgraded uncertainty plot to {outdir}")


def plot_seasonal_uncertainty(seasonal_stats, outdir, channel_name):
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(seasonal_stats))
    ax.bar(x, seasonal_stats["Mean"], yerr=[seasonal_stats["Mean"] - seasonal_stats["P05"],
                                             seasonal_stats["P95"] - seasonal_stats["Mean"]],
           capsize=5, color="cadetblue")
    ax.set_xticks(x)
    ax.set_xticklabels(seasonal_stats["Season"])
    ax.set_ylabel("Flow ($m^3/s$)")
    ax.set_title(f"{channel_name}: Seasonal Uncertainty (mean, 5-95% range)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"9_{channel_name}_seasonal_uncertainty.png"), dpi=200)
    plt.close(fig)


def plot_flow_duration_curve(tidy, outdir, channel_name):
    hist = get_historical(tidy)
    fut = get_future(tidy)

    fig, ax = plt.subplots(figsize=(8, 6))
    exc_h, q_h = flow_duration_curve(hist.values)
    ax.plot(exc_h, q_h, color="black", lw=2, label="Historical")

    for stype, color in {"LU_Only": "seagreen", "CC_Only": "darkorange", "LU_CC": "purple"}.items():
        vals = fut.loc[fut["Scenario_Type"] == stype].groupby("Run_ID")["Flow"].apply(list)
        all_vals = np.concatenate(vals.values) if len(vals) else np.array([])
        if len(all_vals) == 0:
            continue
        exc, q = flow_duration_curve(all_vals)
        ax.plot(exc, q, color=color, lw=1.5, alpha=0.8, label=stype)

    ax.set_yscale("log")
    ax.set_xlabel("Exceedance probability (%)")
    ax.set_ylabel("Flow ($m^3/s$, log scale)")
    ax.set_title(f"{channel_name}: Flow Duration Curve Comparison")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{channel_name}_flow_duration_curve.png"), dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
# 7. ORCHESTRATOR
# --------------------------------------------------------------------------

def run_full_analysis(csv_path, channel_name, replace_name, outdir):
    """
    Runs the complete UQ + source-attribution workflow for one channel and
    writes all tables (CSV) and figures (PNG) to `outdir`.
    """
    _ensure_dir(outdir)

    # 1. Reshape
    tidy = load_wide_csv(csv_path, channel_name)
    channel_name = replace_name if replace_name else channel_name
    tidy.round(2).to_csv(os.path.join(outdir, f"{channel_name}_tidy_long.csv"), index=False)

    # 2. Monthly uncertainty stats (overall, and split by Scenario_Type)
    stats_overall = monthly_uncertainty_stats(tidy)
    stats_overall.round(2).to_csv(os.path.join(outdir, f"{channel_name}_monthly_stats_overall.csv"), index=False)

    stats_by_type = monthly_uncertainty_stats(tidy, group_cols=("Month", "Scenario_Type"))
    stats_by_type.round(2).to_csv(os.path.join(outdir, f"{channel_name}_monthly_stats_by_scenario_type.csv"), index=False)

    stats_by_period_type = monthly_uncertainty_stats(tidy, group_cols=("Month", "Period", "Scenario_Type"))
    stats_by_period_type.round(2).to_csv(os.path.join(outdir, f"{channel_name}_monthly_stats_by_period_and_type.csv"), index=False)
    
    # Seasonal stats
    seasonal_stats = seasonal_uncertainty_stats(tidy)
    seasonal_stats.round(2).to_csv(os.path.join(outdir, f"{channel_name}_seasonal_stats.csv"), index=False)

    # 3. Source attribution
    attrib = source_attribution(tidy)
    attrib.round(2).to_csv(os.path.join(outdir, f"{channel_name}_source_attribution_full.csv"), index=False)

    pct_summary = monthly_pct_change_summary(attrib)
    pct_summary.round(2).to_csv(os.path.join(outdir, f"{channel_name}_pct_change_summary.csv"), index=False)

    # 4. Variance-based attribution (3-way ANOVA)
    anova_table = three_way_variance_decomposition(tidy, method="eta2")
    anova_table.round(2).to_csv(os.path.join(outdir, f"{channel_name}_variance_decomposition.csv"), index=False)

    anova_table_omega2 = three_way_variance_decomposition(tidy, method="omega2")
    anova_table_omega2.round(2).to_csv(os.path.join(outdir, f"{channel_name}_variance_decomposition_omega2.csv"), index=False)

    # 5. Probability of change / extremes / FDC data
    prob_change = probability_of_change(tidy)
    prob_change.round(2).to_csv(os.path.join(outdir, f"{channel_name}_probability_of_change.csv"), index=False)

    top_inc, top_dec = extreme_scenario_identification(tidy)
    top_inc.round(2).to_csv(os.path.join(outdir, f"{channel_name}_extreme_top_increase.csv"), index=False)
    top_dec.round(2).to_csv(os.path.join(outdir, f"{channel_name}_extreme_top_decrease.csv"), index=False)

    # 6. Figures
    plot_hydrograph_uncertainty_band(tidy, outdir, channel_name)
    plot_monthly_boxplots(tidy, outdir, channel_name)
    plot_pct_change_bars(pct_summary, outdir, channel_name)
    plot_combined_ssp_mean_gcm_spread(tidy, outdir, channel_name)
    plot_combined_gcm_mean_ssp_spread(tidy, outdir, channel_name)
    plot_heatmaps(stats_overall, anova_table, outdir, channel_name)
    for period in PERIOD_ORDER:
        if period in anova_table["Period"].unique():
            plot_variance_contribution(anova_table, outdir, channel_name, period=period)
    plot_seasonal_uncertainty(seasonal_stats, outdir, channel_name)
    plot_flow_duration_curve(tidy, outdir, channel_name)

    print(f"[{channel_name}] Analysis complete. Outputs written to: {outdir}")
    return {
        "tidy": tidy,
        "monthly_stats": stats_overall,
        "seasonal_stats": seasonal_stats,
        "attribution": attrib,
        "pct_summary": pct_summary,
        "anova": anova_table,
        "prob_change": prob_change,
        "top_increase": top_inc,
        "top_decrease": top_dec,
    }