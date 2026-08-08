"""
swat_extraction_utils.py
=========================
Shared helpers for extracting SWAT+ channel outflow output into tidy
daily/monthly/seasonal/annual summaries.

Used by 9_data_extraction_from_swat_output.ipynb for all three scenario
groups (historical baseline, climate-change-only, land-use + climate-change)
-- these used to be ~80 lines of identical function definitions, copy-pasted
once per group.
"""

import shutil
from pathlib import Path

import pandas as pd

MONTH_NAMES = {
    1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
    7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec',
}

# Season order kept explicit so output rows are chronological, not alphabetical
SEASON_ORDER = ['DJF', 'MAM', 'JJAS', 'ON']
SEASON_MAP = {
    12: 'DJF', 1: 'DJF', 2: 'DJF',          # Winter
    3: 'MAM', 4: 'MAM', 5: 'MAM',           # Pre-monsoon
    6: 'JJAS', 7: 'JJAS', 8: 'JJAS', 9: 'JJAS',  # Monsoon
    10: 'ON', 11: 'ON',                     # Post-monsoon
}

REQUIRED_OUTPUTS = [
    'lsunit_wb_aa.csv', 'lsunit_wb_mon.csv', 'lsunit_wb_yr.csv',
    'channel_sdmorph_day.csv', 'channel_mon_sdmorph.csv',
    'channel_sdmorph_yr.csv', 'channel_sdmorph_aa.csv',
]


def get_paths(model_scenario: str, inside_scenario: str, output_name: str | None = None) -> tuple[Path, Path, Path]:
    """Build the source/output/pivot directories for one scenario.

    `inside_scenario` is the actual QSWAT+/SWAT+ scenario folder name under
    `Scenarios/` (e.g. "Calibrated_model", "245_Cool-Dry"). `output_name` lets
    the output folder use a different, more descriptive label than the raw
    scenario name -- needed for the historical baseline, whose QSWAT+
    scenario is named "Calibrated_model" but whose output folder is
    "historical_climate" (to read consistently alongside the named future
    scenario folders). Defaults to `inside_scenario` when not given, which is
    correct for every scenario except the historical baseline.
    """
    output_name = output_name or inside_scenario
    source_dir = Path(f'../SWATPlus Models/{model_scenario}/Scenarios/{inside_scenario}/TxtInOut')
    output_dir = Path(f'../SWATPlus Models/SWAT_Outputs/{model_scenario}/{output_name}')
    pivot_dir = output_dir / 'Channel_combined'
    return source_dir, output_dir, pivot_dir


def copy_required_outputs(source_dir: Path, output_dir: Path,
                           filenames: list[str] = REQUIRED_OUTPUTS) -> None:
    """Copy each required SWAT+ output file, warning if any are missing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in filenames:
        src, dst = source_dir / name, output_dir / name
        if src.exists():
            shutil.copy(src, dst)
            print(f"Copied {name}")
        else:
            print(f"Warning: {name} not found in {source_dir}")


def load_daily_channel_data(csv_path: Path, channels: list[int]) -> pd.DataFrame:
    """Load channel_sdmorph_day.csv, clean it, and filter to the target channels."""
    df = pd.read_csv(csv_path, skiprows=1)
    df.columns = df.columns.str.strip()

    numeric_cols = ['jday', 'mon', 'day', 'yr', 'unit', 'flo_out']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Drop non-data rows (e.g. the units line SWAT+ often includes)
    df = df.dropna(subset=['yr', 'mon', 'day', 'unit']).copy()
    df[['yr', 'mon', 'day']] = df[['yr', 'mon', 'day']].astype(int)

    df.insert(0, 'Date', pd.to_datetime(
        df[['yr', 'mon', 'day']].rename(columns={'yr': 'year', 'mon': 'month'})
    ))

    target_ids = {int(str(c).replace('cha', '')) for c in channels}
    return df[df['unit'].isin(target_ids)].copy()


def rename_channel_cols(df: pd.DataFrame, id_col: str | None) -> pd.DataFrame:
    """Rename numeric channel columns (e.g. 53 -> cha53_flo_out).

    id_col is left untouched if given; pass None when every column is a
    channel (e.g. the single-row annual average has no label column).
    """
    df.columns = [
        col if col == id_col else f"cha{int(col)}_flo_out"
        for col in df.columns
    ]
    return df


def build_daily_pivot(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot_table(index='Date', columns='unit', values='flo_out').reset_index()
    return rename_channel_cols(pivot, 'Date')


def build_monthly_pivot(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(Month=df['Date'].dt.to_period('M').dt.to_timestamp())
    pivot = df.groupby(['Month', 'unit'])['flo_out'].mean().unstack().reset_index()
    return rename_channel_cols(pivot, 'Month')


def build_longterm_monthly_avg(df: pd.DataFrame) -> pd.DataFrame:
    """Average flow per calendar month (Jan-Dec) across all years."""
    pivot = (
        df.groupby([df['Date'].dt.month.rename('DateMonth'), 'unit'])['flo_out']
        .mean()
        .unstack()
        .reset_index()
    )
    pivot = rename_channel_cols(pivot, 'DateMonth')
    pivot['DateMonth'] = pivot['DateMonth'].map(MONTH_NAMES)
    return pivot


def build_seasonal_avg(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(Season=df['Date'].dt.month.map(SEASON_MAP))
    pivot = df.groupby(['Season', 'unit'])['flo_out'].mean().unstack().reset_index()
    pivot = rename_channel_cols(pivot, 'Season')
    pivot['Season'] = pd.Categorical(pivot['Season'], categories=SEASON_ORDER, ordered=True)
    return pivot.sort_values('Season').reset_index(drop=True)


def build_annual_avg(df: pd.DataFrame) -> pd.DataFrame:
    """Long-term annual average: mean of each year's average flow, across
    all years present. Averaging year-means (rather than all days at
    once) avoids bias from any partial first/last year in the record.
    """
    yearly_means = df.groupby([df['Date'].dt.year, 'unit'])['flo_out'].mean().unstack()
    annual_avg = yearly_means.mean().to_frame().T
    return rename_channel_cols(annual_avg, id_col=None)


def split_into_periods(df: pd.DataFrame,
                        periods: list[tuple[str, pd.Timestamp, pd.Timestamp | None]]) -> dict[str, pd.DataFrame]:
    """Split a daily dataframe into near/mid/far future slices by date.

    An end date of None means "open ended" -- runs through the last
    date present in df (handles far-future runs that don't all stop
    on the same calendar date).
    """
    periods_out = {}
    for name, start, end in periods:
        period_end = end if end is not None else df['Date'].max()
        mask = (df['Date'] >= start) & (df['Date'] <= period_end)
        periods_out[name] = df.loc[mask].copy()
    return periods_out


def process_scenario(model_scenario: str, inside_scenario: str, channels: list[int],
                      periods=None, round_to: int = 3, output_name: str | None = None,
                      required_outputs: list[str] = REQUIRED_OUTPUTS) -> None:
    """Run the full extraction + summary pipeline for one scenario.

    Pass `periods` (a list of (name, start, end) tuples, see
    `split_into_periods`) for scenario groups that span multiple future
    periods within a single run -- e.g. the climate-change-only group, which
    holds land use fixed and runs 2015-2100 continuously, so near/mid/far
    have to be split out here. Leave `periods` as None for scenario groups
    where each run already corresponds to exactly one period (the historical
    baseline; and the land-use+climate-change group, where the LULC model
    folder itself -- 2040/2065/2090 -- already picks out near/mid/far, so
    there is nothing left to split within one run).

    `output_name` overrides the output folder label (see `get_paths`) --
    needed for the historical baseline, see its docstring.
    """
    print(f"\n=== Processing scenario: {inside_scenario} ===")
    source_dir, output_dir, pivot_dir = get_paths(model_scenario, inside_scenario, output_name)

    copy_required_outputs(source_dir, output_dir, required_outputs)
    filtered_df = load_daily_channel_data(output_dir / 'channel_sdmorph_day.csv', channels)

    pivot_dir.mkdir(parents=True, exist_ok=True)

    full_record_outputs = {
        'channel_daily_combined.csv': build_daily_pivot(filtered_df),
        'channel_monthly_combined.csv': build_monthly_pivot(filtered_df),
    }
    for filename, result_df in full_record_outputs.items():
        result_df.round(round_to).to_csv(pivot_dir / filename, index=False)
        print(f"Saved {filename}")

    if periods is None:
        full_record_summary = {
            'channel_longterm_monthly_avg.csv': build_longterm_monthly_avg(filtered_df),
            'channel_seasonal_avg.csv': build_seasonal_avg(filtered_df),
            'channel_annual_avg.csv': build_annual_avg(filtered_df),
        }
        for filename, result_df in full_record_summary.items():
            result_df.round(round_to).to_csv(pivot_dir / filename, index=False)
            print(f"Success! Saved {filename} to: {pivot_dir}")
        return

    for period_name, period_df in split_into_periods(filtered_df, periods).items():
        if period_df.empty:
            print(f"Warning: no data found for period '{period_name}' -- skipping")
            continue

        period_dir = pivot_dir / period_name
        period_dir.mkdir(parents=True, exist_ok=True)

        date_range = f"{period_df['Date'].min().date()} to {period_df['Date'].max().date()}"
        print(f"{period_name}: {date_range} ({period_df['Date'].dt.year.nunique()} years)")

        period_outputs = {
            'channel_longterm_monthly_avg.csv': build_longterm_monthly_avg(period_df),
            'channel_seasonal_avg.csv': build_seasonal_avg(period_df),
            'channel_annual_avg.csv': build_annual_avg(period_df),
        }
        for filename, result_df in period_outputs.items():
            result_df.round(round_to).to_csv(period_dir / filename, index=False)
            print(f"  Saved {period_name}/{filename}")
