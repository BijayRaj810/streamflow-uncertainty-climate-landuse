# Streamflow Uncertainty Analysis under Climate and Land-Use Change

**Results highlights** — integrated climate / land-use / hydrological uncertainty assessment for a
monsoon-dominated rain-fed river basin

*Conducted as part of a larger hydrology and climate-risk consultancy assessment. Full code and
methodology: see this repository.*

---

This project quantifies how much future streamflow in a monsoon-dominated, rain-fed river basin
(~6,500–7,000 km², spanning steep headwaters to lowland Terai-type plains) is expected to change
under climate and land-use change through 2100, and — critically — how much of that projected
change is genuinely uncertain versus how much is driven by which specific climate model or
land-use scenario is assumed.

| 35 yrs | 3 | 8 | 24 |
|---|---|---|---|
| observed daily climate & discharge baseline (1985–2020) | gauging stations, upstream → downstream, sequentially calibrated | representative CMIP6 GCM×SSP combinations, screened from 28 | integrated future scenarios per channel (4 GCMs × 2 SSPs × 3 horizons) |

## 1. Approach

- **Hydrological model:** semi-distributed SWAT+ model, calibrated upstream→downstream with Sobol
  global sensitivity analysis (23 parameters) and Dynamically Dimensioned Search optimization.
- **Climate ensemble:** 3-step CMIP6 screening (seasonal cycle → annual bias → four-corner
  envelope) narrowed 28 GCMs to 8 representative Hot-Wet / Hot-Dry / Cool-Wet / Cool-Dry
  combinations under SSP2-4.5 and SSP5-8.5.
- **Bias correction:** robust empirical quantile mapping for precipitation, linear (mean+variance)
  transformation for temperature.
- **Land-use change:** CA-Markov / neural-network transition modelling projected land cover from
  2000–2020 observations forward to 2100, validated against an independent year (histogram
  accuracy 90%, location accuracy 77%).
- **Uncertainty quantification:** a purpose-built, open-source Python toolkit separates
  climate-only, land-use-only, and combined effects, and decomposes variance by source (ω²-based
  ANOVA).

## 2. Model performance

The calibrated hydrological model reproduces observed discharge with satisfactory-to-very-good
skill at all three stations: daily Nash–Sutcliffe Efficiency of 0.67–0.77, improving to 0.80–0.93
at the monthly time step. A small, systematic bias emerges and widens moving downstream (≈0% at
the upstream station to ≈−7% at the downstream station) — flagged explicitly for any application
sensitive to absolute flow volumes, rather than smoothed over.

## 3. Key findings

**01 — Monsoon intensification dominates the climate signal, not directional uncertainty.**
Across the 8-member ensemble, temperature increases by +1.6°C to +6.2°C depending on emissions
pathway, and precipitation changes range from −2% to +57%. Every "dry" corner still stays close to
neutral — the real uncertainty is *how much wetter* the monsoon gets, not whether the basin trends
wetter or drier.

**02 — Land-use change has already happened, mostly.**
Forest cover rose from 61.0% to 66.8% of the basin area between 2000 and 2020 (cropland fell from
25.0% to 20.9%); the land-use projection framework shows this trend converging toward equilibrium
after ~2030, meaning most of the century's land-cover-driven hydrological change is already in the
historical record, not still ahead.

**03 — Which uncertainty source matters depends entirely on the season.**
At the representative mid-basin station, climate-model structure alone explains 82–89% of
monsoon-season flow-projection variance — but in the dry season and shoulder months, land-use
assumptions and climate-model × emissions-pathway interaction effects dominate instead. A single
"climate uncertainty" number would misrepresent both halves of the year.

**04 — All 24 integrated future scenarios agree on the annual direction, not the magnitude.**
Every evaluated combination of GCM, emissions pathway, and land-use scenario projects an
**increase** in mean annual flow relative to the historical baseline, ranging from **+17%** to
**+239%** — a spread of more than 14×. The direction is not in question; the magnitude very much
is.

> **Season-dependent risk hidden inside a positive annual mean**
> In the pre-monsoon months, fewer than half of individual ensemble members project flow above the
> historical baseline — even though the ensemble mean for those months stays positive. In the most
> extreme case (far-future, one month), even the ensemble mean itself turns negative. An
> annual-scale increase does not guarantee a seasonal one — this is exactly the kind of risk a
> mean-only summary would hide.

## 4. Methodological contribution

Beyond the basin-specific results, the project produced a reusable, open-source Python toolkit for
uncertainty quantification and source attribution in climate-hydrology impact studies:
variance-based decomposition (GCM × emissions pathway × land-use, bias-corrected ω²),
scenario-differencing source attribution (climate-only / land-use-only / combined / interaction
effects), probability-of-exceedance framing alongside ensemble means, and flow-duration-curve
comparison across scenario types — designed to be channel-agnostic and reusable for other basins
and studies. Code, pipeline, and documentation are published in this repository.

---

`Climate` `Hydrology` `Uncertainty Quantification` `SWAT+` `CMIP6`
