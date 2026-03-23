from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_kl_weights(kl_folder: str | Path) -> pd.DataFrame:
    """Sum all CSV weight files in *kl_folder* cell-by-cell.

    Each CSV is a labour/capital allocation matrix with ISO country code rows
    and GTAP commodity code columns, separated by ``|``.

    Parameters
    ----------
    kl_folder:
        Path to the directory containing the KL weight CSV files.

    Returns
    -------
    pd.DataFrame
        Element-wise sum of all weight matrices (rows=ISO countries,
        cols=GTAP commodity codes).
    """
    kl_folder = Path(kl_folder)
    total: pd.DataFrame | None = None
    for csv_file in sorted(kl_folder.glob("*.csv")):
        df = pd.read_csv(csv_file, sep="|", index_col=0)
        total = df if total is None else total.add(df, fill_value=0)
    if total is None:
        raise FileNotFoundError(f"No CSV files found in {kl_folder}")
    return total


def aggregate_by_region(
    elasticities: pd.Series,
    weights: pd.DataFrame,
    r_map: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Broadcast uniform elasticities to aggregated regions and sum weights.

    Because the elasticities are identical for every country, the
    region-aggregated elasticities are simply the original values broadcast
    to all regions.  The region-aggregated weights are computed by summing
    the individual-country weight rows within each aggregated region.

    Parameters
    ----------
    elasticities:
        Uniform elasticities indexed by GTAP commodity codes.
    weights:
        Weight matrix with ISO country rows and GTAP commodity columns.
    r_map:
        Mapping with columns ``ISO Code``, ``Aggregated Region``.

    Returns
    -------
    r_agg_el:
        Broadcast elasticities (rows=aggregated regions, cols=GTAP commodities).
    r_agg_weights:
        Summed weights (rows=aggregated regions, cols=GTAP commodities).
    """
    agg_reg_names = sorted(set(r_map["Aggregated Region"]))
    gtap_cols = weights.columns

    r_agg_el = pd.DataFrame(index=agg_reg_names, columns=gtap_cols, dtype=float)
    r_agg_weights = pd.DataFrame(index=agg_reg_names, columns=gtap_cols, dtype=float)

    for reg in agg_reg_names:
        iso_codes = list(r_map.loc[r_map["Aggregated Region"] == reg, "ISO Code"])
        present = [c for c in iso_codes if c in weights.index]
        r_agg_el.loc[reg] = elasticities.reindex(gtap_cols)
        r_agg_weights.loc[reg] = weights.loc[present].sum(axis=0).reindex(gtap_cols)

    return r_agg_el, r_agg_weights


def aggregate_by_sector(
    r_agg_el: pd.DataFrame,
    r_agg_weights: pd.DataFrame,
    s_map: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate from GTAP commodities to SCAF sectors using weight-averaged means.

    Parameters
    ----------
    r_agg_el:
        Broadcast elasticities (regions × GTAP commodities).
    r_agg_weights:
        Summed region weights (regions × GTAP commodities).
    s_map:
        Mapping with columns ``GTAP Sectors``, ``GTAP aggregation``,
        ``SCAF aggregation``.

    Returns
    -------
    sr_agg_el:
        Sector-aggregated elasticities (regions × SCAF sectors).
    sr_agg_weights:
        Sector-aggregated weights (regions × SCAF sectors).
    """
    scaf_sectors = sorted(set(s_map["SCAF aggregation"]))
    sr_agg_el = pd.DataFrame(index=r_agg_el.index, columns=scaf_sectors, dtype=float)
    sr_agg_weights = pd.DataFrame(index=r_agg_el.index, columns=scaf_sectors, dtype=float)

    for reg in sr_agg_el.index:
        for sec in sr_agg_el.columns:
            gtap_sectors = list(s_map.loc[s_map["SCAF aggregation"] == sec, "GTAP Sectors"])
            gtap_sectors = [s for s in gtap_sectors if s in r_agg_weights.columns]
            w = r_agg_weights.loc[reg, gtap_sectors]
            e = r_agg_el.loc[reg, gtap_sectors].reindex(w.index)
            total_w = w.sum()
            sr_agg_el.at[reg, sec] = (e * w).sum() / total_w if total_w != 0 else float("nan")
            sr_agg_weights.at[reg, sec] = total_w

    return sr_agg_el, sr_agg_weights
