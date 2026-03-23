from __future__ import annotations

import pandas as pd


def aggregate_by_region(
    consumption: pd.DataFrame,
    income_el: pd.DataFrame,
    price_el: pd.DataFrame,
    s_map: pd.DataFrame,
    r_map: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Expand elasticities to full GTAP-sector resolution, then aggregate to
    the EUR aggregated regions using consumption-weighted averages.

    Parameters
    ----------
    consumption:
        Household consumption by ISO country (rows) × GTAP sector (cols).
    income_el, price_el:
        Elasticities indexed by ISO country (rows) × GTAP-aggregation sector (cols).
    s_map:
        Mapping with columns ``GTAP Sectors``, ``GTAP aggregation``, ``SCAF aggregation``.
    r_map:
        Mapping with columns ``ISO Code``, ``Aggregated Region``.

    Returns
    -------
    r_agg_ie, r_agg_pe, r_agg_cons
        Region-aggregated income elasticities, price elasticities, and consumption.
    """
    # Expand elasticities from GTAP-aggregation resolution to full GTAP sectors
    full_ie = pd.DataFrame(index=consumption.index, columns=consumption.columns, dtype=float)
    full_pe = pd.DataFrame(index=consumption.index, columns=consumption.columns, dtype=float)
    for reg in full_ie.index:
        for sec in full_ie.columns:
            disagg_sec = s_map.loc[s_map["GTAP Sectors"] == sec].iloc[0]["GTAP aggregation"]
            full_ie.at[reg, sec] = income_el.at[reg, disagg_sec]
            full_pe.at[reg, sec] = price_el.at[reg, disagg_sec]

    # Aggregate over ISO countries → aggregated regions
    agg_reg_names = sorted(set(r_map["Aggregated Region"]))
    r_agg_ie = pd.DataFrame(index=agg_reg_names, columns=consumption.columns, dtype=float)
    r_agg_pe = pd.DataFrame(index=agg_reg_names, columns=consumption.columns, dtype=float)
    r_agg_cons = pd.DataFrame(index=agg_reg_names, columns=consumption.columns, dtype=float)

    for sec in r_agg_ie.columns:
        for reg in r_agg_ie.index:
            regions = list(r_map.loc[r_map["Aggregated Region"] == reg, "ISO Code"])
            e_weights = consumption.loc[regions, sec]
            ie_elements = full_ie.loc[regions, sec].reindex(e_weights.index)
            pe_elements = full_pe.loc[regions, sec].reindex(e_weights.index)
            r_agg_ie.at[reg, sec] = (ie_elements * e_weights).sum() / e_weights.sum()
            r_agg_pe.at[reg, sec] = (pe_elements * e_weights).sum() / e_weights.sum()

    for agg_reg in r_agg_ie.index:
        reg_list = list(r_map.loc[r_map["Aggregated Region"] == agg_reg, "ISO Code"])
        r_agg_cons.loc[agg_reg] = (
            consumption.loc[consumption.index.isin(reg_list)]
            .sum(axis=0)
            .reindex(r_agg_cons.columns)
        )

    return r_agg_ie, r_agg_pe, r_agg_cons


def aggregate_by_sector(
    r_agg_ie: pd.DataFrame,
    r_agg_pe: pd.DataFrame,
    r_agg_cons: pd.DataFrame,
    s_map: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate region-level elasticities from GTAP sectors to SCAF sectors
    using consumption-weighted averages.

    Parameters
    ----------
    r_agg_ie, r_agg_pe, r_agg_cons:
        Outputs of :func:`aggregate_by_region`.
    s_map:
        Mapping with columns ``GTAP Sectors``, ``GTAP aggregation``, ``SCAF aggregation``.

    Returns
    -------
    sr_agg_ie, sr_agg_pe, sr_agg_cons
        Sector- and region-aggregated income elasticities, price elasticities, and consumption.
    """
    scaf_sectors = sorted(set(s_map["SCAF aggregation"]))
    sr_agg_ie = pd.DataFrame(index=r_agg_ie.index, columns=scaf_sectors, dtype=float)
    sr_agg_pe = pd.DataFrame(index=r_agg_ie.index, columns=scaf_sectors, dtype=float)
    sr_agg_cons = pd.DataFrame(index=r_agg_ie.index, columns=scaf_sectors, dtype=float)

    for reg in sr_agg_ie.index:
        for sec in sr_agg_ie.columns:
            sectors = list(s_map.loc[s_map["SCAF aggregation"] == sec, "GTAP Sectors"])
            e_weights = r_agg_cons.loc[reg, sectors]
            ie_elements = r_agg_ie.loc[reg, sectors].reindex(e_weights.index)
            pe_elements = r_agg_pe.loc[reg, sectors].reindex(e_weights.index)
            sr_agg_ie.at[reg, sec] = (ie_elements * e_weights).sum() / e_weights.sum()
            sr_agg_pe.at[reg, sec] = (pe_elements * e_weights).sum() / e_weights.sum()

    for agg_sector in sr_agg_cons.columns:
        sec_list = list(s_map.loc[s_map["SCAF aggregation"] == agg_sector, "GTAP Sectors"])
        sr_agg_cons.loc[:, agg_sector] = (
            r_agg_cons.loc[:, r_agg_cons.columns.isin(sec_list)]
            .sum(axis=1)
            .reindex(r_agg_cons.index)
        )

    return sr_agg_ie, sr_agg_pe, sr_agg_cons


def compute_compensated_price_elasticities(
    sr_agg_pe: pd.DataFrame,
    sr_agg_ie: pd.DataFrame,
    sr_agg_cons: pd.DataFrame,
) -> pd.DataFrame:
    """Compute compensated own-price elasticities via the Slutsky equation.

    For each aggregated region ``r`` and sector ``s``::

        compensated[r, s] = -uncompensated[r, s]
                            + income_el[r, s] * (cons[r, s] / cons[r, :].sum())

    Parameters
    ----------
    sr_agg_pe:
        Uncompensated own-price elasticities (output of :func:`aggregate_by_sector`).
    sr_agg_ie:
        Aggregated income elasticities (output of :func:`aggregate_by_sector`).
    sr_agg_cons:
        Aggregated consumption (output of :func:`aggregate_by_sector`).

    Returns
    -------
    pd.DataFrame
        Compensated own-price elasticities with the same index/columns as the inputs.
    """
    budget_shares = sr_agg_cons.div(sr_agg_cons.sum(axis=1), axis=0)
    return -sr_agg_pe + sr_agg_ie * budget_shares
