import pandas as pd 
import numpy as np

def remove_unnamed_labels(df):
    import pandas as pd
    import re

    # Fix column MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = pd.MultiIndex.from_tuples([
            tuple(
                None if (isinstance(x, str) and re.match(r'^Unnamed: \d+_level_\d+$', x)) else x
                for x in col
            )
            for col in df.columns
        ], names=df.columns.names)

    # Fix index MultiIndex
    if isinstance(df.index, pd.MultiIndex):
        df.index = pd.MultiIndex.from_tuples([
            tuple(
                None if (isinstance(x, str) and re.match(r'^Unnamed: \d+_level_\d+$', x)) else x
                for x in idx
            )
            for idx in df.index
        ], names=df.index.names)

    return df



def aggregate_sectors(df, in_sectors, out_sector):
    import pandas as pd

    df = df.copy()

    # --- Aggregate rows per index level 0 ---
    for lvl0 in df.index.get_level_values(0).unique():
        # Mask for rows to aggregate
        mask = (df.index.get_level_values(0) == lvl0) & (df.index.get_level_values(1).isin(in_sectors))
        indices_to_sum = df.index[mask]

        if len(indices_to_sum) == 0:
            continue

        # Sum them
        summed_row = df.loc[indices_to_sum].sum(min_count=1)

        # Get insertion point BEFORE deletion
        first_row_pos = df.index.to_list().index(indices_to_sum[0])

        # Drop original rows
        df = df.drop(index=indices_to_sum)

        # Insert new row at the same spot
        upper = df.iloc[:first_row_pos]
        lower = df.iloc[first_row_pos:]
        new_df = pd.DataFrame([summed_row.values], columns=df.columns)
        new_df.index = pd.MultiIndex.from_tuples([(lvl0, out_sector)], names=df.index.names)

        # Preserve empty names
        if df.index.names == [None, None]:
            new_df.index.names = [None, None]

        df = pd.concat([upper, new_df, lower])

    # --- Aggregate columns per index level 0 ---
    for lvl0 in df.columns.get_level_values(0).unique():
        mask = (df.columns.get_level_values(0) == lvl0) & (df.columns.get_level_values(1).isin(in_sectors))
        cols_to_sum = df.columns[mask]

        if len(cols_to_sum) == 0:
            continue

        summed_col = df.loc[:, cols_to_sum].sum(axis=1, min_count=1)

        # Get insert position BEFORE deletion
        first_col_pos = df.columns.to_list().index(cols_to_sum[0])

        # Drop original columns
        df = df.drop(columns=cols_to_sum)

        # Insert new column at the same spot
        left = df.iloc[:, :first_col_pos]
        right = df.iloc[:, first_col_pos:]
        new_series = pd.DataFrame(summed_col.values, columns=[(lvl0, out_sector)], index=df.index)
        new_series.columns = pd.MultiIndex.from_tuples([(lvl0, out_sector)], names=df.columns.names)

        # Preserve empty names
        if df.columns.names == [None, None]:
            new_series.columns.names = [None, None]

        df = pd.concat([left, new_series, right], axis=1)

    return df




