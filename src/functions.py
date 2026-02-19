import numpy as np
import pandas as pd


def build_exogenous_timeseries(sectors, exo_mask, growth_ratios, years,
                              variable_col="variable",
                              sector1_col="Sector_1",
                              sector2_col="Sector_2"):
    """
    Build a dict { var: array } of exogenous growth-ratio time series.

    Output shapes:
      - scalars: (T,)
      - 1D sectoral vectors: (N_exo, T) following the order in `sectors`
      - 2D: (N_exo, T) where N_exo = number of True pairs in the mask (row-major)

    Validation rule:
      Every row in growth_ratios must refer to an EXOGENOUS element according to exo_mask,
      otherwise raise ValueError.

    Missing entries in growth_ratios are allowed for exogenous variables/elements:
      they will be filled with ones (1.0) over the whole time horizon.
    """

    df = growth_ratios.copy()

    T = len(years)
    ones_T = np.ones(T, dtype=float)


    # --- helper: list exogenous elements in the correct order ---
    def exo_elements(var):
        """
        returns (kind, elements)
          kind = "scalar" | "vector" | "matrix"
          elements:
            scalar -> ["__scalar__"] if True, otherwise []
            vector -> list of exogenous sectors (in `sectors` order)
            matrix -> list of exogenous pairs (s1,s2) in row-major order
        """
        if var not in exo_mask:
            raise ValueError("Variable %r is in growth_ratios but not in exo_mask" % var)

        m = exo_mask[var]

        if isinstance(m, bool):
            return "scalar", (["__scalar__"] if m else [])

        if m.ndim == 1:
            if len(m) != len(sectors):
                raise ValueError("1D mask for %r has length %d but sectors has length %d"
                                 % (var, len(m), len(sectors)))
            elems = [sectors[i] for i in range(len(sectors)) if bool(m[i])]
            return "vector", elems

        if m.ndim == 2:
            if m.shape != (len(sectors), len(sectors)):
                raise ValueError("2D mask for %r has shape %r but expected %r"
                                 % (var, m.shape, (len(sectors), len(sectors))))
            idx = np.argwhere(m)
            # Sort by (i,j) -> row-major
            idx = idx[np.lexsort((idx[:, 1], idx[:, 0]))]
            elems = [(sectors[i], sectors[j]) for i, j in idx]
            return "matrix", elems

        raise ValueError("Mask for %r with ndim=%d is not supported" % (var, m.ndim))

    # Precompute exogeneity info (fail fast if some mask is invalid)
    exo_info = {}
    for var in exo_mask.keys():
        exo_info[var] = exo_elements(var)

    # --- helper to check whether a sector cell is “empty” ---
    def is_empty(x):
        if pd.isna(x):
            return True
        if isinstance(x, str) and x.strip() == "":
            return True
        return False

    # =========================
    # 1) VALIDATE growth_ratios
    # =========================
    for ridx, row in df.iterrows():
        var = row.get(variable_col)


        if var not in exo_info:
            raise ValueError("Row %d: variable %r not in the model" % (ridx, var))

        kind, elems = exo_info[var]

        s1 = row.get(sector1_col, np.nan)
        s2 = row.get(sector2_col, np.nan)

        if kind == "scalar":
            if "__scalar__" not in elems:
                raise ValueError("growth_ratios contains %r (scalar) but exo_mask[%r] is False" % (var, var))
            if (not is_empty(s1)) or (not is_empty(s2)):
                raise ValueError("Row %d: %r is scalar but Sector_1/Sector_2 are not empty" % (ridx, var))

        elif kind == "vector":
            if is_empty(s1) or (not is_empty(s2)):
                raise ValueError("Row %d: %r is 1D: Sector_1 must be filled and Sector_2 must be empty" % (ridx, var))
            s1 = str(s1)
            if s1 not in elems:
                raise ValueError("Row %d: %r sector %r is NOT exogenous" % (ridx, var, s1))

        elif kind == "matrix":
            if is_empty(s1) or is_empty(s2):
                raise ValueError("Row %d: %r is 2D: both Sector_1 and Sector_2 must be filled" % (ridx, var))
            pair = (str(s1), str(s2))
            if pair not in elems:
                raise ValueError("Row %d: %r pair %r is NOT exogenous" % (ridx, var, pair))

        else:
            raise ValueError("Unknown kind for %r: %r" % (var, kind))

    # =========================
    # 2) LOOKUP series in the dataframe
    # =========================
    def get_scalar_series(var):
        sub = df[df[variable_col].astype(str) == var]
        sub = sub[(sub[sector1_col].apply(is_empty)) & (sub[sector2_col].apply(is_empty))]
        if sub.empty:
            return None
        if len(sub) > 1:
            raise ValueError("Multiple scalar rows found for %r in growth_ratios" % var)
        return sub.iloc[0][years].to_numpy(dtype=float)

    def get_vector_series(var, sector):
        sub = df[df[variable_col].astype(str) == var]
        sub = sub[sub[sector1_col].astype(str) == sector]
        sub = sub[sub[sector2_col].apply(is_empty)]  # 1D -> Sector_2 must be empty
        if sub.empty:
            return None
        if len(sub) > 1:
            raise ValueError("Multiple rows found for %r sector %r" % (var, sector))
        return sub.iloc[0][years].to_numpy(dtype=float)

    def get_matrix_series(var, s1, s2):
        sub = df[df[variable_col].astype(str) == var]
        sub = sub[(sub[sector1_col].astype(str) == s1) & (sub[sector2_col].astype(str) == s2)]
        if sub.empty:
            return None
        if len(sub) > 1:
            raise ValueError("Multiple rows found for %r pair (%r,%r)" % (var, s1, s2))
        return sub.iloc[0][years].to_numpy(dtype=float)

    # =========================
    # 3) BUILD OUTPUT
    # =========================
    out = {}

    for var, (kind, elems) in exo_info.items():
        if len(elems) == 0:
            continue  # No exogenous element -> do not include this variable

        if kind == "scalar":
            ser = get_scalar_series(var)
            out[var] = ser if ser is not None else ones_T.copy()

        elif kind == "vector":
            exo_sectors = elems  # list of strings
            arr = np.empty((len(exo_sectors), T), dtype=float)
            for i, sec in enumerate(exo_sectors):
                ser = get_vector_series(var, sec)
                arr[i, :] = ser if ser is not None else ones_T
            out[var] = arr

        elif kind == "matrix":
            exo_pairs = elems  # list of tuples (s1,s2)
            arr = np.empty((len(exo_pairs), T), dtype=float)
            for i, (s1, s2) in enumerate(exo_pairs):
                ser = get_matrix_series(var, s1, s2)
                arr[i, :] = ser if ser is not None else ones_T
            out[var] = arr

        else:
            raise ValueError("Unknown kind for %r: %r" % (var, kind))

    return out
