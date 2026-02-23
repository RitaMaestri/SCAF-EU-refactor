import numpy as np
import pandas as pd


def build_exo_idx(sectors, exo_mask):
    """
    Build exo_idx containing ONLY exogenous items.

    exo_idx structure:
      - scalar exogenous variable:
          exo_idx[var] = ("scalar", True)
      - 1D exogenous sectors (mask length == len(sectors)):
          exo_idx[var] = ("vector", {"ENERGY", "SERVICES", ...})
      - 1D exogenous sectors (mask length < len(sectors)):
          exo_idx[var] = ("vector_short", {"ENERGY", ...}, N)
          where N = len(sectors). Sector names are mapped from the
          available mask entries only; any growth-ratio row for this
          variable will raise an "undefined position" error.
      - 2D exogenous pairs (row-major):
          exo_idx[var] = ("matrix", {("A","B"), ("C","D"), ...})

    Note: sets are used to make membership checks easy/fast.
    """
    exo_idx = {}

    for var, mask in exo_mask.items():

        # Scalar
        if isinstance(mask, bool):
            if mask:
                exo_idx[var] = ("scalar", True)
            continue

        # 1D
        if mask.ndim == 1:
            n = min(len(mask), len(sectors))
            exo_set = {sectors[i] for i in range(n) if bool(mask[i])}
            if exo_set:
                if len(mask) < len(sectors):
                    # Position of exogenous sectors is ambiguous beyond the mask length.
                    # Store N so the consistency checker can produce a precise error.
                    exo_idx[var] = ("vector_short", exo_set, len(sectors))
                else:
                    exo_idx[var] = ("vector", exo_set)
            continue

        # 2D
        if mask.ndim == 2:
            if mask.shape != (len(sectors), len(sectors)):
                raise ValueError("2D mask for %r has shape %r but expected %r"
                                 % (var, mask.shape, (len(sectors), len(sectors))))
            idx = np.argwhere(mask)
            idx = idx[np.lexsort((idx[:, 1], idx[:, 0]))]  # row-major
            exo_set = {(sectors[i], sectors[j]) for i, j in idx}
            if exo_set:
                exo_idx[var] = ("matrix", exo_set)
            continue

        raise ValueError("Mask for %r with ndim=%d is not supported" % (var, mask.ndim))

    return exo_idx


def check_growth_ratios_consistency(exo_idx, growth_ratios,
                                            variable_col="variable",
                                            sector1_col="Sector_1",
                                            sector2_col="Sector_2"):
    """
    Validate that every row in growth_ratios refers to an EXOGENOUS element by checking:
      - scalar: var must be present in exo_idx
      - vector: var present and sector present in exo_idx[var] sector set
      - matrix: var present and (s1,s2) present in exo_idx[var] pair set

    Also enforces the expected filling of Sector_1/Sector_2 for each dimension.
    """
    df = growth_ratios

    def is_empty(x):
        if pd.isna(x):
            return True
        if isinstance(x, str) and x.strip() == "":
            return True
        return False

    for ridx, row in df.iterrows():
        var = row.get(variable_col)
        if pd.isna(var):
            raise ValueError("Row %d: variable is NaN" % ridx)
        var = str(var)

        s1 = row.get(sector1_col, np.nan)
        s2 = row.get(sector2_col, np.nan)

        # If the variable isn't exogenous at all -> error immediately
        if var not in exo_idx:
            raise ValueError("Row %d: %r is not exogenous (not in exo_idx)" % (ridx, var))

        entry = exo_idx[var]
        dimension, allowed = entry[0], entry[1]

        if dimension == "scalar":
            # For scalars, just presence in exo_idx is the exogeneity check
            if (not is_empty(s1)) or (not is_empty(s2)):
                raise ValueError("Row %d: %r is scalar but Sector_1/Sector_2 are not empty" % (ridx, var))

        elif dimension == "vector_short":
            # The mask is shorter than the sector list: sector-to-position mapping is
            # unreliable. Any growth-ratio row specifying a sector is ambiguous.
            N = entry[2]
            sec = str(s1) if not is_empty(s1) else "<unspecified>"
            raise ValueError(
                "Undefined position of sector %r within array %r of %d elements" % (sec, var, N)
            )

        elif dimension == "vector":
            if is_empty(s1) or (not is_empty(s2)):
                raise ValueError("Row %d: %r is 1D: Sector_1 must be filled and Sector_2 must be empty" % (ridx, var))
            sec = str(s1)
            if sec not in allowed:
                raise ValueError("Row %d: %r sector %r is not exogenous" % (ridx, var, sec))

        elif dimension == "matrix":
            if is_empty(s1) or is_empty(s2):
                raise ValueError("Row %d: %r is 2D: both Sector_1 and Sector_2 must be filled" % (ridx, var))
            pair = (str(s1), str(s2))
            if pair not in allowed:
                raise ValueError("Row %d: %r pair %r is not exogenous" % (ridx, var, pair))

        else:
            raise ValueError("Unknown dimension for %r: %r" % (var, dimension))


def build_exogenous_output_exo_only(sectors, exo_idx, growth_ratios, years,
                                   variable_col="variable",
                                   sector1_col="Sector_1",
                                   sector2_col="Sector_2"):
    """
    Build the output dict { var: array } using exo_idx that contains ONLY exogenous items.

    Ordering rule for output rows:
      - vector: sectors appear in the same order as `sectors`, filtered to exogenous ones
      - matrix: pairs appear in row-major order (i index then j index)
    """
    df = growth_ratios.copy()

    years = list(years)
    for y in years:
        if y not in df.columns:
            raise ValueError("Year column %r not found in growth_ratios columns" % y)

    T = len(years)
    ones_T = np.ones(T, dtype=float)

    # --- lookup helpers (minimal; assumes dataframe already validated) ---
    def get_scalar_series(var):
        sub = df[df[variable_col].astype(str) == var]
        if sub.empty:
            return None
        if len(sub) > 1:
            raise ValueError("Scalar variable %r appears multiple times in growth_ratios" % var)
        return sub.iloc[0][years].to_numpy(dtype=float)

    def get_vector_series(var, sector):
        sub = df[df[variable_col].astype(str) == var]
        sub = sub[sub[sector1_col].astype(str) == sector]
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

    # --- build output ---
    out = {}

    for var, entry in exo_idx.items():
        dimension, allowed = entry[0], entry[1]

        if dimension == "scalar":
            ser = get_scalar_series(var)
            out[var] = ser if ser is not None else ones_T.copy()

        elif dimension in ("vector", "vector_short"):
            # For "vector_short" variables no growth-ratio rows exist (they would have
            # been rejected by the consistency checker), so every sector defaults to 1.
            exo_sectors_ordered = [s for s in sectors if s in allowed]
            arr = np.empty((len(exo_sectors_ordered), T), dtype=float)
            for i, sec in enumerate(exo_sectors_ordered):
                ser = get_vector_series(var, sec)
                arr[i, :] = ser if ser is not None else ones_T
            out[var] = arr

        elif dimension == "matrix":
            # Preserve row-major order using `sectors` x `sectors`
            exo_pairs_ordered = []
            for s1 in sectors:
                for s2 in sectors:
                    if (s1, s2) in allowed:
                        exo_pairs_ordered.append((s1, s2))

            arr = np.empty((len(exo_pairs_ordered), T), dtype=float)
            for i, (s1, s2) in enumerate(exo_pairs_ordered):
                ser = get_matrix_series(var, s1, s2)
                arr[i, :] = ser if ser is not None else ones_T
            out[var] = arr

        else:
            raise ValueError("Unknown dimension for %r: %r" % (var, dimension))

    return out


def build_exogenous_timeseries(sectors, exo_mask, growth_ratios, years,
                              variable_col="variable",
                              sector1_col="Sector_1",
                              sector2_col="Sector_2"):
    """
    Convenience wrapper:
      1) build exo_idx (exogenous-only)
      2) validate growth_ratios against exo_idx
      3) build output dict
    """
    exo_idx = build_exo_idx(sectors, exo_mask)
    check_growth_ratios_consistency(
        exo_idx, growth_ratios,
        variable_col=variable_col, sector1_col=sector1_col, sector2_col=sector2_col
    )
    out = build_exogenous_output_exo_only(
        sectors, exo_idx, growth_ratios, years,
        variable_col=variable_col, sector1_col=sector1_col, sector2_col=sector2_col
    )
    return out