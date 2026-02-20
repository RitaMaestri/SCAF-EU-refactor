import numpy as np
import pandas as pd


import numpy as np
import pandas as pd

import numpy as np
import pandas as pd


def build_exo_dict(sectors, exo_mask):
    """
    Build exo_dict containing ONLY exogenous items.

    exo_dict structure:
      - scalar exogenous variable:
          exo_dict[var] = ("scalar", True)
      - 1D exogenous sectors (full mapping known):
          exo_dict[var] = ("vector", {"ENERGY", "SERVICES", ...})
      - 2D exogenous pairs (full mapping known, row-major):
          exo_dict[var] = ("matrix", {("A","B"), ("C","D"), ...})

    Special case: masks shorter than the global sectors dimension (mapping unknown).
      - 1D short mask:
          exo_dict[var] = ("vector_short", {"N": N, "exo_count": k})
      - 2D short mask (must be square):
          exo_dict[var] = ("matrix_short", {"N": N, "exo_count": k})

    Note: sets are used to make membership checks easy/fast.
    """
    exo_dict = {}

    for var, m in exo_mask.items():

        # Scalar
        if isinstance(m, bool):
            if m:
                exo_dict[var] = ("scalar", True)
            continue

        # 1D
        if m.ndim == 1:
            N = len(m)
            k = int(np.sum(m))

            if k == 0:
                continue  # fully endogenous -> not included

            if N == len(sectors):
                exo_set = {sectors[i] for i in range(len(sectors)) if bool(m[i])}
                if exo_set:
                    exo_dict[var] = ("vector", exo_set)
            else:
                # mapping to sector names is unknown -> keep only N and how many exogenous items exist
                exo_dict[var] = ("vector_short", {"N": N, "exo_count": k})
            continue

        # 2D
        if m.ndim == 2:
            n0, n1 = m.shape
            k = int(np.sum(m))

            if k == 0:
                continue  # fully endogenous -> not included

            if (n0, n1) == (len(sectors), len(sectors)):
                idx = np.argwhere(m)
                idx = idx[np.lexsort((idx[:, 1], idx[:, 0]))]  # row-major
                exo_set = {(sectors[i], sectors[j]) for i, j in idx}
                if exo_set:
                    exo_dict[var] = ("matrix", exo_set)
            else:
                # mapping unknown -> keep only N and how many exogenous items exist
                if n0 != n1:
                    raise ValueError("Short 2D mask for %r must be square, got shape %r" % (var, m.shape))
                exo_dict[var] = ("matrix_short", {"N": n0, "exo_count": k})
            continue

        raise ValueError("Mask for %r with ndim=%d is not supported" % (var, m.ndim))

    return exo_dict




def check_growth_ratios_consistency(exo_dict, growth_ratios,
                                   variable_col="variable",
                                   sector1_col="Sector_1",
                                   sector2_col="Sector_2",
                                   sectors=None):
    df = growth_ratios

    has_variable_col = (variable_col in df.columns)
    has_sector_cols = (sector1_col in df.columns) and (sector2_col in df.columns)

    def is_empty(x):
        if pd.isna(x):
            return True
        if isinstance(x, str) and x.strip() == "":
            return True
        return False

    for ridx, row in df.iterrows():
        # Variable comes from a column (long) OR from the index (your current df)
        var = row.get(variable_col) if has_variable_col else ridx
        if pd.isna(var):
            raise ValueError("Row %s: variable is NaN" % str(ridx))
        var = str(var)

        # If the variable isn't exogenous at all -> error immediately
        if var not in exo_dict:
            raise ValueError("Row %s: %r is not exogenous (not in exo_dict)" % (str(ridx), var))

        dimension, allowed = exo_dict[var]

        # If we do not have sector columns, we cannot validate sector-level membership.
        # We only validate "scalar must not have sector info" when sector cols exist.
        if not has_sector_cols:
            # Nothing more to check here (sector info is absent by construction)
            continue

        s1 = row.get(sector1_col, np.nan)
        s2 = row.get(sector2_col, np.nan)

        if dimension == "scalar":
            if (not is_empty(s1)) or (not is_empty(s2)):
                raise ValueError("Row %s: %r is scalar but Sector_1/Sector_2 are not empty" % (str(ridx), var))

        elif dimension == "vector":
            if is_empty(s1) or (not is_empty(s2)):
                raise ValueError("Row %s: %r is 1D: Sector_1 must be filled and Sector_2 must be empty" % (str(ridx), var))
            sec = str(s1)
            if sec not in allowed:
                raise ValueError("Row %s: %r sector %r is not exogenous" % (str(ridx), var, sec))

        elif dimension == "matrix":
            if is_empty(s1) or is_empty(s2):
                raise ValueError("Row %s: %r is 2D: both Sector_1 and Sector_2 must be filled" % (str(ridx), var))
            pair = (str(s1), str(s2))
            if pair not in allowed:
                raise ValueError("Row %s: %r pair %r is not exogenous" % (str(ridx), var, pair))

        elif dimension == "vector_short":
            # Error ONLY if growth_ratios defines a sector for it
            if (not is_empty(s1)) or (not is_empty(s2)):
                if sectors is None:
                    raise ValueError(
                        "growth_ratios defines a sector for variable %r, but its exo_mask is shorter than sectors; "
                        "pass `sectors` to report the sector position." % var
                    )
                sec = str(s1) if not is_empty(s1) else ""
                N = int(allowed["N"])
                if sec not in sectors:
                    raise ValueError("Row %s: sector %r is not in the global sectors list" % (str(ridx), sec))
                pos0 = sectors.index(sec)
                raise ValueError(
                    "Sector '%s' is at position %d in the global sectors list, which is outside variable '%s' array of N=%d elements."
                    % (sec, pos0, var, N)
                )

        elif dimension == "matrix_short":
            # Error ONLY if growth_ratios defines a pair for it
            if (not is_empty(s1)) or (not is_empty(s2)):
                if sectors is None:
                    raise ValueError(
                        "growth_ratios defines a sector pair for variable %r, but its exo_mask is shorter than sectors; "
                        "pass `sectors` to report the sector position." % var
                    )
                a = str(s1) if not is_empty(s1) else ""
                b = str(s2) if not is_empty(s2) else ""
                N = int(allowed["N"])
                if a not in sectors or b not in sectors:
                    raise ValueError("Row %s: (%r,%r) not in the global sectors list" % (str(ridx), a, b))
                i0 = sectors.index(a)
                j0 = sectors.index(b)
                raise ValueError(
                    "Pair ('%s','%s') has indices (%d,%d) in the global sectors list, which is outside variable '%s' array of N=%d×%d elements."
                    % (a, b, i0, j0, var, N, N)
                )

        else:
            raise ValueError("Unknown dimension for %r: %r" % (var, dimension))


def build_exogenous_output_exo_only(sectors, exo_dict, growth_ratios, years,
                                   variable_col="variable",
                                   sector1_col="Sector_1",
                                   sector2_col="Sector_2"):
    df = growth_ratios.copy()
    years = list(years)
    T = len(years)
    ones_T = np.ones(T, dtype=float)

    has_variable_col = (variable_col in df.columns)
    has_sector_cols = (sector1_col in df.columns) and (sector2_col in df.columns)

    # --- scalar lookup works both in long and in your index-based format ---
    def get_scalar_series(var):
        if not has_variable_col:
            # index-based
            if var not in df.index:
                return None
            sub = df.loc[[var], years]
            if len(sub) > 1:
                raise ValueError("Scalar variable %r appears multiple times in growth_ratios" % var)
            return sub.iloc[0].to_numpy(dtype=float)

        # long format
        sub = df[df[variable_col].astype(str) == var]
        if sub.empty:
            return None
        if len(sub) > 1:
            raise ValueError("Scalar variable %r appears multiple times in growth_ratios" % var)
        return sub.iloc[0][years].to_numpy(dtype=float)

    # --- vector/matrix lookup only if sector columns exist (long format) ---
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

    out = {}

    for var, (dimension, allowed) in exo_dict.items():

        if dimension == "scalar":
            ser = get_scalar_series(var)
            out[var] = ser if ser is not None else ones_T.copy()

        elif dimension == "vector":
            exo_sectors_ordered = [s for s in sectors if s in allowed]
            arr = np.empty((len(exo_sectors_ordered), T), dtype=float)

            if has_sector_cols and has_variable_col:
                # long format -> we can lookup by sector
                for i, sec in enumerate(exo_sectors_ordered):
                    ser = get_vector_series(var, sec)
                    arr[i, :] = ser if ser is not None else ones_T
            else:
                # no sector info available -> fill ones for all exogenous elements
                arr[:, :] = 1.0

            out[var] = arr

        elif dimension == "matrix":
            exo_pairs_ordered = []
            for s1 in sectors:
                for s2 in sectors:
                    if (s1, s2) in allowed:
                        exo_pairs_ordered.append((s1, s2))

            arr = np.empty((len(exo_pairs_ordered), T), dtype=float)

            if has_sector_cols and has_variable_col:
                for i, (s1, s2) in enumerate(exo_pairs_ordered):
                    ser = get_matrix_series(var, s1, s2)
                    arr[i, :] = ser if ser is not None else ones_T
            else:
                arr[:, :] = 1.0

            out[var] = arr

        elif dimension == "vector_short":
            k = int(allowed["exo_count"])
            out[var] = np.ones((k, T), dtype=float)

        elif dimension == "matrix_short":
            k = int(allowed["exo_count"])
            out[var] = np.ones((k, T), dtype=float)

        else:
            raise ValueError("Unknown dimension for %r: %r" % (var, dimension))

    return out


def build_exogenous_timeseries(sectors, exo_mask, growth_ratios, years,
                              variable_col="variable",
                              sector1_col="Sector_1",
                              sector2_col="Sector_2"):
    """
    Convenience wrapper:
      1) build exo_dict (exogenous-only, supports short masks)
      2) validate growth_ratios against exo_dict (special short-mask rule)
      3) build output dict
    """
    exo_dict = build_exo_dict(sectors, exo_mask)

    check_growth_ratios_consistency(
        exo_dict, growth_ratios,
        variable_col=variable_col, sector1_col=sector1_col, sector2_col=sector2_col,
        sectors=sectors
    )

    out = build_exogenous_output_exo_only(
        sectors, exo_dict, growth_ratios, years,
        variable_col=variable_col, sector1_col=sector1_col, sector2_col=sector2_col
    )

    return out