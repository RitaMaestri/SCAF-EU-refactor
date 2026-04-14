import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from lib import (
    EXIOBASE_flow_extractor,
    REMIND_time_series_extractor,
    replace_unnamed_multiindex_labels,
    convert_remind_price_to_exiobase,
    vol_prices_timeseries_to_df,
    compute_energy_trade_rhos,
)


import traceback

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from common.path_loader import load_config

# useful paths
module_dir = Path(__file__).resolve().parent
repo_root = module_dir.parents[2]
this_folder = str(module_dir)
config = load_config(module_dir)

REMIND_path = os.path.join(config["raw_data_root"], config["remind_file"])
IOTs_path = os.path.join(config["calibration_output_root"], config["iots_folder"])
out_path = os.path.join(config["calibration_output_root"], config["out_folder"])
conversion_mapping_path = config["conversion_mapping_path"]
calibration_year = config["calibration_year"]

# Extract EUR import-volume time series from REMIND using the EU trade mapping.
trade_mapping_path = os.path.join(this_folder, "mapping", "EU_energy_trade.csv")
trade_mapping = pd.read_csv(trade_mapping_path)

import_volume_variables = trade_mapping["import_volume"].tolist()
export_volume_variables = trade_mapping["export_volume"].tolist()
prices_variables= trade_mapping["price"].tolist()

REMIND_raw = pd.read_csv(REMIND_path, sep=";")
years_of_interest = [
    int(col) for col in REMIND_raw.columns if str(col).isdigit() and int(col) >= int(calibration_year)
]

REMIND_imports = REMIND_time_series_extractor(
    REMIND_raw,
    import_volume_variables,
    years_of_interest,
    "EUR",
)

REMIND_exports = REMIND_time_series_extractor(
    REMIND_raw,
    export_volume_variables,
    years_of_interest,
    "EUR",
)

REMIND_prices = REMIND_time_series_extractor(
    REMIND_raw,
    prices_variables,
    years_of_interest,
    "EUR",
)

year_cols = [str(year) for year in years_of_interest]

imports_by_year = REMIND_imports[year_cols]
exports_by_year = REMIND_exports[year_cols]
prices_by_year = REMIND_prices[year_cols]


### IMPORT ###

aggregated_energy_import_volumes_time_series = imports_by_year.sum(axis=0)

aggregated_energy_import_values_time_series = pd.Series(
    (imports_by_year.to_numpy() * prices_by_year.to_numpy()).sum(axis=0),
    index=year_cols,
)
## ffil propagates the last valid price forward in case the volume is 0
energy_aggregated_import_price_time_series = (
    aggregated_energy_import_values_time_series / aggregated_energy_import_volumes_time_series
).ffill()

### EXPORT ###

aggregated_energy_export_volumes_time_series = exports_by_year.sum(axis=0)

aggregated_energy_export_values_time_series = pd.Series(
    (exports_by_year.to_numpy() * prices_by_year.to_numpy()).sum(axis=0),
    index=year_cols,
)
## ffil propagates the last valid price forward in case the volume is 0
energy_aggregated_export_price_time_series = (
    aggregated_energy_export_values_time_series / aggregated_energy_export_volumes_time_series
).ffill()

### CACHE: energy trade values ###
energy_trade_values_cache_path = repo_root / config["energy_trade_values_cache"]
energy_trade_values_cache_path.parent.mkdir(parents=True, exist_ok=True)

_model = REMIND_raw["Model"].iloc[0]
_scenario = REMIND_raw["Scenario"].iloc[0]
_trade_unit = "bn US$2017"

energy_trade_values_df = pd.concat(
    [
        pd.DataFrame(
            {
                "Model": [_model, _model],
                "Scenario": [_scenario, _scenario],
                "Region": ["EUR", "EUR"],
                "Variable": [
                    "aggregated_energy_import_values",
                    "aggregated_energy_export_values",
                ],
                "Unit": [_trade_unit, _trade_unit],
            }
        ),
        pd.DataFrame(
            [
                aggregated_energy_import_values_time_series.to_numpy(),
                aggregated_energy_export_values_time_series.to_numpy(),
            ],
            columns=year_cols,
        ),
    ],
    axis=1,
)
energy_trade_values_df.to_csv(energy_trade_values_cache_path, index=False)

### RHOS ###
remind_prices_path = repo_root / config["remind_prices_by_energy_use"]
rho_rows = compute_energy_trade_rhos(
    import_price_ts=energy_aggregated_import_price_time_series,
    export_price_ts=energy_aggregated_export_price_time_series,
    remind_prices_by_energy_use_path=remind_prices_path,
    year_cols=year_cols,
)


IOT=pd.read_csv(
    str(IOTs_path)+"EUR.csv",
    header=[0, 1],
    index_col=[0, 1],
    keep_default_na=False,
)
IOT = replace_unnamed_multiindex_labels(IOT)

EXIOBASE_energy_imports = EXIOBASE_flow_extractor(IOT,
                            [("M","∑")],
                            [ ("Imp","ENERGY")],
                            new_row_indexes=["M"],
                            new_col_indexes=["ENERGY"]
                            )

EXIOBASE_energy_export = EXIOBASE_flow_extractor(IOT,
                            [("CI_imp","ENERGY")],
                            [("","EXP")],
                            new_row_indexes=["ENERGY"],
                            new_col_indexes=["X"]
                            )

##################
##### OUTPUT #####
##################



#prices comparison
prices_comparison = pd.DataFrame({
    "REMIND": convert_remind_price_to_exiobase(energy_aggregated_import_price_time_series[calibration_year], conversion_mapping_path),
    "EXIOBASE": EXIOBASE_energy_imports / aggregated_energy_import_volumes_time_series[calibration_year]
}
)
prices_comparison.index = ["Energy price (M EUR/EJ)"]

Path(out_path).mkdir(parents=True, exist_ok=True)

prices_comparison.to_csv(os.path.join(out_path, "prices_comparison.csv"))

#energy trade projections
EXIOBASE_energy_import_price=(float(EXIOBASE_energy_imports) / 
                              aggregated_energy_import_volumes_time_series[calibration_year])

REMIND_to_EXIOBASE_import_price_conversion_factor=EXIOBASE_energy_import_price / energy_aggregated_import_price_time_series[calibration_year]


EXIOBASE_energy_export_price=(float(EXIOBASE_energy_export) / 
                              aggregated_energy_export_volumes_time_series[calibration_year])

REMIND_to_EXIOBASE_export_price_conversion_factor=EXIOBASE_energy_export_price / energy_aggregated_export_price_time_series[calibration_year]


out_df= vol_prices_timeseries_to_df(import_volume_ts=aggregated_energy_import_volumes_time_series,
    export_volume_ts=aggregated_energy_export_volumes_time_series,
    import_price_ts=energy_aggregated_import_price_time_series,
    export_price_ts=energy_aggregated_export_price_time_series,
    volume_unit="EJ",
    price_unit="M 2020 EUR/EJ",
    import_price_conversion_factor=REMIND_to_EXIOBASE_import_price_conversion_factor,
    export_price_conversion_factor=REMIND_to_EXIOBASE_export_price_conversion_factor)

out_df = pd.concat([out_df, rho_rows], ignore_index=True)

out_df.to_csv(os.path.join(out_path, "energy_trade_projection.csv"), index=False)