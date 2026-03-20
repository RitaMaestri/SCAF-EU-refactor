import pandas as pd
import sys
from pathlib import Path

from lib import build_growth_factors

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from common.path_loader import load_config

module_dir = Path(__file__).resolve().parent
repo_root = module_dir.parents[2]
config = load_config(module_dir)

template_path = repo_root / config["growth_factors_template"]
mapping_path = repo_root / config["mapping_file"]
calibration_root = repo_root / config["calibration_output_root"]
out_path = repo_root / config["out_file"]
end_year = int(config["end_year"]) if config.get("end_year") else None

template_df = pd.read_csv(template_path)
mapping_df = pd.read_csv(mapping_path)

growth_factors_df = build_growth_factors(template_df, mapping_df, calibration_root, end_year)

out_path.parent.mkdir(parents=True, exist_ok=True)
growth_factors_df.to_csv(out_path, index=False)
print(f"Written {len(growth_factors_df)} rows to {out_path}")
