import subprocess
import sys
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SRC_ROOT.parents[1]

PIPELINE = [
    {"name": "consumption_elasticities", "script": "Data_preprocessing/src/consumption_elasticities/run.py"},
    {"name": "reformat_exiobase", "script": "Data_preprocessing/src/reformat_EXIOBASE/run_reformat_EXIOBASE_1_energy.py"},
    {"name": "ssp2", "script": "Data_preprocessing/src/SSP2/run.py"},
    {"name": "productivities", "script": "Data_preprocessing/src/Productivities/run.py"},
    {"name": "hybridization", "script": "Data_preprocessing/src/Hybridization/run.py"},
    {"name": "energy_trade", "script": "Data_preprocessing/src/Energy_trade_scenarisation/run.py"},
    {"name": "technical_coefficients", "script": "Data_preprocessing/src/Technical_coefficients/run.py"},
    {"name": "macro_indicators", "script": "Data_preprocessing/src/macro_indicators/run.py"},
    {"name": "to_solver_format", "script": "Data_preprocessing/src/to_Solver_format/run.py"},
]
if __name__ == "__main__":
    for step in PIPELINE:
        script = step["script"]
        print(f"Running {step['name']}: {script}")
        subprocess.run([sys.executable, script], check=True, cwd=str(REPO_ROOT))
