import json
from pathlib import Path


def load_config(module_dir: Path, config_name: str = "config.json") -> dict:
    """Load common config and module config, then merge them.

    Module keys override common keys when names overlap.
    """
    src_root = module_dir.parent
    common_path = src_root / "common" / "config.shared.json"
    module_path = module_dir / config_name

    with open(common_path, "r", encoding="utf-8") as f:
        common_config = json.load(f)

    with open(module_path, "r", encoding="utf-8") as f:
        module_config = json.load(f)

    return {**common_config, **module_config}
