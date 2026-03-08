"""
Configuration loading utilities.
"""

import os
from pathlib import Path
import yaml


def load_config(config_name: str = "default.yaml") -> dict:
    """
    Load configuration from config/ directory.

    Args:
        config_name: Name of the config file (e.g., "default.yaml")

    Returns:
        Configuration dictionary
    """
    config_dir = Path(__file__).parent
    config_path = config_dir / config_name

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config
