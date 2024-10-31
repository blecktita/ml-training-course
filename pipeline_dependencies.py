# Standard library imports
import yaml
import logging
from pathlib import Path
from enum import Enum, auto
from typing import Optional, Tuple, Dict, Any

# Third-party imports
import pandas as pd
import numpy as np

# Import your existing modules
from src.preprocessing.data_loader import DataLoader
from src.preprocessing.data_splitter import DataSplitter
from src.preprocessing.missing_value import MissingValueHandler
from src.training.train_model import ModelTrainer
from src.utils.target_validation import validate_target_variable, TargetType
from src.utils.combine_rare_categories import combine_rare_categories
from src.utils.config_accessor import DatasetConfigAccessor
from src.utils.model_config_accessor import ModelConfigAccessor

def load_config(config_path: str) -> dict:
    """Load configuration settings from a YAML file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file) or {}
        return config
    except FileNotFoundError:
        logging.error("Configuration file not found at %s", config_path)
        raise
    except yaml.YAMLError as yaml_error:
        logging.error("Error parsing configuration file %s: %s", config_path, yaml_error)
        raise

def save_results(results: pd.DataFrame, importance: pd.DataFrame, predictions: np.ndarray,
                output_dir: str = 'outputs') -> None:
    """Save model results, feature importance, and predictions to files."""
    Path(output_dir).mkdir(exist_ok=True)

    results.to_csv(f'{output_dir}/model_results.csv', index=False)
    importance.to_csv(f'{output_dir}/feature_importance.csv', index=False)
    pd.DataFrame({'predictions': predictions}).to_csv(
        f'{output_dir}/predictions.csv', index=False)

    logging.info(f"Results saved to {output_dir}/")