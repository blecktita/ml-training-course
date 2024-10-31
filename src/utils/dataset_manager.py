"""
This module contains the DatasetDetector class which is used to detect new datasets.
"""
import yaml
import os
import logging
from pathlib import Path
from typing import Set

class DatasetDetector:
    """Simple class to detect new datasets"""
    def __init__(self, config_path: str = 'src/utils/configs/dataset_config.yaml'):
        self.config_path = config_path
        self.data_dirs = ['data', 'data/raw', 'data/processed']

    def get_configured_datasets(self) -> Set[str]:
        """Get set of datasets that are already configured"""
        try:
            if not os.path.exists(self.config_path):
                return set()

            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
                if not config or 'data' not in config:
                    return set()

                # Get all configured dataset paths
                configured_paths = set()
                for dataset_config in config['data'].values():
                    # Extract path from config (handles both dataset_path and dataset_key_path)
                    path_keys = [k for k in dataset_config.keys() if k.endswith('_path')]
                    for key in path_keys:
                        configured_paths.add(os.path.normpath(dataset_config[key]))

                return configured_paths
        except Exception as e:
            logging.error(f"Error reading configuration: {e}")
            return set()

    def get_available_datasets(self) -> Set[str]:
        """Get set of all available dataset files"""
        available_paths = set()

        for dir_path in self.data_dirs:
            if os.path.exists(dir_path):
                for file in Path(dir_path).rglob('*.csv'):
                    available_paths.add(os.path.normpath(str(file)))

        return available_paths

    def detect_new_datasets(self) -> Set[str]:
        """Detect any new datasets that aren't configured yet"""
        configured = self.get_configured_datasets()
        available = self.get_available_datasets()
        return available - configured
