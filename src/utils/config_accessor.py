# pylint: disable=astroid-error
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class TargetValidation:
    type: str
    n_classes: Optional[int]
    value_range: Optional[Tuple[float, float]]

@dataclass
class SplitConfig:
    target_column: str
    test_size: float
    val_size: float
    random_state: int = 42
    shuffle: bool = True
    stratify_column: Optional[str] = None
    stratify_threshold: Optional[float] = None
    log_transform: bool = False

@dataclass
class DatasetFeatures:
    categorical_features: List[str]
    numerical_features: List[str]
    target_column: str

class DatasetConfigAccessor:
    """Accessor class for dataset configurations"""
    
    def __init__(self, config: Dict):
        """
        Initialize with full configuration dictionary.
        
        Args:
            config: Complete configuration dictionary containing all dataset configs
        """
        self.config = config
        
    def get_dataset_config(self, data_key: str) -> Dict:
        """Get configuration for a specific dataset"""
        try:
            return self.config['data'][data_key]
        except KeyError:
            available = list(self.config['data'].keys())
            raise ValueError(f"Dataset '{data_key}' not found. Available: {available}")

    def get_data_path(self, data_key: str) -> Tuple[Path, str]:
        """Get data path and delimiter"""
        dataset = self.get_dataset_config(data_key)
        return (
            Path(dataset[f'{data_key}_path']),
            dataset['delimiter']
        )

    def get_target_validation(self, data_key: str) -> TargetValidation:
        """Get target validation configuration"""
        dataset = self.get_dataset_config(data_key)
        target_val = dataset['target_validation']
        return TargetValidation(
            type=target_val['type'],
            n_classes=target_val['n_classes'],
            value_range=target_val['value_range']
        )

    def get_split_config(self, data_key: str) -> SplitConfig:
        """Get split configuration including optional stratification"""
        dataset = self.get_dataset_config(data_key)
        split_config = dataset['split_config']
        
        # Handle optional stratification
        stratify_config = split_config.get('stratify', {})
        
        return SplitConfig(
            target_column=split_config['target_column'],
            test_size=split_config['test_size'],
            val_size=split_config['val_size'],
            random_state=split_config.get('random_state', 42),
            shuffle=split_config.get('shuffle', True),
            stratify_column=stratify_config.get('stratify_column'),
            stratify_threshold=stratify_config.get('stratify_threshold'),
            log_transform=split_config.get('log_transform', False)
        )

    def get_features(self, data_key: str) -> DatasetFeatures:
        """Get feature configurations"""
        dataset = self.get_dataset_config(data_key)
        return DatasetFeatures(
            categorical_features=dataset['categorical_features'],
            numerical_features=dataset['numerical_features'],
            target_column=dataset['target_column']
        )

    def has_stratification(self, data_key: str) -> bool:
        """Check if dataset uses stratification"""
        dataset = self.get_dataset_config(data_key)
        return 'stratify' in dataset['split_config']

    def validate_config(self, data_key: str) -> bool:
        """Validate that all required fields are present for a dataset"""
        try:
            self.get_data_path(data_key)
            self.get_target_validation(data_key)
            self.get_split_config(data_key)
            self.get_features(data_key)
            return True
        except (KeyError, TypeError) as e:
            logger.error(f"Invalid configuration for dataset {data_key}: {str(e)}")
            return False

    @property
    def available_datasets(self) -> List[str]:
        """Get list of available dataset keys"""
        return list(self.config['data'].keys())