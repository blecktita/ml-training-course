"""
This module provides classes for splitting data into train/validation/test sets
with configuration management.
"""

from dataclasses import dataclass
from typing import Tuple, Union, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

@dataclass
class SplitterConfig:
    """Configuration class for data splitting parameters."""
    val_size: float = 0.2
    test_size: float = 0.2
    random_state: int = 42
    target_column: str = 'target'
    log_transform: bool = False
    shuffle: bool = True
    stratify_column: Optional[str] = None
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'SplitterConfig':
        """
        Create config from dictionary, handling nested stratification config.
        
        Args:
            config_dict (Dict): Configuration dictionary.
            
        Returns:
            SplitterConfig: Configuration object.
        """
        # Handle nested stratify structure from YAML if present
        if 'stratify' in config_dict and isinstance(config_dict['stratify'], dict):
            stratify_config = config_dict.pop('stratify')
            config_dict['stratify_column'] = stratify_config.get('stratify_column')
            
        return cls(**config_dict)

class DataSplitter:
    """Class for splitting data into train/validation/test sets."""

    def __init__(self, config: Union[Dict, SplitterConfig]):
        """
        Initialize DataSplitter with configuration.

        Args:
            config: Either a dictionary of parameters or SplitterConfig object
        """
        if isinstance(config, dict):
            self.config = SplitterConfig.from_dict(config)
        else:
            self.config = config

        self.train_idx = None
        self.val_idx = None
        self.test_idx = None

    def _generate_indices(self, n: int, stratify_values: Optional[np.ndarray] = None) -> None:
        """
        Generate split indices using stratification if specified.

        Args:
            n (int): Total number of samples.
            stratify_values (np.ndarray, optional): Array to stratify on.
        """
        # First split off the test set
        test_size = self.config.test_size
        train_val_size = 1 - test_size

        if self.config.stratify_column and stratify_values is not None:
            stratify_test = stratify_values
        else:
            stratify_test = None

        X_temp, self.test_idx = train_test_split(
            np.arange(n),
            test_size=test_size,
            random_state=self.config.random_state,
            shuffle=self.config.shuffle,
            stratify=stratify_test
        )

        # Now split the remaining into train and validation sets
        if self.config.stratify_column and stratify_values is not None:
            stratify_train_val = stratify_values[X_temp]
        else:
            stratify_train_val = None

        val_size = self.config.val_size / train_val_size

        self.train_idx, self.val_idx = train_test_split(
            X_temp,
            test_size=val_size,
            random_state=self.config.random_state,
            shuffle=self.config.shuffle,
            stratify=stratify_train_val
        )

    def _transform_target(self, y: np.ndarray) -> np.ndarray:
        """
        Apply transformation to target variable if configured.

        Args:
            y (np.ndarray): Target values.

        Returns:
            np.ndarray: Transformed target values.
        """
        if self.config.log_transform:
            return np.log1p(y)
        return y

    def split(
        self,
        data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
             np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train/validation/test sets.

        Args:
            data (pd.DataFrame): Input DataFrame.

        Returns:
            Tuple containing:
            - train DataFrame
            - validation DataFrame
            - test DataFrame
            - train target values
            - validation target values
            - test target values
        """
        # Extract stratify values if stratification is enabled
        if self.config.stratify_column and self.config.stratify_column in data.columns:
            stratify_values = data[self.config.stratify_column].values
        else:
            stratify_values = None

        # Generate split indices
        self._generate_indices(len(data), stratify_values)

        # Split data
        df_train = data.iloc[self.train_idx].reset_index(drop=True)
        df_val = data.iloc[self.val_idx].reset_index(drop=True)
        df_test = data.iloc[self.test_idx].reset_index(drop=True)

        # Extract and transform target
        y_train = self._transform_target(df_train[self.config.target_column].values)
        y_val = self._transform_target(df_val[self.config.target_column].values)
        y_test = self._transform_target(df_test[self.config.target_column].values)

        # Remove target column from features
        df_train = df_train.drop(columns=[self.config.target_column])
        df_val = df_val.drop(columns=[self.config.target_column])
        df_test = df_test.drop(columns=[self.config.target_column])

        return df_train, df_val, df_test, y_train, y_val, y_test

    def get_split_sizes(self) -> Dict[str, int]:
        """
        Get the sizes of each split.

        Returns:
            Dict[str, int]: Dictionary containing split sizes.
        """
        if self.train_idx is None:
            raise ValueError("Data hasn't been split yet. Call split() first.")

        return {
            'train_size': len(self.train_idx),
            'validation_size': len(self.val_idx),
            'test_size': len(self.test_idx)
        }