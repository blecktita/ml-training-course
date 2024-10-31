"""
This module provides a class for handling missing values in datasets.
"""
from typing import List, Union, Dict
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

class MissingValueHandler:
    """Class for handling missing values in datasets using various strategies."""

    def __init__(self):
        """Initializes the MissingValueHandler."""
        self.numerical_imputers: Dict[str, SimpleImputer] = {}
        self.categorical_imputers: Dict[str, SimpleImputer] = {}
        self.fill_values: Dict[str, Union[str, float]] = {}

    @staticmethod
    def get_missing_info(data_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Get information about missing values in the DataFrame.
        
        Args:
            data_frame (pd.DataFrame): Input DataFrame to analyze.
            
        Returns:
            pd.DataFrame: DataFrame containing missing value statistics.
        """
        missing_info = pd.DataFrame({
            'missing_count': data_frame.isnull().sum(),
            'missing_percentage': (data_frame.isnull().sum() / len(data_frame) * 100).round(2)
        })
        missing_info = missing_info[missing_info['missing_count'] > 0].sort_values(
            'missing_percentage', ascending=False
        )
        return missing_info

    @staticmethod
    def get_numeric_columns(data_frame: pd.DataFrame) -> List[str]:
        """
        Get list of numeric columns in the DataFrame.
        
        Args:
            data_frame (pd.DataFrame): Input DataFrame to analyze.
            
        Returns:
            List[str]: List of numeric column names.
        """
        return list(data_frame.select_dtypes(include=['int64', 'float64']).columns)

    @staticmethod
    def get_categorical_columns(data_frame: pd.DataFrame) -> List[str]:
        """
        Get list of categorical columns in the DataFrame.
        
        Args:
            data_frame (pd.DataFrame): Input DataFrame to analyze.
            
        Returns:
            List[str]: List of categorical column names.
        """
        return list(data_frame.select_dtypes(include=['object', 'category']).columns)

    def fit_numeric_imputer(self, data_frame: pd.DataFrame, strategy: str = 'mean',
                          columns: List[str] = None) -> None:
        """
        Fit imputer for numeric columns.
        
        Args:
            data_frame (pd.DataFrame): Input DataFrame.
            strategy (str): Imputation strategy ('mean', 'median', 'most_frequent', 'constant').
            columns (List[str], optional): Specific numeric columns to impute. 
            If None, all numeric columns are used.
        """
        if columns is None:
            columns = self.get_numeric_columns(data_frame)

        for col in columns:
            if col in data_frame.columns and data_frame[col].isnull().any():
                imputer = SimpleImputer(strategy=strategy)
                imputer.fit(data_frame[[col]])
                self.numerical_imputers[col] = imputer

    def fit_categorical_imputer(self, data_frame: pd.DataFrame, strategy: str = 'most_frequent',
                              columns: List[str] = None) -> None:
        """
        Fit imputer for categorical columns.
        
        Args:
            data_frame (pd.DataFrame): Input DataFrame.
            strategy (str): Imputation strategy ('most_frequent', 'constant').
            columns (List[str], optional): Specific categorical columns to impute. 
            If None, all categorical columns are used.
        """
        if columns is None:
            columns = self.get_categorical_columns(data_frame)

        for col in columns:
            if col in data_frame.columns and data_frame[col].isnull().any():
                imputer = SimpleImputer(strategy=strategy)
                imputer.fit(data_frame[[col]])
                self.categorical_imputers[col] = imputer

    def set_custom_fill_values(self, fill_dict: Dict[str, Union[str, float]]) -> None:
        """
        Set custom fill values for specific columns.
        
        Args:
            fill_dict (Dict[str, Union[str, float]]): Dictionary mapping column names 
            to fill values.
        """
        self.fill_values.update(fill_dict)

    def transform(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the fitted imputation to the DataFrame.
        
        Args:
            data_frame (pd.DataFrame): Input DataFrame to transform.
            
        Returns:
            pd.DataFrame: Transformed DataFrame with imputed values.
        """
        df_copy = data_frame.copy()

        # Apply numerical imputers
        for col, imputer in self.numerical_imputers.items():
            if col in df_copy.columns:
                # Flatten the array to 1D before assignment
                df_copy[col] = imputer.transform(df_copy[[col]]).ravel()

        # Apply categorical imputers
        for col, imputer in self.categorical_imputers.items():
            if col in df_copy.columns:
                # Flatten the array to 1D before assignment
                df_copy[col] = imputer.transform(df_copy[[col]]).ravel()

        # Apply custom fill values
        for col, value in self.fill_values.items():
            if col in df_copy.columns:
                df_copy[col].fillna(value, inplace=True)

        return df_copy

    def fit_transform(self, data_frame: pd.DataFrame, numeric_strategy: str = 'mean',
                     categorical_strategy: str = 'most_frequent') -> pd.DataFrame:
        """
        Fit imputers and transform the DataFrame in one step.
        
        Args:
            data_frame (pd.DataFrame): Input DataFrame.
            numeric_strategy (str): Imputation strategy for numeric columns.
            categorical_strategy (str): Imputation strategy for categorical columns.
            
        Returns:
            pd.DataFrame: Transformed DataFrame with imputed values.
        """
        # Fit imputers
        self.fit_numeric_imputer(data_frame, strategy=numeric_strategy)
        self.fit_categorical_imputer(data_frame, strategy=categorical_strategy)

        # Transform data
        return self.transform(data_frame)
