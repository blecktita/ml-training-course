# pylint: disable=too-few-public-methods
"""
This module provides a class for loading and cleaning datasets.
"""
import os
from typing import List
import pandas as pd

class DataLoader:
    """Class for loading and cleaning datasets."""

    def __init__(self, data_path: str, sep: str = ','):
        """
        Initializes the DataLoader with the path to the dataset.

        Args:
            data_path (str): Path to the CSV file containing the dataset.
            sep (str): Separator used in the CSV file. Defaults to comma.
        """
        self.data_path = data_path
        self.sep = sep

    @staticmethod
    def clean_column_names(data_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the column names of the DataFrame by converting them to lowercase 
        and replacing spaces with underscores.

        Args:
            data_frame (pd.DataFrame): Input DataFrame with columns to clean.

        Returns:
            pd.DataFrame: DataFrame with cleaned column names.
        """
        data_frame.columns = data_frame.columns.str.lower().str.replace(' ', '_')
        return data_frame

    @staticmethod
    def get_string_columns(data_frame: pd.DataFrame) -> List[str]:
        """
        Identifies columns with string (object) data type.

        Args:
            data_frame (pd.DataFrame): Input DataFrame to analyze.

        Returns:
            List[str]: List of column names that contain string data.
        """
        return list(data_frame.dtypes[data_frame.dtypes == 'object'].index)

    @staticmethod
    def clean_string_columns(data_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans all string (object) columns by converting to lowercase and replacing
        spaces with underscores.

        Args:
            data_frame (pd.DataFrame): Input DataFrame with string columns to clean.

        Returns:
            pd.DataFrame: DataFrame with cleaned string columns.
        """
        string_columns = DataLoader.get_string_columns(data_frame)

        for col in string_columns:
            data_frame[col] = data_frame[col].str.lower().str.replace(' ', '_')

        return data_frame

    def load_and_clean_data(self, clean_strings: bool = True) -> pd.DataFrame:
        """
        Loads the dataset into a pandas DataFrame, cleans column names, and optionally
        cleans string columns.

        Args:
            clean_strings (bool): Whether to clean string columns. Defaults to True.

        Returns:
            pd.DataFrame: The loaded and cleaned dataset.

        Raises:
            FileNotFoundError: If the data file is not found at the specified path.
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found at {self.data_path}")

        data_frame = pd.read_csv(self.data_path, sep=self.sep)
        data_frame = self.clean_column_names(data_frame)

        if clean_strings:
            data_frame = self.clean_string_columns(data_frame)
        return data_frame
