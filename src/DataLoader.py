import pandas as pd
import os

class DataLoader:
    """Class for loading the Bank Marketing dataset."""

    def __init__(self, data_path, sep):
        """
        Initializes the DataLoader with the path to the dataset.

        Args:
            filepath (str): Path to the CSV file containing the dataset.
        """
        self.data_path = data_path
        self.sep = sep

    def load_data(self):
            """
            Loads the dataset into a pandas DataFrame.

            Returns:
                pd.DataFrame: The loaded dataset.
            """
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Data file not found at {self.data_path}")
            df = pd.read_csv(self.data_path, sep=self.sep)
            return df