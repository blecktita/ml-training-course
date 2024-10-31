"""
This module provides a function to combine rare categories in specified categorical features 
into an 'Other' category.
"""

import pandas as pd
import logging
from typing import List, Union

def combine_rare_categories(
    df: pd.DataFrame,
    features: Union[str, List[str]],
    threshold: float = 0.05
) -> pd.DataFrame:
    """
    Combine rare categories in specified categorical features into an 'Other' category.

    Args:
        df (pd.DataFrame): Input DataFrame.
        features (str or List[str]): Single feature name or list of feature names.
        threshold (float): Frequency threshold below which categories are considered rare.

    Returns:
        pd.DataFrame: DataFrame with rare categories combined into 'Other'.
    """
    if isinstance(features, str):
        features = [features]  # Convert single feature to list for uniform processing

    for feature in features:
        if feature not in df.columns:
            logging.warning(f"Feature '{feature}' not found in DataFrame. Skipping.")
            continue

        # Calculate frequency of each category
        freq = df[feature].value_counts(normalize=True)
        rare_categories = freq[freq < threshold].index
        num_rare = len(rare_categories)

        if num_rare > 0:
            # Replace rare categories with 'Other'
            df[feature] = df[feature].apply(lambda x: 'Other' if x in rare_categories else x)
            logging.info(f"Combined {num_rare} rare categories in '{feature}' into 'Other'.")
        else:
            logging.info(f"No rare categories found in '{feature}' with threshold {threshold}.")

    return df
