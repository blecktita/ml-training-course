from enum import Enum
from typing import Optional, Union
import pandas as pd
import numpy as np
import logging

class TargetType(Enum):
    BINARY = "binary"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"

def validate_target_variable(
    target_variable: pd.Series,
    target_type: Union[str, TargetType],
    n_classes: Optional[int] = None,
    value_range: Optional[tuple[float, float]] = None
) -> tuple[TargetType, dict]:
    """
    Validate target variable based on specified type and constraints.
    
    Args:
        target_variable: Series containing target values
        target_type: 'binary', 'multiclass', or 'regression'
        n_classes: Expected number of classes for multiclass
        value_range: Optional (min, max) range for regression values
    
    Returns:
        tuple: (TargetType, dict of statistics)
    """
    if isinstance(target_type, str):
        target_type = TargetType(target_type.lower())

    unique_values = np.unique(target_variable)
    
    # Initialize basic stats that apply to all types
    def _format_unique_values(unique_values: np.ndarray, max_display: int = 10) -> list:
        """
        Format unique values for display, limiting the number of values shown
        and rounding floats for cleaner display.
        
        Args:
            unique_values: Array of unique values
            max_display: Maximum number of values to display
            
        Returns:
            List of formatted values
        """
        # Convert to list and limit length
        if len(unique_values) > max_display:
            selected_values = unique_values[:max_display]
            
            # Round float values if present
            if np.issubdtype(unique_values.dtype, np.floating):
                formatted_values = [f"{v:.2f}" for v in selected_values]
            else:
                formatted_values = selected_values.tolist()
                
            return formatted_values + [f"... and {len(unique_values) - max_display} more"]
        
        # If we're showing all values, still round floats
        if np.issubdtype(unique_values.dtype, np.floating):
            return [f"{v:.2f}" for v in unique_values]
        
        return unique_values.tolist()

    stats = {
        "unique_count": len(unique_values),
        "unique_values": _format_unique_values(unique_values)
    }

    if target_type in [TargetType.BINARY, TargetType.MULTICLASS]:
        # For categorical targets (binary and multiclass)
        if isinstance(target_variable, pd.Series):
            value_counts = target_variable.value_counts()
            stats.update({
                "class_distribution": value_counts.to_dict(),
                "most_frequent": value_counts.index[0],
                "least_frequent": value_counts.index[-1]
            })
        elif isinstance(target_variable, np.ndarray):
            target_series = pd.Series(target_variable)
            value_counts = target_series.value_counts()
            stats.update({
                "class_distribution": value_counts.to_dict(),
                "most_frequent": value_counts.index[0],
                "least_frequent": value_counts.index[-1]
            })
        else:
            raise ValueError("target_variable must be a pandas Series or a NumPy array")
        
        if target_type == TargetType.BINARY:
            if len(unique_values) != 2:
                raise ValueError(
                    f"Binary target must have exactly 2 values. "
                    f"Found: {unique_values}"
                )
            
        elif target_type == TargetType.MULTICLASS:
            if n_classes and len(unique_values) != n_classes:
                raise ValueError(
                    f"Expected {n_classes} classes but found {len(unique_values)}"
                )
    
    elif target_type == TargetType.REGRESSION:
        # For numerical targets (regression)
        try:
            numeric_series = pd.to_numeric(target_variable)
            stats.update({
                "min": float(numeric_series.min()),
                "max": float(numeric_series.max()),
                "mean": float(numeric_series.mean()),
                "std": float(numeric_series.std())
            })
            
            if value_range:
                min_val, max_val = value_range
                if numeric_series.min() < min_val or numeric_series.max() > max_val:
                    raise ValueError(
                        f"Values must be between {min_val} and {max_val}. "
                        f"Found range: {numeric_series.min()} to {numeric_series.max()}"
                    )
        except ValueError as e:
            raise ValueError(
                f"Regression target must contain numeric values. "
                f"Error: {str(e)}"
            )

    logging.info(f"Target validation passed. Type: {target_type.value}, Stats: {stats}")
    return target_type, stats