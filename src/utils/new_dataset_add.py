"""
This module provides functionality for generating and updating dataset configurations.
It includes classes and methods for loading, cleaning, and analyzing datasets, as well as
interactively selecting target columns and inferring column types.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import questionary
from rich.console import Console
from rich.table import Table
from rich import print as rprint
import os
from src.preprocessing.data_loader import DataLoader

class EnhancedDatasetConfigGenerator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.console = Console()

    def infer_delimiter(self, file_path: str) -> str:
        """Attempt to infer the delimiter of a file by reading the first line"""
        with open(file_path, 'r') as file:
            first_line = file.readline()
            if ',' in first_line:
                return ','
            elif ';' in first_line:
                return ';'
            elif '\t' in first_line:
                return '\t'
        return ','  # default to comma if unable to determine

    def load_and_clean_data(self, file_path: str, delimiter: str) -> pd.DataFrame:
        """Load and clean the dataset using DataLoader"""
        try:
            loader = DataLoader(file_path, delimiter)
            df = loader.load_and_clean_data(clean_strings=True)
            self.console.print("[green]✓ Data loaded and cleaned successfully[/green]")
            return df
        except Exception as e:
            self.console.print(f"[red]Error loading data: {str(e)}[/red]")
            raise

    def display_dataframe_info(self, df: pd.DataFrame):
        """Display detailed information about the DataFrame"""
        # Create a table for general information
        info_table = Table(title="Dataset Overview")
        info_table.add_column("Metric", style="cyan")
        info_table.add_column("Value", style="green")

        info_table.add_row("Number of Rows", str(len(df)))
        info_table.add_row("Number of Columns", str(len(df.columns)))
        info_table.add_row("Memory Usage", f"{df.memory_usage().sum() / 1024**2:.2f} MB")

        self.console.print(info_table)

        # Create a table for column information
        column_table = Table(title="Column Information")
        column_table.add_column("Column Name", style="cyan")
        column_table.add_column("Type", style="green")
        column_table.add_column("Non-Null Count", style="blue")
        column_table.add_column("Unique Values", style="magenta")
        column_table.add_column("Sample Values", style="yellow")

        for col in df.columns:
            sample_values = str(df[col].head(3).tolist())[:50] + "..."
            column_table.add_row(
                col,
                str(df[col].dtype),
                f"{df[col].count()}/{len(df)}",
                str(df[col].nunique()),
                sample_values
            )

        self.console.print(column_table)

    def infer_column_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Infer categorical and numerical columns from DataFrame"""
        categorical_features = []
        numerical_features = []

        # Create a table for column classification
        table = Table(title="Column Classification")
        table.add_column("Column", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Reason", style="yellow")

        for column in df.columns:
            if column.lower() in ['target', 'label', 'y', 'class']:
                continue

            reason = ""
            if df[column].dtype in ['object', 'category']:
                categorical_features.append(column)
                reason = "String/categorical dtype"
            elif df[column].dtype in ['int64', 'float64']:
                if df[column].nunique() / len(df) < 0.05:
                    categorical_features.append(column)
                    reason = f"Numeric but low cardinality ({df[column].nunique()} unique values)"
                else:
                    numerical_features.append(column)
                    reason = "Numeric with high cardinality"

            table.add_row(column,
                         "Categorical" if column in categorical_features else "Numerical",
                         reason)

        self.console.print(table)
        return categorical_features, numerical_features

    def suggest_target_column(self, df: pd.DataFrame) -> str:
        """Interactive target column selection with rich display"""
        likely_target_names = ['target', 'label', 'y', 'class', 'price', 'amount', 'value']

        # Create a table of columns with their properties
        table = Table(title="Available Columns")
        table.add_column("Column Name", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Unique Values", style="magenta")
        table.add_column("Sample Values", style="yellow")

        choices = []
        for col in df.columns:
            sample_values = str(df[col].head(3).tolist())[:50] + "..."
            table.add_row(
                col,
                str(df[col].dtype),
                str(df[col].nunique()),
                sample_values
            )
            choices.append(col)

        self.console.print(table)

        # Use questionary for interactive selection
        suggested = [col for col in df.columns if col.lower() in likely_target_names]
        if suggested:
            self.console.print(
                f"\n[yellow]Suggested target column(s): {', '.join(suggested)}[/yellow]"
            )

        target = questionary.select(
            "Which column should be the target?",
            choices=choices,
            default=suggested[0] if suggested else choices[0]
        ).ask()

        return target

    def infer_target_type(self, series: pd.Series) -> Dict:
        """Infer the type of the target variable and its properties"""
        unique_values = series.nunique()

        # Create a table for target variable analysis
        table = Table(title="Target Variable Analysis")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Unique Values", str(unique_values))
        table.add_row("Data Type", str(series.dtype))

        if series.dtype in ['int64', 'float64']:
            table.add_row("Min Value", str(series.min()))
            table.add_row("Max Value", str(series.max()))
            table.add_row("Mean Value", str(series.mean()))

        self.console.print(table)

        # Determine target type
        if unique_values == 2:
            target_type = {
                "type": "binary",
                "n_classes": 2,
                "value_range": None
            }
        elif unique_values < len(series) * 0.1 and unique_values < 100:
            target_type = {
                "type": "multiclass",
                "n_classes": int(unique_values),
                "value_range": None
            }
        else:
            target_type = {
                "type": "regression",
                "n_classes": None,
                "value_range": [float(series.min()), float(series.max())]
            }

        self.console.print(f"\n[green]Inferred target type: {target_type['type']}[/green]")
        return target_type

    def generate_config(self,
                       file_path: str,
                       dataset_key: Optional[str] = None,
                       target_column: Optional[str] = None,
                       delimiter: Optional[str] = None) -> Dict:
        """Generate configuration for a dataset"""
        self.console.print("\n[bold blue]Starting Dataset Analysis[/bold blue]")

        # Infer dataset key from filename if not provided
        if dataset_key is None:
            dataset_key = Path(file_path).stem.lower().replace(" ", "_")
            self.console.print(f"Using dataset key: [cyan]{dataset_key}[/cyan]")

        # Infer delimiter if not provided
        if delimiter is None:
            delimiter = self.infer_delimiter(file_path)
            self.console.print(f"Inferred delimiter: [cyan]{delimiter}[/cyan]")

        # Load and clean the data
        df = self.load_and_clean_data(file_path, delimiter)

        # Display dataset information
        self.console.print("\n[bold blue]Dataset Information[/bold blue]")
        self.display_dataframe_info(df)

        # Identify target column if not provided
        if target_column is None:
            self.console.print("\n[bold blue]Target Column Selection[/bold blue]")
            target_column = self.suggest_target_column(df)

        # Infer column types
        self.console.print("\n[bold blue]Column Type Analysis[/bold blue]")
        categorical_features, numerical_features = self.infer_column_types(df)

        # Infer target properties
        self.console.print("\n[bold blue]Target Variable Analysis[/bold blue]")
        target_validation = self.infer_target_type(df[target_column])

        # Create configuration dictionary
        config = {
            dataset_key: {
                f"{dataset_key}_path": str(file_path),
                "delimiter": delimiter,
                "target_validation": target_validation,
                "split_config": {
                    "target_column": target_column,
                    "test_size": 0.2,
                    "val_size": 0.2,
                    "random_state": 42
                },
                "categorical_features": categorical_features,
                "numerical_features": numerical_features,
                "target_column": target_column
            }
        }

        return config

    def update_yaml_config(self,
                          new_config: Dict,
                          yaml_path: str = 'src/utils/configs/dataset_config.yaml') -> None:
        """Update the existing YAML configuration file with new dataset config"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(yaml_path), exist_ok=True)

            # Read existing config if file exists, otherwise create new
            if os.path.exists(yaml_path):
                with open(yaml_path, 'r') as file:
                    existing_config = yaml.safe_load(file) or {'data': {}}
            else:
                existing_config = {'data': {}}

            # Update the config
            existing_config['data'].update(new_config)

            # Write back to file
            with open(yaml_path, 'w') as file:
                yaml.dump(existing_config, file, default_flow_style=False, sort_keys=False)

            self.console.print(
                f"\n[green]Configuration updated successfully in {yaml_path}[/green]"
            )

        except Exception as e:
            self.console.print(f"[red]Error updating configuration: {str(e)}[/red]")
            raise

def get_available_datasets() -> List[str]:
    """Get list of CSV files in the data directory"""
    data_files = []
    search_paths = ['.', 'data', 'data/raw', 'data/processed']

    for path in search_paths:
        if Path(path).exists():
            data_files.extend([str(p) for p in Path(path).glob('*.csv')])

    return sorted(data_files)

def validate_file_path(path: str) -> bool:
    """Validate if the file exists and is a CSV"""
    return Path(path).exists() and path.lower().endswith('.csv')

def main():
    """Enhanced CLI interface for the config generator"""
    console = Console()
    generator = EnhancedDatasetConfigGenerator()

    console.print("\n[bold blue]Dataset Configuration Generator[/bold blue]")
    console.print("[blue]------------------------------[/blue]\n")

    # Get available datasets
    available_datasets = get_available_datasets()

    # File selection
    if available_datasets:
        console.print("[green]Found the following CSV files:[/green]")
        for idx, file in enumerate(available_datasets, 1):
            console.print(f"  {idx}. {file}")

        file_choice = questionary.select(
            "Select your dataset:",
            choices=available_datasets + ['Enter custom path...']
        ).ask()

        if file_choice == 'Enter custom path...':
            while True:
                file_path = questionary.text(
                    "Enter the path to your CSV file:"
                ).ask()
                if validate_file_path(file_path):
                    break
                console.print("[red]File not found or not a CSV. Please try again.[/red]")
        else:
            file_path = file_choice
    else:
        console.print("[yellow]No CSV files found in standard directories.[/yellow]")
        while True:
            file_path = questionary.text(
                "Enter the path to your CSV file:"
            ).ask()
            if validate_file_path(file_path):
                break
            console.print("[red]File not found or not a CSV. Please try again.[/red]")

    # Optional inputs with questionary
    dataset_key = questionary.text(
        "Enter dataset key (press Enter to infer from filename):",
        default=""
    ).ask()

    delimiter = questionary.select(
        "Select delimiter:",
        choices=[
            {'name': 'Comma (,)', 'value': ','},
            {'name': 'Semicolon (;)', 'value': ';'},
            {'name': 'Tab (\\t)', 'value': '\t'},
            {'name': 'Auto-detect', 'value': None}
        ]
    ).ask()

    # Generate configuration
    config = generator.generate_config(
        file_path=file_path,
        dataset_key=dataset_key or None,
        delimiter=delimiter
    )

    # Show the generated config
    console.print("\n[bold blue]Generated Configuration:[/bold blue]")
    console.print(yaml.dump(config, default_flow_style=False))

    # Ask to save
    if questionary.confirm("Would you like to update the dataset_config.yaml file?").ask():
        generator.update_yaml_config(config)

if __name__ == "__main__":
    main()
