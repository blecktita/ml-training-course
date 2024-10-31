"""
This module orchestrates the pipeline
"""
import yaml
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from src.preprocessing.data_loader import DataLoader
from src.preprocessing.data_splitter import DataSplitter
from src.preprocessing.missing_value import MissingValueHandler
from src.training.train_model import ModelTrainer
from src.utils.target_validation import validate_target_variable, TargetType
from src.utils.combine_rare_categories import combine_rare_categories
from src.utils.config_accessor import DatasetConfigAccessor
from src.utils.model_config_accessor import ModelConfigAccessor

# Set up logging configuration (keeping your existing setup)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/test2.log'),
        logging.StreamHandler()
    ]
)

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

def process_dataset(data_key: str, dataset_accessor: DatasetConfigAccessor,
                   model_accessor: ModelConfigAccessor) -> None:
    """Process a single dataset using its configuration"""
    
    logging.info(f"Processing dataset: {data_key}")
    
    # Validate configuration
    if not dataset_accessor.validate_config(data_key):
        logging.error(f"Invalid configuration for {data_key}")
        return
    
    # Get all configurations using the accessor
    data_path, delimiter = dataset_accessor.get_data_path(data_key)
    target_validation = dataset_accessor.get_target_validation(data_key)
    split_config = dataset_accessor.get_split_config(data_key)
    
    # Load and preprocess data
    logging.info("Loading and preprocessing data...")
    data_loader = DataLoader(data_path, delimiter)
    mv_handler = MissingValueHandler()

    data_frame = data_loader.load_and_clean_data()
    logging.info(f"Available columns: {list(data_frame.columns)}")
    logging.info(f"Data loaded successfully. Shape: {data_frame.shape}")
    logging.info(f"Simple View of data:\n{data_frame.head(5)}")

    # Handle missing values
    missing_info_original = mv_handler.get_missing_info(data_frame)
    logging.info(f"Missing values in original dataset:\n{missing_info_original}")

    mv_handler.fit_numeric_imputer(data_frame, strategy='mean')
    mv_handler.fit_categorical_imputer(data_frame, strategy='most_frequent')
    df_processed = mv_handler.fit_transform(data_frame)

    missing_info_processed = mv_handler.get_missing_info(df_processed)
    if not missing_info_processed.empty:
        logging.warning(f"Missing values after processing:\n{missing_info_processed}")
    else:
        logging.info("No missing values after processing.")

    # Handle stratification if configured
    if dataset_accessor.has_stratification(data_key):
        stratify_config = split_config.stratify_column
        threshold = split_config.stratify_threshold
        df_processed = combine_rare_categories(
            df_processed,
            stratify_config,
            threshold=threshold
        )
    else:
        logging.info("No stratification configured for this dataset")

    # Split data
    splitter = DataSplitter(split_config)
    df_train, df_val, df_test, y_train, y_val, y_test = splitter.split(df_processed)

    # Validate target variable
    target_type, stats = validate_target_variable(
        y_train,
        target_type=target_validation.type,
        n_classes=target_validation.n_classes,
        value_range=target_validation.value_range
    )

    # Log validation results
    logging.info(f"Target validation complete. Type: {target_type.value}")
    logging.info(f"Target statistics: {stats}")
    logging.info(f"""
    Dataset splits:
    Train: {df_train.shape} (X), {y_train.shape} (y)
    Validation: {df_val.shape} (X), {y_val.shape} (y)
    Test: {df_test.shape} (X), {y_test.shape} (y)
    """)

    # Train and evaluate models
    trainer = ModelTrainer(
        model_accessor=model_accessor,
        dataset_accessor=dataset_accessor,
        data_key=data_key
    )

    # Preprocess features
    X_train_processed, X_val_processed, X_test_processed = trainer.preprocess_data(
        df_train, df_val, df_test)

    logging.info("Training models...")
    trainer.train_models(X_train_processed, y_train)

    # Evaluate models
    results = trainer.evaluate_models(X_val_processed, y_val)
    logging.info("\nModel Evaluation Results:")
    logging.info("\n" + str(results))

    # Get feature importance
    importance = trainer.get_feature_importance()
    logging.info("\nFeature Importance:")
    logging.info("\n" + str(importance.head()))

    # Make predictions
    predictions = trainer.predict(X_test_processed)
    logging.info(f"\nPredictions shape: {predictions.shape}")
    logging.info(f"Unique predicted values: {np.unique(predictions)}")

    # Save results
    save_results(results, importance, predictions)

    logging.info("Pipeline completed successfully for dataset: " + data_key)

    return results, importance, predictions

def main():
    """
    Main function to execute the pipeline.
    """
    try:
        # Load all configurations
        dataset_config = load_config('src/utils/configs/dataset_config.yaml')
        model_config = load_config('src/utils/configs/model_config.yaml')
        hyperparams_config = load_config('src/utils/configs/hyperparameters_config.yaml')

        # Initialize accessors
        dataset_accessor = DatasetConfigAccessor(dataset_config)
        model_accessor = ModelConfigAccessor(model_config, hyperparams_config)

        # Process specific dataset
        data_key = 'bank_data'  # or get from command line args
        results, importance, predictions = process_dataset(
            data_key, dataset_accessor, model_accessor
        )

        # Or process all datasets
        # for data_key in config_accessor.available_datasets:
        #     try:
        #         process_dataset(data_key, config_accessor, model_config, hyperparams_config)
        #     except Exception as e:
        #         logging.error(f"Error processing {data_key}: {str(e)}")

        return results, importance, predictions

    except Exception as e:
        logging.error(f"Error in pipeline: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
