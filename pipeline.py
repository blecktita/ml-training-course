"""
This module defines the MLPipeline class and its associated stages for
loading, preprocessing, training, and evaluating machine learning models.
"""

from pipeline_dependencies import (
    load_config,
    DatasetConfigAccessor,
    ModelConfigAccessor,
    DataLoader,
    MissingValueHandler,
    combine_rare_categories,
    DataSplitter,
    validate_target_variable,
    ModelTrainer,
    save_results
)
from enum import Enum, auto
from typing import Optional
import logging
from src.utils.dataset_manager import DatasetDetector
from src.utils.new_dataset_add import main as config_generator_main

class PipelineStage(Enum):
    DETECT_DATASETS = auto()
    LOAD_CONFIG = auto()
    LOAD_DATA = auto()
    PREPROCESS = auto()
    SPLIT_DATA = auto()
    VALIDATE_TARGET = auto()
    TRAIN = auto()
    EVALUATE = auto()
    SAVE_RESULTS = auto()

class MLPipeline:
    def __init__(self, data_key: str = 'bank_data'):
        self.data_key = data_key
        self.dataset_accessor = None
        self.model_accessor = None
        self.data_frame = None
        self.processed_data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.trainer = None
        self.results = None
        self.importance = None
        self.predictions = None
        self.detector = DatasetDetector()

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/pipeline.log'),
                logging.StreamHandler()
            ]
        )

    def detect_datasets(self) -> bool:
        """Stage 0: Check for new datasets and run configuration if needed"""
        try:
            new_datasets = self.detector.detect_new_datasets()

            if new_datasets:
                logging.info(f"New datasets detected: {new_datasets}")
                logging.info("Launching configuration generator...")
                config_generator_main()

                # Reload configurations after new datasets are added
                self.load_config()
                logging.info("Configurations reloaded after new dataset detection")
                return True

            logging.info("No new datasets detected.")
            return False
        except Exception as e:
            logging.error(f"Error in dataset detection: {str(e)}")
            raise

    def load_config(self) -> None:
        """Stage 1: Load all configuration files"""
        try:
            dataset_config = load_config('src/utils/configs/dataset_config.yaml')
            model_config = load_config('src/utils/configs/model_config.yaml')
            hyperparams_config = load_config('src/utils/configs/hyperparameters_config.yaml')

            self.dataset_accessor = DatasetConfigAccessor(dataset_config)
            self.model_accessor = ModelConfigAccessor(model_config, hyperparams_config)

            if not self.dataset_accessor.validate_config(self.data_key):
                raise ValueError(f"Invalid configuration for {self.data_key}")

            logging.info("Configurations loaded successfully")
            return True
        except Exception as e:
            logging.error(f"Error loading configurations: {str(e)}")
            raise

    def load_data(self) -> None:
        """Stage 2: Load and initially clean the data"""
        try:
            data_path, delimiter = self.dataset_accessor.get_data_path(self.data_key)
            data_loader = DataLoader(data_path, delimiter)
            self.data_frame = data_loader.load_and_clean_data()

            logging.info(f"Data loaded successfully. Shape: {self.data_frame.shape}")
            logging.info(f"Available columns: {list(self.data_frame.columns)}")
            return True
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise

    def preprocess_data(self) -> None:
        """Stage 3: Handle missing values and categorical variables"""
        try:
            mv_handler = MissingValueHandler()

            # Log initial missing values
            missing_info_original = mv_handler.get_missing_info(self.data_frame)
            logging.info(f"Missing values in original dataset:\n{missing_info_original}")

            # Handle missing values
            mv_handler.fit_numeric_imputer(self.data_frame, strategy='mean')
            mv_handler.fit_categorical_imputer(self.data_frame, strategy='most_frequent')
            self.processed_data = mv_handler.fit_transform(self.data_frame)

            # Handle stratification if configured
            if self.dataset_accessor.has_stratification(self.data_key):
                split_config = self.dataset_accessor.get_split_config(self.data_key)
                stratify_config = split_config.stratify_column
                threshold = split_config.stratify_threshold
                self.processed_data = combine_rare_categories(
                    self.processed_data,
                    stratify_config,
                    threshold=threshold
                )

            logging.info("Preprocessing completed successfully")
            return True
        except Exception as e:
            logging.error(f"Error in preprocessing: {str(e)}")
            raise

    def split_data(self) -> None:
        """Stage 4: Split the data into train/val/test sets"""
        try:
            split_config = self.dataset_accessor.get_split_config(self.data_key)
            splitter = DataSplitter(split_config)

            df_train, df_val, df_test, y_train, y_val, y_test = splitter.split(self.processed_data)
            self.train_data = (df_train, y_train)
            self.val_data = (df_val, y_val)
            self.test_data = (df_test, y_test)

            logging.info(f"""
            Dataset splits:
            Train: {df_train.shape} (X), {y_train.shape} (y)
            Validation: {df_val.shape} (X), {y_val.shape} (y)
            Test: {df_test.shape} (X), {y_test.shape} (y)
            """)
            return True
        except Exception as e:
            logging.error(f"Error splitting data: {str(e)}")
            raise

    def validate_target(self) -> None:
        """Stage 5: Validate target variable"""
        try:
            target_validation = self.dataset_accessor.get_target_validation(self.data_key)
            _, y_train = self.train_data

            target_type, stats = validate_target_variable(
                y_train,
                target_type=target_validation.type,
                n_classes=target_validation.n_classes,
                value_range=target_validation.value_range
            )

            logging.info(f"Target validation complete. Type: {target_type.value}")
            logging.info(f"Target statistics: {stats}")
            return True
        except Exception as e:
            logging.error(f"Error validating target: {str(e)}")
            raise

    def train_models(self) -> None:
        """Stage 6: Train the models"""
        try:
            self.trainer = ModelTrainer(
                model_accessor=self.model_accessor,
                dataset_accessor=self.dataset_accessor,
                data_key=self.data_key
            )

            # Preprocess features
            df_train, y_train = self.train_data
            df_val, y_val = self.val_data
            df_test, _ = self.test_data

            X_train_processed, X_val_processed, X_test_processed = self.trainer.preprocess_data(
                df_train, df_val, df_test)

            logging.info("Training models...")
            self.trainer.train_models(X_train_processed, y_train)

            # Store processed data for evaluation
            self.processed_splits = (X_train_processed, X_val_processed, X_test_processed)
            return True
        except Exception as e:
            logging.error(f"Error training models: {str(e)}")
            raise

    def evaluate_models(self) -> None:
        """Stage 7: Evaluate models and generate predictions"""
        try:
            _, X_val_processed, X_test_processed = self.processed_splits
            _, y_val = self.val_data

            self.results = self.trainer.evaluate_models(X_val_processed, y_val)
            self.importance = self.trainer.get_feature_importance()
            self.predictions = self.trainer.predict(X_test_processed)

            logging.info("\nModel Evaluation Results:")
            logging.info("\n" + str(self.results))
            logging.info("\nFeature Importance:")
            logging.info("\n" + str(self.importance.head()))
            logging.info(f"\nPredictions shape: {self.predictions.shape}")
            return True
        except Exception as e:
            logging.error(f"Error evaluating models: {str(e)}")
            raise

    def save_results(self, output_dir: str = 'outputs') -> None:
        """Stage 8: Save all results"""
        try:
            save_results(self.results, self.importance, self.predictions, output_dir)
            logging.info("Pipeline results saved successfully")
            return True
        except Exception as e:
            logging.error(f"Error saving results: {str(e)}")
            raise

    def run_stage(self, stage: PipelineStage) -> bool:
        """Run a specific stage of the pipeline"""
        stage_mapping = {
            PipelineStage.DETECT_DATASETS: self.detect_datasets,
            PipelineStage.LOAD_CONFIG: self.load_config,
            PipelineStage.LOAD_DATA: self.load_data,
            PipelineStage.PREPROCESS: self.preprocess_data,
            PipelineStage.SPLIT_DATA: self.split_data,
            PipelineStage.VALIDATE_TARGET: self.validate_target,
            PipelineStage.TRAIN: self.train_models,
            PipelineStage.EVALUATE: self.evaluate_models,
            PipelineStage.SAVE_RESULTS: self.save_results
        }

        if stage not in stage_mapping:
            raise ValueError(f"Unknown pipeline stage: {stage}")

        logging.info(f"Running pipeline stage: {stage.name}")
        return stage_mapping[stage]()

    def run_pipeline(self, start_stage: Optional[PipelineStage] = None,
                    end_stage: Optional[PipelineStage] = None) -> None:
        """Run the pipeline with dataset detection"""
        # Check for new datasets first
        if self.check_new_datasets():
            logging.info("New dataset(s) configured. Reloading configurations...")
            self.load_config()  # Reload configurations after new datasets are added

        """Run the pipeline from start_stage to end_stage"""
        stages = list(PipelineStage)

        if start_stage:
            start_idx = stages.index(start_stage)
        else:
            start_idx = 0

        if end_stage:
            end_idx = stages.index(end_stage) + 1
        else:
            end_idx = len(stages)

        for stage in stages[start_idx:end_idx]:
            success = self.run_stage(stage)
            if not success:
                logging.error(f"Pipeline failed at stage: {stage.name}")
                break

        logging.info("Pipeline execution completed")
