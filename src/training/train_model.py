"""
This module provides a unified ModelTrainer class capable of handling both
classification and regression problems with automatic task type detection.
"""
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge,
    Lasso, ElasticNet
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, mean_absolute_error
)
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
from enum import Enum
from src.utils.model_config_accessor import ModelConfigAccessor
from src.utils.config_accessor import DatasetConfigAccessor, DatasetFeatures

class TaskType(Enum):
    """Enumeration for types of machine learning tasks."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"

class ModelTrainer:
    """Unified class for training and evaluating models with preprocessing."""

    def __init__(self, model_accessor: ModelConfigAccessor, dataset_accessor: DatasetConfigAccessor,
                 data_key: str, task_type: Optional[TaskType] = None):
        """
        Initialize ModelTrainer with configuration accessor.
        
        Args:
            model_accessor: Accessor for model and hyperparameter configurations
            dataset_accessor: Accessor for dataset configurations
            data_key: Key for the dataset being processed
            task_type: Optional task type (will be auto-detected if not provided)
        """
        self.model_accessor = model_accessor
        self.dataset_accessor = dataset_accessor
        self.data_key = data_key
        self.task_type = task_type
        self.models = {}
        self.preprocessor = None
        self.best_model = None
        self.feature_names = None
        self.label_encoder = LabelEncoder()

    def _detect_task_type(self, y: np.ndarray) -> TaskType:
        """Detect whether the problem is classification or regression."""
        unique_values = np.unique(y)
        
        if (
            isinstance(y[0], (bool, str)) or
            len(unique_values) <= min(10, len(y) * 0.05) or
            (np.issubdtype(y.dtype, np.integer) and len(unique_values) <= 10)
        ):
            return TaskType.CLASSIFICATION
        return TaskType.REGRESSION

    def _create_preprocessor(self, features: DatasetFeatures) -> ColumnTransformer:
        """Create preprocessing pipeline using dataset-specific features"""
        numerical_features = features.numerical_features
        categorical_features = features.categorical_features

        transformers = []
        
        if numerical_features:
            transformers.append(('num', StandardScaler(), numerical_features))
        
        if categorical_features:
            transformers.append(
                ('cat', OneHotEncoder(drop='first', sparse_output=False), 
                categorical_features)
            )

        return ColumnTransformer(transformers=transformers)

    def preprocess_data(self, X_train: pd.DataFrame, X_val: pd.DataFrame,
                       X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess training, validation, and test data.
        
        Args:
            X_train: Training features
            X_val: Validation features
            X_test: Test features
            
        Returns:
            Tuple of processed training, validation, and test features
        """
        features = self.dataset_accessor.get_features(self.data_key)
        
        if self.preprocessor is None:
            self.preprocessor = self._create_preprocessor(features)
            
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_val_processed = self.preprocessor.transform(X_val)
        X_test_processed = self.preprocessor.transform(X_test)

        # Get feature names
        feature_names = []
        
        # Add numerical feature names
        if features.numerical_features:
            feature_names.extend(features.numerical_features)
        
        # Add transformed categorical feature names
        if features.categorical_features:
            for i, feature in enumerate(features.categorical_features):
                cats = self.preprocessor.named_transformers_['cat'].categories_[i][1:]
                feature_names.extend([f"{feature}_{cat}" for cat in cats])

        self.feature_names = feature_names

        return X_train_processed, X_val_processed, X_test_processed

    def _preprocess_target(self, y: np.ndarray) -> np.ndarray:
        """Preprocess target variable (e.g., log transform for price data)"""
        split_config = self.dataset_accessor.get_split_config(self.data_key)
        if split_config.log_transform:
            return np.log1p(y)
        return y

    def _postprocess_predictions(self, y_pred: np.ndarray) -> np.ndarray:
        """Reverse target preprocessing"""
        split_config = self.dataset_accessor.get_split_config(self.data_key)
        if split_config.log_transform:
            return np.expm1(y_pred)
        return y_pred

    def train_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """Train multiple models based on task type."""
        if self.task_type is None:
            self.task_type = self._detect_task_type(y_train)

        task_str = self.task_type.value
        enabled_models = self.model_accessor.get_enabled_models(task_str)

        if self.task_type == TaskType.CLASSIFICATION:
            return self._train_classification_models(X_train, y_train, enabled_models)
        return self._train_regression_models(X_train, y_train, enabled_models)

    def _train_classification_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                                   enabled_models: List[str]) -> Dict:
        """Train classification models."""
        try:
            y_train_encoded = self.label_encoder.fit_transform(y_train)
        except Exception as e:
            raise ValueError(f"Error encoding target variable: {str(e)}. "
                           f"Target values: {np.unique(y_train)}") from e
        
        models = {}

        if 'logistic' in enabled_models:
            params = self.model_accessor.get_model_params('logistic')
            for c in params['C']:
                models[f'logistic_C_{c}'] = LogisticRegression(
                    C=c,
                    random_state=params['random_state'],
                    max_iter=1000
                ).fit(X_train, y_train_encoded)

        self.models = models
        return models

    def _train_regression_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                               enabled_models: List[str]) -> Dict:
        """Train regression models."""
        y_train_processed = self._preprocess_target(y_train)
        models = {}

        if 'linear' in enabled_models:
            params = self.model_accessor.get_model_params('linear')
            models['linear'] = LinearRegression().fit(X_train, y_train_processed)

        if 'ridge' in enabled_models:
            params = self.model_accessor.get_model_params('ridge')
            for alpha in params['alphas']:
                models[f'ridge_alpha_{alpha}'] = Ridge(
                    alpha=alpha,
                    random_state=params['random_state'],
                    max_iter=params['max_iter'],
                    tol=params['tol']
                ).fit(X_train, y_train_processed)

        if 'lasso' in enabled_models:
            params = self.model_accessor.get_model_params('lasso')
            for alpha in params['alphas']:
                models[f'lasso_alpha_{alpha}'] = Lasso(
                    alpha=alpha,
                    random_state=params['random_state'],
                    max_iter=params['max_iter'],
                    tol=params['tol']
                ).fit(X_train, y_train_processed)

        if 'elastic_net' in enabled_models:
            params = self.model_accessor.get_model_params('elastic_net')
            for alpha in params['alphas']:
                models[f'elastic_net_alpha_{alpha}'] = ElasticNet(
                    alpha=alpha,
                    l1_ratio=params['l1_ratio'],
                    random_state=params['random_state'],
                    max_iter=params['max_iter'],
                    tol=params['tol']
                ).fit(X_train, y_train_processed)

        self.models = models
        return models

    def evaluate_models(self, X_val: np.ndarray, y_val: np.ndarray) -> pd.DataFrame:
        """Evaluate models based on task type."""
        cv_settings = self.model_accessor.get_cv_settings()
        
        if self.task_type == TaskType.CLASSIFICATION:
            return self._evaluate_classification_models(X_val, y_val, cv_settings)
        return self._evaluate_regression_models(X_val, y_val, cv_settings)

    def _evaluate_classification_models(self, X_val: np.ndarray, y_val: np.ndarray, 
                                     cv_settings: Dict) -> pd.DataFrame:
        """Evaluate classification models."""
        try:
            y_val_encoded = self.label_encoder.transform(y_val)
        except Exception as e:
            raise ValueError(f"Error transforming validation target: {str(e)}") from e
            
        results = []

        for name, model in self.models.items():
            y_pred = model.predict(X_val)

            metrics = {
                'model': name,
                'accuracy': accuracy_score(y_val_encoded, y_pred),
                'precision': precision_score(y_val_encoded, y_pred, zero_division=0),
                'recall': recall_score(y_val_encoded, y_pred, zero_division=0),
                'f1_score': f1_score(y_val_encoded, y_pred, zero_division=0),
            }

            cv_scores = cross_val_score(
                model, X_val, y_val_encoded,
                cv=cv_settings['n_splits'],
                scoring='accuracy'
            )
            metrics['cv_accuracy'] = cv_scores.mean()

            results.append(metrics)

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('f1_score', ascending=False)
        
        # Format numeric columns
        numeric_cols = ['accuracy', 'precision', 'recall', 'f1_score', 'cv_accuracy']
        results_df[numeric_cols] = results_df[numeric_cols].round(4)
        
        self.best_model = self.models[results_df.iloc[0]['model']]
        return results_df

    def _evaluate_regression_models(self, X_val: np.ndarray, y_val: np.ndarray, 
                                  cv_settings: Dict) -> pd.DataFrame:
        """Evaluate regression models."""
        results = []

        for name, model in self.models.items():
            y_pred_log = model.predict(X_val)
            y_pred = self._postprocess_predictions(y_pred_log)

            mse = mean_squared_error(y_val, y_pred)
            metrics = {
                'model': name,
                'rmse': np.sqrt(mse),
                'mae': mean_absolute_error(y_val, y_pred),
                'r2': r2_score(y_val, y_pred)
            }

            y_val_log = self._preprocess_target(y_val)
            cv_scores = cross_val_score(
                model, X_val, y_val_log,
                cv=cv_settings['n_splits'],
                scoring='neg_mean_squared_error'
            )
            metrics['cv_rmse'] = np.sqrt(-cv_scores.mean())

            results.append(metrics)

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('rmse')
        
        # Format numeric columns
        numeric_cols = ['rmse', 'mae', 'r2', 'cv_rmse']
        results_df[numeric_cols] = results_df[numeric_cols].round(4)
        
        self.best_model = self.models[results_df.iloc[0]['model']]
        return results_df

    def get_feature_importance(self, model: Optional[BaseEstimator] = None) -> pd.DataFrame:
        """Get feature importance scores."""
        if model is None:
            model = self.best_model

        if hasattr(model, 'coef_'):
            coef = model.coef_[0] if self.task_type == TaskType.CLASSIFICATION else model.coef_
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': np.abs(coef)
            })
            return importance.sort_values('importance', ascending=False)
        
        raise ValueError("Model does not support feature importance calculation")

    def predict(self, X: np.ndarray, model: Optional[BaseEstimator] = None) -> np.ndarray:
        """Make predictions using specified or best model."""
        if model is None:
            model = self.best_model
            
        predictions = model.predict(X)
        
        if self.task_type == TaskType.CLASSIFICATION:
            return self.label_encoder.inverse_transform(predictions)
        return self._postprocess_predictions(predictions)