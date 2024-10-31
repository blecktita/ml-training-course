# pylint: disable=astroid-error
from dataclasses import dataclass
from typing import List, Dict, Optional, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ElasticNetParams:
    l1_ratio: float
    max_iter: int
    tol: float

@dataclass
class LassoParams:
    max_iter: int
    tol: float

@dataclass
class RidgeParams:
    max_iter: int
    tol: float

@dataclass
class HyperParameters:
    alphas: List[float]
    c_values: List[float]
    elastic_net_params: ElasticNetParams
    lasso_params: LassoParams
    ridge_params: RidgeParams
    random_state: int
    cv_folds: int

@dataclass
class ModelConfig:
    classification_models: Dict[str, bool]
    regression_models: Dict[str, bool]

class ModelConfigAccessor:
    """Accessor class for model and hyperparameter configurations"""
    
    def __init__(self, model_config: Dict, hyperparams_config: Dict):
        """
        Initialize with model and hyperparameter configurations.
        
        Args:
            model_config: Model configuration dictionary
            hyperparams_config: Hyperparameter configuration dictionary
        """
        self.model_config = model_config
        self.hyperparams_config = hyperparams_config
        self._initialize_configs()
    
    def _initialize_configs(self) -> None:
        """Initialize structured configurations"""
        # Initialize hyperparameters
        hyper = self.hyperparams_config['hyperparameters']
        self.hyperparameters = HyperParameters(
            alphas=hyper['alphas'],
            c_values=hyper['c_values'],
            elastic_net_params=ElasticNetParams(**hyper['elastic_net_params']),
            lasso_params=LassoParams(**hyper['lasso_params']),
            ridge_params=RidgeParams(**hyper['ridge_params']),
            random_state=self.hyperparams_config['random_state'],
            cv_folds=self.hyperparams_config['cv_folds']
        )
        
        # Initialize model configuration
        self.models = ModelConfig(**self.model_config['models'])
    
    def get_enabled_models(self, task_type: str) -> List[str]:
        """
        Get list of enabled models for a specific task type.
        
        Args:
            task_type: Either 'classification' or 'regression'
            
        Returns:
            List of enabled model names
        """
        if task_type == 'classification':
            return [name for name, enabled in self.models.classification_models.items() 
                   if enabled]
        elif task_type == 'regression':
            return [name for name, enabled in self.models.regression_models.items() 
                   if enabled]
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def get_model_params(self, model_name: str) -> Dict:
        """
        Get hyperparameters for a specific model.
        
        Args:
            model_name: Name of the model (e.g., 'elastic_net', 'lasso', etc.)
            
        Returns:
            Dictionary of hyperparameters for the model
        """
        params = {
            'random_state': self.hyperparameters.random_state,
            'cv': self.hyperparameters.cv_folds
        }
        
        if model_name == 'elastic_net':
            params.update({
                'alphas': self.hyperparameters.alphas,
                'l1_ratio': self.hyperparameters.elastic_net_params.l1_ratio,
                'max_iter': self.hyperparameters.elastic_net_params.max_iter,
                'tol': self.hyperparameters.elastic_net_params.tol
            })
        elif model_name == 'lasso':
            params.update({
                'alphas': self.hyperparameters.alphas,
                'max_iter': self.hyperparameters.lasso_params.max_iter,
                'tol': self.hyperparameters.lasso_params.tol
            })
        elif model_name == 'ridge':
            params.update({
                'alphas': self.hyperparameters.alphas,
                'max_iter': self.hyperparameters.ridge_params.max_iter,
                'tol': self.hyperparameters.ridge_params.tol
            })
        elif model_name == 'logistic':
            params.update({
                'C': self.hyperparameters.c_values
            })
            
        return params
    
    def validate_config(self) -> bool:
        """Validate that all required configurations are present"""
        try:
            # Check hyperparameters
            assert self.hyperparameters.alphas, "Alphas not configured"
            assert self.hyperparameters.c_values, "C values not configured"
            assert self.hyperparameters.cv_folds > 1, "CV folds must be > 1"
            
            # Check if at least one model is enabled for each task
            has_classification = any(self.models.classification_models.values())
            has_regression = any(self.models.regression_models.values())
            assert has_classification or has_regression, "No models enabled"
            
            return True
        except AssertionError as e:
            logger.error(f"Invalid configuration: {str(e)}")
            return False

    def get_cv_settings(self) -> Dict:
        """Get cross-validation settings"""
        return {
            'n_splits': self.hyperparameters.cv_folds,
            'random_state': self.hyperparameters.random_state
        }