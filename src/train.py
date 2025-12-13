"""Model training script with hyperparameter tuning and MLflow tracking"""

import json
import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.data_processing import FEATURE_SCHEMA_PATH, PROCESSED_DIR, WOE_MAPPING_PATH

MLFLOW_MODEL_REQUIREMENTS = [
    "mlflow==3.7.0",
    "scikit-learn==1.8.0",
    "pandas==2.3.3",
    "numpy==2.3.5",
]

logger = logging.getLogger(__name__)


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Expected file missing: {path}")
    return json.loads(path.read_text())


def load_training_artifacts(processed_dir: Path = PROCESSED_DIR):
    """Load processed datasets along with feature schema and WoE mappings"""

    processed_dir = Path(processed_dir)
    required = ['X_train.csv', 'X_test.csv', 'y_train.csv', 'y_test.csv']
    missing = [fname for fname in required if not (processed_dir / fname).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing processed artifacts: {missing}. Run run_data_processing_pipeline first."
        )

    feature_schema = _read_json(FEATURE_SCHEMA_PATH)
    feature_columns = feature_schema.get('features', [])
    if not feature_columns:
        raise ValueError("Feature schema does not contain any features")

    woe_mappings = _read_json(WOE_MAPPING_PATH)

    X_train = pd.read_csv(processed_dir / 'X_train.csv')[feature_columns]
    X_test = pd.read_csv(processed_dir / 'X_test.csv')[feature_columns]
    y_train = pd.read_csv(processed_dir / 'y_train.csv').squeeze('columns')
    y_test = pd.read_csv(processed_dir / 'y_test.csv').squeeze('columns')

    return X_train, y_train, X_test, y_test, feature_columns, woe_mappings


class ModelTrainer:
    """Handle model training, hyperparameter tuning, and evaluation"""
    
    def __init__(self, model_name, model, param_grid, param_dist=None):
        """
        Initialize model trainer
        
        Args:
            model_name (str): Name of the model
            model: Scikit-learn model instance
            param_grid (dict): Parameter grid for GridSearchCV
            param_dist (dict): Parameter distribution for RandomizedSearchCV
        """
        self.model_name = model_name
        self.model = model
        self.param_grid = param_grid
        self.param_dist = param_dist or param_grid
        self.best_model = None
        self.results = {}
    
    def grid_search(self, X_train, y_train, cv=5, n_jobs=-1):
        """
        Perform grid search for hyperparameter tuning
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            cv (int): Number of cross-validation folds
            n_jobs (int): Number of parallel jobs
        
        Returns:
            dict: Grid search results
        """
        logger.info(f"Starting Grid Search for {self.model_name}")
        
        grid_search = GridSearchCV(
            self.model, self.param_grid, cv=cv, n_jobs=n_jobs, 
            scoring='roc_auc', verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_model = grid_search.best_estimator_
        self.results['grid_search'] = {
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_,
            'best_estimator': grid_search.best_estimator_,
        }
        
        logger.info(f"Best CV Score: {grid_search.best_score_:.4f}")
        logger.info(f"Best Parameters: {grid_search.best_params_}")
        
        return self.results['grid_search']
    
    def random_search(self, X_train, y_train, cv=5, n_iter=20, n_jobs=-1):
        """
        Perform random search for hyperparameter tuning
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            cv (int): Number of cross-validation folds
            n_iter (int): Number of iterations
            n_jobs (int): Number of parallel jobs
        
        Returns:
            dict: Random search results
        """
        logger.info(f"Starting Random Search for {self.model_name}")
        
        random_search = RandomizedSearchCV(
            self.model, self.param_dist, cv=cv, n_iter=n_iter,
            n_jobs=n_jobs, scoring='roc_auc', verbose=1, random_state=42
        )
        
        random_search.fit(X_train, y_train)
        
        self.best_model = random_search.best_estimator_
        self.results['random_search'] = {
            'best_params': random_search.best_params_,
            'best_cv_score': random_search.best_score_,
            'best_estimator': random_search.best_estimator_,
        }
        
        logger.info(f"Best CV Score: {random_search.best_score_:.4f}")
        logger.info(f"Best Parameters: {random_search.best_params_}")
        
        return self.results['random_search']
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test set
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
        
        Returns:
            dict: Evaluation metrics
        """
        if self.best_model is None:
            raise ValueError("Model not trained yet. Run grid_search or random_search first.")
        
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        self.results['evaluation'] = metrics
        logger.info(f"Evaluation Results: {metrics}")
        
        return metrics
    
    def save_model(self, filepath):
        """
        Save trained model to disk
        
        Args:
            filepath (str): Path to save model
        """
        if self.best_model is None:
            raise ValueError("No trained model to save")
        
        joblib.dump(self.best_model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load model from disk
        
        Args:
            filepath (str): Path to model file
        """
        self.best_model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")


def train_and_track_models(
    X_train,
    y_train,
    X_test,
    y_test,
    feature_columns,
    woe_mappings,
    experiment_name: str = 'credit-risk-model',
):
    """
    Train multiple models and track with MLflow
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        experiment_name (str): MLflow experiment name
    
    Returns:
        dict: Results from all models
    """
    mlflow.set_experiment(experiment_name)
    
    # Model configurations
    models_config = {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=1000, random_state=42),
            'param_grid': {
                'C': [0.1, 1.0, 10.0],
                'solver': ['lbfgs', 'liblinear'],
                'penalty': ['l2'],
            },
            'search_type': 'grid',
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'param_grid': {
                'n_estimators': [50, 100],
                'max_depth': [10, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
            },
            'param_dist': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [None, 5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2'],
            },
            'search_type': 'random',
            'n_iter': 20,
        },
    }
    
    results = {}
    best_run = {'roc_auc': -np.inf, 'run_id': None, 'model_uri': None, 'model_name': None}
    
    minority_class_count = int(y_train.value_counts().min())
    cv_folds = max(2, min(5, minority_class_count))

    for model_name, config in models_config.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {model_name}")
        logger.info(f"{'='*50}")
        
        with mlflow.start_run(run_name=model_name) as run:
            trainer = ModelTrainer(
                model_name,
                config['model'],
                config['param_grid'],
                config.get('param_dist'),
            )

            search_type = config.get('search_type', 'grid')
            if search_type == 'random':
                mlflow.log_param('tuning_method', 'random_search')
                search_results = trainer.random_search(
                    X_train,
                    y_train,
                    cv=cv_folds,
                    n_iter=config.get('n_iter', 20),
                    n_jobs=1,
                )
            else:
                mlflow.log_param('tuning_method', 'grid_search')
                search_results = trainer.grid_search(
                    X_train,
                    y_train,
                    cv=cv_folds,
                    n_jobs=1,
                )

            # Evaluation
            metrics = trainer.evaluate(X_test, y_test)
            mlflow.log_metric('cv_best_roc_auc', search_results['best_cv_score'])

            # Log to MLflow
            mlflow.log_param('model_type', model_name)
            mlflow.log_param('feature_count', len(feature_columns))
            mlflow.log_param('feature_columns', '|'.join(feature_columns))
            mlflow.log_param('train_positive_rate', float(y_train.mean()))
            mlflow.log_params(search_results['best_params'])

            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            mlflow.log_dict({'features': feature_columns}, 'metadata/feature_schema.json')
            mlflow.log_dict(woe_mappings, 'metadata/woe_mappings.json')
            mlflow.log_artifact(FEATURE_SCHEMA_PATH, artifact_path='metadata')
            mlflow.log_artifact(WOE_MAPPING_PATH, artifact_path='metadata')
            mlflow.log_text('\n'.join(X_train.columns), 'metadata/feature_list.txt')
            # Log model with explicit requirements to skip slow inference
            mlflow.sklearn.log_model(
                trainer.best_model,
                model_name.lower().replace(' ', '_'),
                pip_requirements=MLFLOW_MODEL_REQUIREMENTS,
            )

            model_uri = f"runs:/{run.info.run_id}/{model_name.lower().replace(' ', '_')}"
            if metrics['roc_auc'] > best_run['roc_auc']:
                best_run = {
                    'roc_auc': metrics['roc_auc'],
                    'run_id': run.info.run_id,
                    'model_uri': model_uri,
                    'model_name': model_name,
                }
            
            results[model_name] = trainer.results
    
    if best_run['run_id']:
        registry_name = f"{experiment_name}-best-model"
        mlflow.register_model(best_run['model_uri'], registry_name)
        logger.info(
            "Registered best model %s with ROC-AUC %.4f to registry %s",
            best_run['model_name'],
            best_run['roc_auc'],
            registry_name,
        )

    return results


def main():
    """Main training pipeline"""
    logging.basicConfig(level=logging.INFO)
    
    logger.info("Loading processed training artifacts...")
    X_train, y_train, X_test, y_test, feature_columns, woe_mappings = load_training_artifacts()
    logger.info("Feature matrix ready with %s columns", len(feature_columns))

    results = train_and_track_models(
        X_train,
        y_train,
        X_test,
        y_test,
        feature_columns,
        woe_mappings,
    )

    logger.info("Training complete for %d models", len(results))


if __name__ == "__main__":
    main()
