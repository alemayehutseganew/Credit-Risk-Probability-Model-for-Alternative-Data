"""
Prediction and inference script
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import joblib
import mlflow
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScoringConfig:
    """Configuration for scoring and recommendations."""

    threshold: float = 0.5
    base_score: int = 300
    max_score: int = 850
    min_amount: int = 1000
    max_amount: int = 100000
    min_months: int = 6
    max_months: int = 36


class CreditRiskPredictor:
    """Make predictions using trained credit risk model"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        mlflow_model_uri: Optional[str] = None,
        scoring_config: Optional[ScoringConfig] = None,
    ) -> None:
        """
        Initialize predictor
        
        Args:
            model_path (str): Path to saved model file
            mlflow_model_uri (str): MLflow model URI
        """
        self.model = None
        self.scoring_config = scoring_config or ScoringConfig()
        
        if model_path:
            self.load_from_file(model_path)
        elif mlflow_model_uri:
            self.load_from_mlflow(mlflow_model_uri)
    
    def load_from_file(self, model_path: str) -> None:
        """Load model from disk"""
        self.model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
    
    def load_from_mlflow(self, model_uri: str) -> None:
        """Load model from MLflow"""
        self.model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Model loaded from MLflow: {model_uri}")
    
    def predict_risk_probability(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict risk probability for customers
        
        Args:
            X (pd.DataFrame): Customer features
        
        Returns:
            np.ndarray: Risk probability scores [0, 1]
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        if not hasattr(self.model, "predict_proba"):
            raise ValueError("Loaded model does not support predict_proba")

        return self.model.predict_proba(X)[:, 1]
    
    def predict_risk_category(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """
        Predict risk category (low/high)
        
        Args:
            X (pd.DataFrame): Customer features
            threshold (float): Classification threshold
        
        Returns:
            np.ndarray: Risk categories
        """
        risk_proba = self.predict_risk_probability(X)
        applied_threshold = self.scoring_config.threshold if threshold is None else threshold
        return np.where(risk_proba >= applied_threshold, 'high_risk', 'low_risk')
    
    def predict_credit_score(
        self,
        X: pd.DataFrame,
        base_score: Optional[int] = None,
        max_score: Optional[int] = None,
    ) -> np.ndarray:
        """
        Convert risk probability to credit score
        
        Args:
            X (pd.DataFrame): Customer features
            base_score (int): Minimum credit score
            max_score (int): Maximum credit score
        
        Returns:
            np.ndarray: Credit scores
        """
        risk_proba = self.predict_risk_probability(X)
        applied_base = self.scoring_config.base_score if base_score is None else base_score
        applied_max = self.scoring_config.max_score if max_score is None else max_score
        # Higher probability = lower score
        credit_scores = applied_base + (1 - risk_proba) * (applied_max - applied_base)
        return credit_scores.astype(int)
    
    def predict_loan_amount(
        self,
        X: pd.DataFrame,
        min_amount: Optional[float] = None,
        max_amount: Optional[float] = None,
        reference_amount_col: Optional[str] = None,
    ) -> np.ndarray:
        """
        Recommend loan amount based on risk
        
        Args:
            X (pd.DataFrame): Customer features
            min_amount (float): Minimum loan amount
            max_amount (float): Maximum loan amount
            reference_amount_col (str): Column with reference amounts
        
        Returns:
            np.ndarray: Recommended loan amounts
        """
        risk_proba = self.predict_risk_probability(X)
        applied_min = self.scoring_config.min_amount if min_amount is None else min_amount
        applied_max = self.scoring_config.max_amount if max_amount is None else max_amount

        # Lower risk = higher loan amount
        risk_factor = 1 - risk_proba

        if reference_amount_col and reference_amount_col in X.columns:
            recommended = X[reference_amount_col].values * risk_factor
        else:
            recommended = applied_min + risk_factor * (applied_max - applied_min)
        
        # Clip to min/max
        recommended = np.clip(recommended, applied_min, applied_max)
        return recommended.astype(int)
    
    def predict_loan_duration(
        self,
        X: pd.DataFrame,
        min_months: Optional[int] = None,
        max_months: Optional[int] = None,
    ) -> np.ndarray:
        """
        Recommend loan duration based on risk
        
        Args:
            X (pd.DataFrame): Customer features
            min_months (int): Minimum loan duration
            max_months (int): Maximum loan duration
        
        Returns:
            np.ndarray: Recommended durations in months
        """
        risk_proba = self.predict_risk_probability(X)
        applied_min = self.scoring_config.min_months if min_months is None else min_months
        applied_max = self.scoring_config.max_months if max_months is None else max_months

        # Lower risk = longer duration
        risk_factor = 1 - risk_proba
        recommended = applied_min + risk_factor * (applied_max - applied_min)
        
        return recommended.astype(int)
    
    def predict_batch(
        self,
        X: pd.DataFrame,
        include_score: bool = True,
        include_loan: bool = True,
    ) -> pd.DataFrame:
        """
        Make comprehensive predictions for batch of customers
        
        Args:
            X (pd.DataFrame): Customer features
            include_score (bool): Include credit score
            include_loan (bool): Include loan recommendations
        
        Returns:
            pd.DataFrame: Predictions with all components
        """
        results = pd.DataFrame({
            'risk_probability': self.predict_risk_probability(X),
            'risk_category': self.predict_risk_category(X)
        })
        
        if include_score:
            results['credit_score'] = self.predict_credit_score(X)
        
        if include_loan:
            results['recommended_amount'] = self.predict_loan_amount(X)
            results['recommended_duration_months'] = self.predict_loan_duration(X)
        
        return results


def main() -> None:
    """Example usage"""
    logging.basicConfig(level=logging.INFO)
    
    # Example: Load model and make predictions
    # predictor = CreditRiskPredictor(model_path='models/best_model.pkl')
    # predictions = predictor.predict_batch(X_new)
    
    logger.info("Prediction module ready")


if __name__ == "__main__":
    main()
