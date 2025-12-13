"""
Prediction and inference script
"""

import logging
import numpy as np
import pandas as pd
import joblib
import mlflow

logger = logging.getLogger(__name__)


class CreditRiskPredictor:
    """Make predictions using trained credit risk model"""
    
    def __init__(self, model_path=None, mlflow_model_uri=None):
        """
        Initialize predictor
        
        Args:
            model_path (str): Path to saved model file
            mlflow_model_uri (str): MLflow model URI
        """
        self.model = None
        
        if model_path:
            self.load_from_file(model_path)
        elif mlflow_model_uri:
            self.load_from_mlflow(mlflow_model_uri)
    
    def load_from_file(self, model_path):
        """Load model from disk"""
        self.model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
    
    def load_from_mlflow(self, model_uri):
        """Load model from MLflow"""
        self.model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Model loaded from MLflow: {model_uri}")
    
    def predict_risk_probability(self, X):
        """
        Predict risk probability for customers
        
        Args:
            X (pd.DataFrame): Customer features
        
        Returns:
            np.ndarray: Risk probability scores [0, 1]
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        risk_proba = self.model.predict_proba(X)[:, 1]
        return risk_proba
    
    def predict_risk_category(self, X, threshold=0.5):
        """
        Predict risk category (low/high)
        
        Args:
            X (pd.DataFrame): Customer features
            threshold (float): Classification threshold
        
        Returns:
            np.ndarray: Risk categories
        """
        risk_proba = self.predict_risk_probability(X)
        return np.where(risk_proba >= threshold, 'high_risk', 'low_risk')
    
    def predict_credit_score(self, X, base_score=300, max_score=850):
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
        # Higher probability = lower score
        credit_scores = base_score + (1 - risk_proba) * (max_score - base_score)
        return credit_scores.astype(int)
    
    def predict_loan_amount(self, X, min_amount=1000, max_amount=100000,
                           reference_amount_col=None):
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
        
        # Lower risk = higher loan amount
        risk_factor = 1 - risk_proba
        
        if reference_amount_col and reference_amount_col in X.columns:
            recommended = X[reference_amount_col].values * risk_factor
        else:
            recommended = min_amount + risk_factor * (max_amount - min_amount)
        
        # Clip to min/max
        recommended = np.clip(recommended, min_amount, max_amount)
        return recommended.astype(int)
    
    def predict_loan_duration(self, X, min_months=6, max_months=36):
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
        
        # Lower risk = longer duration
        risk_factor = 1 - risk_proba
        recommended = min_months + risk_factor * (max_months - min_months)
        
        return recommended.astype(int)
    
    def predict_batch(self, X, include_score=True, include_loan=True):
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


def main():
    """Example usage"""
    logging.basicConfig(level=logging.INFO)
    
    # Example: Load model and make predictions
    # predictor = CreditRiskPredictor(model_path='models/best_model.pkl')
    # predictions = predictor.predict_batch(X_new)
    
    logger.info("Prediction module ready")


if __name__ == "__main__":
    main()
