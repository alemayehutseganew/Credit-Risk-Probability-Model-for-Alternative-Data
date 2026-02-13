"""
Pydantic models for API request/response validation
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime


class CustomerFeatures(BaseModel):
    """Model-ready customer feature payload"""

    customer_id: str = Field(..., description="Unique customer identifier")
    Recency: float = Field(..., description="Days since last transaction")
    Frequency: float = Field(..., description="Number of transactions in window")
    Monetary: float = Field(..., description="Total monetary value in window")
    total_amount: float = Field(..., description="Sum of Amount column")
    avg_amount: float = Field(..., description="Average Amount value")
    std_amount: float = Field(0.0, description="Std dev of Amount")
    min_amount: float = Field(..., description="Minimum Amount")
    max_amount: float = Field(..., description="Maximum Amount")
    transaction_count: int = Field(..., description="Raw transaction count")
    total_value: float = Field(..., description="Sum of Value column")
    avg_value: float = Field(..., description="Average Value")
    std_value: float = Field(0.0, description="Std dev of Value")
    min_value: float = Field(..., description="Minimum Value")
    max_value: float = Field(..., description="Maximum Value")
    debit_ratio: float = Field(..., description="Share of positive Amount transactions")
    credit_ratio: float = Field(..., description="Share of negative Amount transactions")
    
    # Temporal features
    transaction_hour_mean: float = Field(12.0, description="Mean hour of transactions")
    transaction_day_mean: float = Field(15.0, description="Mean day of month")
    transaction_month_mean: float = Field(6.0, description="Mean month")
    transaction_year_mean: float = Field(2019.0, description="Mean year")
    transaction_dayofweek_mean: float = Field(3.0, description="Mean day of week (0-6)")
    weekend_transaction_ratio: float = Field(0.0, description="Ratio of weekend transactions")

    primary_channel: str = Field('UNKNOWN', description="Most common transaction channel")
    primary_category: str = Field('UNKNOWN', description="Most common product category")
    primary_currency: str = Field('UNKNOWN', description="Most common currency code")
    primary_pricing: str = Field('UNKNOWN', description="Most common pricing strategy")

    class Config:
        schema_extra = {
            "example": {
                "customer_id": "CUST_12345",
                "Recency": 12,
                "Frequency": 34,
                "Monetary": 8450.0,
                "total_amount": 10234.5,
                "avg_amount": 312.5,
                "std_amount": 45.7,
                "min_amount": 10.0,
                "max_amount": 1500.0,
                "transaction_count": 58,
                "total_value": 10987.6,
                "avg_value": 189.4,
                "std_value": 32.1,
                "min_value": 8.5,
                "max_value": 900.0,
                "debit_ratio": 0.82,
                "credit_ratio": 0.18,
                "primary_channel": "MOBILE",
                "primary_category": "AIRTIME",
                "primary_currency": "UGX",
                "primary_pricing": "PRICING_STRATEGY_1"
            }
        }


class PredictionResponse(BaseModel):
    """API response with predictions"""
    
    customer_id: str = Field(..., description="Customer identifier")
    risk_probability: float = Field(..., description="Risk probability [0-1]")
    risk_category: str = Field(..., description="Risk category: 'low_risk' or 'high_risk'")
    credit_score: int = Field(..., description="Credit score [300-850]")
    recommended_amount: float = Field(..., description="Recommended loan amount")
    recommended_duration_months: int = Field(..., description="Recommended loan duration")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Prediction timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "customer_id": "CUST_12345",
                "risk_probability": 0.23,
                "risk_category": "low_risk",
                "credit_score": 750,
                "recommended_amount": 50000,
                "recommended_duration_months": 12,
                "timestamp": "2024-12-10T10:30:00"
            }
        }


class HealthCheckResponse(BaseModel):
    """Health check response"""
    
    status: str = Field(..., description="Service status")
    model_version: Optional[str] = Field(None, description="Model version")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """Error response"""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
