"""
Unit tests for data processing module
"""

import pytest
import pandas as pd
import numpy as np
from src.data_processing import (
    RFMCalculator, AggregateFeatureEngineer, TemporalFeatureEngineer,
    CategoricalEncoder, CreditRiskDataPipeline
)


class TestRFMCalculator:
    """Tests for RFM calculation"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample transaction data"""
        return pd.DataFrame({
            'CustomerId': ['C1', 'C1', 'C2', 'C2', 'C2'],
            'Value': [100, 200, 150, 50, 75],
            'TransactionStartTime': pd.date_range('2024-01-01', periods=5)
        })
    
    def test_rfm_calculation(self, sample_data):
        """Test that RFM metrics are calculated correctly"""
        calculator = RFMCalculator()
        rfm = calculator.calculate(sample_data)
        
        assert len(rfm) == 2
        assert 'Recency' in rfm.columns
        assert 'Frequency' in rfm.columns
        assert 'Monetary' in rfm.columns
    
    def test_rfm_monetary_sum(self, sample_data):
        """Test that monetary value is correctly summed"""
        calculator = RFMCalculator()
        rfm = calculator.calculate(sample_data)
        
        # Customer C1: 100 + 200 = 300
        assert rfm.loc['C1', 'Monetary'] == 300
        # Customer C2: 150 + 50 + 75 = 275
        assert rfm.loc['C2', 'Monetary'] == 275
    
    def test_rfm_frequency_count(self, sample_data):
        """Test that frequency is correctly counted"""
        calculator = RFMCalculator()
        rfm = calculator.calculate(sample_data)
        
        assert rfm.loc['C1', 'Frequency'] == 2
        assert rfm.loc['C2', 'Frequency'] == 3


class TestAggregateFeatureEngineer:
    """Tests for aggregate feature engineering"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample transaction data"""
        return pd.DataFrame({
            'CustomerId': ['C1', 'C1', 'C2', 'C2'],
            'Value': [100.0, 200.0, 150.0, 50.0]
        })
    
    def test_aggregate_features_created(self, sample_data):
        """Test that aggregate features are created"""
        features = AggregateFeatureEngineer.create_features(sample_data)
        
        assert len(features) == 2
        assert 'Value_sum' in features.columns
        assert 'Value_mean' in features.columns
        assert 'Value_count' in features.columns
    
    def test_aggregate_features_values(self, sample_data):
        """Test that aggregate feature values are correct"""
        features = AggregateFeatureEngineer.create_features(sample_data)
        
        # Customer C1 sum: 100 + 200 = 300
        assert features[features['CustomerId'] == 'C1']['Value_sum'].values[0] == 300
        # Customer C1 mean: 300 / 2 = 150
        assert features[features['CustomerId'] == 'C1']['Value_mean'].values[0] == 150
        # Customer C1 count: 2
        assert features[features['CustomerId'] == 'C1']['Value_count'].values[0] == 2


class TestTemporalFeatureEngineer:
    """Tests for temporal feature engineering"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data with datetime"""
        return pd.DataFrame({
            'TransactionStartTime': pd.date_range('2024-01-01 14:30:00', periods=3),
            'Amount': [100, 200, 150]
        })
    
    def test_temporal_features_created(self, sample_data):
        """Test that temporal features are created"""
        features = TemporalFeatureEngineer.create_features(sample_data)
        
        assert 'transaction_hour' in features.columns
        assert 'transaction_day' in features.columns
        assert 'transaction_month' in features.columns
        assert 'transaction_year' in features.columns
    
    def test_temporal_feature_values(self, sample_data):
        """Test that temporal feature values are correct"""
        features = TemporalFeatureEngineer.create_features(sample_data)
        
        assert features.iloc[0]['transaction_hour'] == 14
        assert features.iloc[0]['transaction_month'] == 1
        assert features.iloc[0]['transaction_year'] == 2024


class TestCategoricalEncoder:
    """Tests for categorical encoding"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data with categorical variables"""
        return pd.DataFrame({
            'CountryCode': ['US', 'UK', 'US', 'FR'],
            'ChannelId': ['web', 'mobile', 'web', 'mobile']
        })
    
    def test_categorical_encoding(self, sample_data):
        """Test that categorical encoding works"""
        encoder = CategoricalEncoder(encoding_type='onehot')
        encoded = encoder.fit_transform(sample_data, ['CountryCode', 'ChannelId'])
        
        # Check that original columns still exist
        assert 'CountryCode' in encoded.columns or 'CountryCode_UK' in encoded.columns
        assert 'ChannelId' in encoded.columns or 'ChannelId_mobile' in encoded.columns


class TestWoEEncodingAndSplitting:
    """Tests for WoE encoding and deterministic splitting"""

    @pytest.fixture
    def encoded_payload(self):
        df = pd.DataFrame({
            'CustomerId': ['C1', 'C2', 'C3', 'C4', 'C5'],
            'is_high_risk': [1, 0, 1, 0, 1],
            'primary_channel': ['web', 'mobile', 'web', 'mobile', 'web'],
            'primary_category': ['cat1', 'cat2', 'cat1', 'cat2', 'cat1'],
            'primary_currency': ['USD', 'EUR', 'USD', 'EUR', 'USD'],
            'primary_pricing': ['A', 'B', 'A', 'B', 'A'],
            'Recency': [10, 20, 15, 30, 5],
            'Frequency': [3, 1, 4, 2, 5],
            'Monetary': [100, 50, 80, 40, 120],
        })
        pipeline = CreditRiskDataPipeline(random_state=99)
        encoded_df, woe_maps = pipeline.apply_woe_encoding(df)
        return pipeline, encoded_df, woe_maps

    def test_apply_woe_encoding_adds_columns(self, encoded_payload):
        pipeline, encoded_df, woe_maps = encoded_payload

        for col in pipeline.CATEGORICAL_FEATURES:
            assert f"{col}_woe" in encoded_df.columns
            assert col in woe_maps

        assert 'CustomerId' in encoded_df.columns
        assert 'is_high_risk' in encoded_df.columns

    def test_split_scale_dataset_reproducible(self, encoded_payload):
        pipeline, encoded_df, woe_maps = encoded_payload

        first = pipeline.split_scale_dataset(encoded_df, woe_maps, test_size=0.4)
        second = pipeline.split_scale_dataset(encoded_df, woe_maps, test_size=0.4)

        pd.testing.assert_frame_equal(first.X_train, second.X_train)
        pd.testing.assert_series_equal(first.y_train, second.y_train)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
