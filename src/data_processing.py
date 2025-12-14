"""
Data processing and feature engineering pipeline for credit risk model
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "data.csv"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURE_SCHEMA_PATH = PROCESSED_DIR / "feature_schema.json"
WOE_MAPPING_PATH = PROCESSED_DIR / "woe_mappings.json"
PREPROCESSOR_PATH = PROCESSED_DIR / "feature_preprocessor.joblib"


DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42


class RFMCalculator:
    """Calculate Recency, Frequency, and Monetary metrics for customers"""
    
    def __init__(self, snapshot_date=None):
        """
        Initialize RFM calculator
        
        Args:
            snapshot_date (datetime): Reference date for recency calculation.
                                      Defaults to max date in data.
        """
        self.snapshot_date = snapshot_date
    
    def calculate(self, df, customer_id_col='CustomerId', 
                  amount_col='Value', date_col='TransactionStartTime'):
        """
        Calculate RFM metrics for each customer
        
        Args:
            df (pd.DataFrame): Transaction data
            customer_id_col (str): Column name for customer ID
            amount_col (str): Column name for transaction amount
            date_col (str): Column name for transaction date
        
        Returns:
            pd.DataFrame: RFM metrics per customer
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        if self.snapshot_date is None:
            self.snapshot_date = df[date_col].max()
        
        rfm = df.groupby(customer_id_col).agg({
            date_col: lambda x: (self.snapshot_date - x.max()).days,  # Recency
            customer_id_col: 'count',  # Frequency
            amount_col: 'sum'  # Monetary
        }).rename(columns={
            date_col: 'Recency',
            customer_id_col: 'Frequency',
            amount_col: 'Monetary'
        })
        
        logger.info(f"Calculated RFM metrics for {len(rfm)} customers")
        return rfm


class AggregateFeatureEngineer:
    """Create aggregate features from transaction data"""
    
    @staticmethod
    def create_features(df, customer_id_col='CustomerId', amount_col='Value'):
        """
        Create aggregate features per customer
        
        Args:
            df (pd.DataFrame): Transaction data
            customer_id_col (str): Customer ID column
            amount_col (str): Transaction amount column
        
        Returns:
            pd.DataFrame: Aggregate features
        """
        df = df.copy()
        df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce')
        
        features = df.groupby(customer_id_col).agg({
            amount_col: [
                'sum',      # Total amount
                'mean',     # Average amount
                'std',      # Standard deviation
                'min',      # Minimum
                'max',      # Maximum
                'count'     # Transaction count
            ]
        }).reset_index()
        
        # Flatten column names
        features.columns = [
            f"{col[0]}_{col[1]}" if col[1] else col[0]
            for col in features.columns.values
        ]
        features.rename(columns={f"{customer_id_col}_": customer_id_col}, 
                       inplace=True)
        
        # Fill NaN std with 0
        features.fillna(0, inplace=True)
        
        logger.info(f"Created aggregate features with {features.shape[1]} columns")
        return features


class TemporalFeatureEngineer:
    """Extract temporal features from transaction timestamps"""
    
    @staticmethod
    def create_features(df, date_col='TransactionStartTime'):
        """
        Extract temporal features
        
        Args:
            df (pd.DataFrame): Transaction data
            date_col (str): Datetime column name
        
        Returns:
            pd.DataFrame: DataFrame with temporal features
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        df['transaction_hour'] = df[date_col].dt.hour
        df['transaction_day'] = df[date_col].dt.day
        df['transaction_month'] = df[date_col].dt.month
        df['transaction_year'] = df[date_col].dt.year
        df['transaction_dayofweek'] = df[date_col].dt.dayofweek
        df['transaction_quarter'] = df[date_col].dt.quarter
        
        logger.info("Created temporal features")
        return df


class CategoricalEncoder:
    """Handle categorical variable encoding"""
    
    def __init__(self, encoding_type='onehot', low_cardinality_threshold=10):
        """
        Initialize encoder
        
        Args:
            encoding_type (str): 'onehot' or 'label'
            low_cardinality_threshold (int): Threshold for one-hot encoding
        """
        self.encoding_type = encoding_type
        self.low_cardinality_threshold = low_cardinality_threshold
        self.encoders = {}
    
    def fit_transform(self, df, categorical_cols):
        """
        Fit and transform categorical variables
        
        Args:
            df (pd.DataFrame): Input data
            categorical_cols (list): List of categorical column names
        
        Returns:
            pd.DataFrame: Encoded data
        """
        df = df.copy()
        
        for col in categorical_cols:
            if col not in df.columns:
                continue
            
            cardinality = df[col].nunique()
            
            if self.encoding_type == 'onehot' or cardinality <= self.low_cardinality_threshold:
                # One-hot encoding
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                self.encoders[col] = 'onehot'
            else:
                # Label encoding
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
        
        logger.info(f"Encoded {len(categorical_cols)} categorical variables")
        return df


@dataclass
class ProcessedDataset:
    """Container for processed customer-level dataset"""

    customer_features: pd.DataFrame
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_columns: List[str]
    woe_mappings: Dict[str, Dict[str, float]]
    preprocessor: Optional[Pipeline] = None


class CreditRiskDataPipeline:
    """End-to-end pipeline to build model-ready features from raw transactions"""

    NUMERIC_FEATURES = [
        'Recency', 'Frequency', 'Monetary',
        'total_amount', 'avg_amount', 'std_amount', 'min_amount', 'max_amount',
        'transaction_count', 'total_value', 'avg_value', 'std_value', 'min_value', 'max_value',
        'debit_ratio', 'credit_ratio',
        'transaction_hour_mean', 'transaction_day_mean', 'transaction_month_mean',
        'transaction_year_mean', 'transaction_dayofweek_mean', 'weekend_transaction_ratio'
    ]
    CATEGORICAL_FEATURES = ['primary_channel', 'primary_category', 'primary_currency', 'primary_pricing']

    def __init__(self, raw_path: Path = RAW_DATA_PATH, processed_dir: Path = PROCESSED_DIR,
                 snapshot_date: Optional[datetime] = None, random_state: int = DEFAULT_RANDOM_STATE):
        self.raw_path = Path(raw_path)
        self.processed_dir = Path(processed_dir)
        self.snapshot_date = snapshot_date
        self.random_state = random_state
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Data ingestion & helpers
    # ------------------------------------------------------------------
    def load_transactions(self) -> pd.DataFrame:
        """Load raw Kaggle transactions with required type coercion"""
        if not self.raw_path.exists():
            raise FileNotFoundError(
                f"Raw data not found at {self.raw_path}. Please download the Kaggle Xente file."
            )

        df = pd.read_csv(self.raw_path, parse_dates=['TransactionStartTime'])
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], utc=True).dt.tz_localize(None)
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        df = df.dropna(subset=['TransactionStartTime', 'Value'])
        logger.info("Loaded %s rows from %s", len(df), self.raw_path.name)
        return df

    def compute_rfm(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Recency, Frequency, Monetary metrics per customer"""
        calculator = RFMCalculator(snapshot_date=self.snapshot_date)
        rfm = calculator.calculate(df).reset_index()
        rfm = rfm.rename(columns={'CustomerId': 'CustomerId'})
        return rfm

    @staticmethod
    def _safe_mode(series: pd.Series) -> str:
        if series.isnull().all():
            return 'UNKNOWN'
        mode = series.mode()
        if mode.empty:
            return str(series.dropna().iloc[0])
        return str(mode.iloc[0])

    def _aggregate_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        agg = df.groupby('CustomerId').agg(
            total_amount=('Amount', 'sum'),
            avg_amount=('Amount', 'mean'),
            std_amount=('Amount', 'std'),
            min_amount=('Amount', 'min'),
            max_amount=('Amount', 'max'),
            transaction_count=('TransactionId', 'count'),
            total_value=('Value', 'sum'),
            avg_value=('Value', 'mean'),
            std_value=('Value', 'std'),
            min_value=('Value', 'min'),
            max_value=('Value', 'max'),
            debit_ratio=('Amount', lambda x: (x > 0).mean()),
            credit_ratio=('Amount', lambda x: (x < 0).mean()),
            transaction_hour_mean=('transaction_hour', 'mean'),
            transaction_day_mean=('transaction_day', 'mean'),
            transaction_month_mean=('transaction_month', 'mean'),
            transaction_year_mean=('transaction_year', 'mean'),
            transaction_dayofweek_mean=('transaction_dayofweek', 'mean'),
            weekend_transaction_ratio=('transaction_dayofweek', lambda x: (x >= 5).mean()),
        ).reset_index()
        agg[['std_amount', 'std_value']] = agg[['std_amount', 'std_value']].fillna(0)
        agg[['debit_ratio', 'credit_ratio', 'weekend_transaction_ratio']] = (
            agg[['debit_ratio', 'credit_ratio', 'weekend_transaction_ratio']].fillna(0)
        )
        temporal_cols = [
            'transaction_hour_mean',
            'transaction_day_mean',
            'transaction_month_mean',
            'transaction_year_mean',
            'transaction_dayofweek_mean',
        ]
        agg[temporal_cols] = agg[temporal_cols].fillna(0)
        return agg

    def _aggregate_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        agg = df.groupby('CustomerId').agg(
            primary_channel=('ChannelId', self._safe_mode),
            primary_category=('ProductCategory', self._safe_mode),
            primary_currency=('CurrencyCode', self._safe_mode),
            primary_pricing=('PricingStrategy', self._safe_mode),
        ).reset_index()
        return agg

    def _label_high_risk(self, rfm: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
        km = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        rfm['cluster'] = km.fit_predict(scaled)

        profile = rfm.groupby('cluster')[['Recency', 'Frequency', 'Monetary']].mean()
        profile['recency_rank'] = profile['Recency'].rank(ascending=False)
        profile['frequency_rank'] = profile['Frequency'].rank()
        profile['monetary_rank'] = profile['Monetary'].rank()
        profile['risk_score'] = profile[['recency_rank', 'frequency_rank', 'monetary_rank']].sum(axis=1)
        high_risk_cluster = profile['risk_score'].idxmax()

        rfm['is_high_risk'] = (rfm['cluster'] == high_risk_cluster).astype(int)
        logger.info("Identified high-risk cluster %s covering %.2f%% of customers",
                    high_risk_cluster, rfm['is_high_risk'].mean() * 100)
        return rfm.drop(columns=['cluster'])

    def build_customer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        enriched = TemporalFeatureEngineer.create_features(df)
        rfm = self._label_high_risk(self.compute_rfm(enriched))
        numeric = self._aggregate_numeric(enriched)
        categorical = self._aggregate_categorical(enriched)

        features = rfm.merge(numeric, on='CustomerId', how='left').merge(categorical, on='CustomerId', how='left')
        features[self.NUMERIC_FEATURES] = features[self.NUMERIC_FEATURES].fillna(0)
        features[self.CATEGORICAL_FEATURES] = features[self.CATEGORICAL_FEATURES].fillna('UNKNOWN')
        return features

    @staticmethod
    def _compute_woe_mapping(df: pd.DataFrame, column: str, target: str, smoothing: float = 0.5) -> Dict[str, float]:
        grouped = df.groupby(column)[target].agg(['sum', 'count'])
        grouped['non_event'] = grouped['count'] - grouped['sum']
        grouped['event_dist'] = (grouped['sum'] + smoothing) / (grouped['sum'].sum() + smoothing * len(grouped))
        grouped['non_event_dist'] = (grouped['non_event'] + smoothing) / (
            grouped['non_event'].sum() + smoothing * len(grouped)
        )
        grouped['woe'] = np.log((grouped['event_dist']) / (grouped['non_event_dist']))
        return grouped['woe'].to_dict()

    def apply_woe_encoding(self, df: pd.DataFrame, target_col: str = 'is_high_risk') -> tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
        woe_maps: Dict[str, Dict[str, float]] = {}
        encoded_df = df.copy()
        for col in self.CATEGORICAL_FEATURES:
            mapping = self._compute_woe_mapping(encoded_df, col, target_col)
            encoded_df[f"{col}_woe"] = encoded_df[col].map(mapping).fillna(0)
            woe_maps[col] = mapping
        encoded_df = encoded_df.drop(columns=self.CATEGORICAL_FEATURES)
        return encoded_df, woe_maps

    def split_scale_dataset(self, encoded_df: pd.DataFrame, woe_maps: Dict[str, Dict[str, float]],
                            test_size: float = DEFAULT_TEST_SIZE) -> ProcessedDataset:
        drop_cols = ['CustomerId', 'is_high_risk']
        feature_columns = [col for col in encoded_df.columns if col not in drop_cols]

        X = encoded_df[feature_columns].copy()
        y = encoded_df['is_high_risk'].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.random_state
        )

        preprocessor = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ])

        X_train_processed = pd.DataFrame(
            preprocessor.fit_transform(X_train),
            columns=feature_columns,
            index=X_train.index,
        )
        X_test_processed = pd.DataFrame(
            preprocessor.transform(X_test),
            columns=feature_columns,
            index=X_test.index,
        )

        logger.info("Split dataset: train=%s, test=%s", X_train_processed.shape, X_test_processed.shape)
        return ProcessedDataset(
            customer_features=encoded_df,
            X_train=X_train_processed,
            X_test=X_test_processed,
            y_train=y_train,
            y_test=y_test,
            feature_columns=feature_columns,
            woe_mappings=woe_maps,
            preprocessor=preprocessor,
        )

    def prepare_feature_matrix(self, customer_features: pd.DataFrame,
                               test_size: float = DEFAULT_TEST_SIZE) -> ProcessedDataset:
        encoded_df, woe_maps = self.apply_woe_encoding(customer_features)
        return self.split_scale_dataset(encoded_df, woe_maps, test_size=test_size)

    def build_processing_pipeline(self, test_size: float = DEFAULT_TEST_SIZE) -> Pipeline:
        """Create a scikit-learn Pipeline chaining customer feature creation, WoE encoding, and scaling."""

        def woe_step(features: pd.DataFrame) -> np.ndarray:
            encoded, woe_maps = self.apply_woe_encoding(features)
            return np.array([(encoded, woe_maps)], dtype=object)

        def split_scale_step(payload_array: np.ndarray) -> np.ndarray:
            encoded_df, woe_maps = payload_array[0]
            dataset = self.split_scale_dataset(
                encoded_df,
                woe_maps,
                test_size=test_size,
            )
            return np.array([dataset], dtype=object)

        processing_pipeline = Pipeline(steps=[
            ('customer_features', FunctionTransformer(self.build_customer_features, validate=False)),
            ('woe_encoding', FunctionTransformer(woe_step, validate=False)),
            ('train_test_split', FunctionTransformer(split_scale_step, validate=False)),
        ])

        logger.info("Initialized end-to-end processing pipeline")
        return processing_pipeline

    def persist_artifacts(self, dataset: ProcessedDataset) -> None:
        dataset.customer_features.to_csv(self.processed_dir / 'customer_features.csv', index=False)
        dataset.X_train.to_csv(self.processed_dir / 'X_train.csv', index=False)
        dataset.X_test.to_csv(self.processed_dir / 'X_test.csv', index=False)
        dataset.y_train.to_csv(self.processed_dir / 'y_train.csv', index=False)
        dataset.y_test.to_csv(self.processed_dir / 'y_test.csv', index=False)

        FEATURE_SCHEMA_PATH.write_text(json.dumps({'features': dataset.feature_columns}, indent=2))
        WOE_MAPPING_PATH.write_text(json.dumps(dataset.woe_mappings, indent=2))
        if dataset.preprocessor is not None:
            joblib.dump(dataset.preprocessor, PREPROCESSOR_PATH)
        logger.info("Artifacts saved to %s", self.processed_dir)


def run_data_processing_pipeline(raw_path: Path = RAW_DATA_PATH,
                                 processed_dir: Path = PROCESSED_DIR,
                                 snapshot_date: Optional[datetime] = None) -> ProcessedDataset:
    """Convenience wrapper that executes the full feature pipeline"""
    pipeline = CreditRiskDataPipeline(raw_path=raw_path, processed_dir=processed_dir, snapshot_date=snapshot_date)
    transactions = pipeline.load_transactions()
    processing_pipeline = pipeline.build_processing_pipeline()
    dataset = processing_pipeline.fit_transform(transactions)[0]
    pipeline.persist_artifacts(dataset)
    return dataset


def create_preprocessing_pipeline(numerical_cols, categorical_cols,
                                 strategy='mean'):
    """
    Create a scikit-learn preprocessing pipeline
    
    Args:
        numerical_cols (list): Numerical column names
        categorical_cols (list): Categorical column names
        strategy (str): Imputation strategy ('mean', 'median', 'most_frequent')
    
    Returns:
        Pipeline: Scikit-learn pipeline object
    """
    # Numerical transformer
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=strategy)),
        ('scaler', StandardScaler())
    ])
    
    # Categorical transformer
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    logger.info("Created preprocessing pipeline")
    return preprocessor


def load_and_process_data(filepath, target_col='is_high_risk'):
    """
    Load and process data for model training
    
    Args:
        filepath (str): Path to CSV file
        target_col (str): Target variable column name
    
    Returns:
        tuple: (X, y) - Features and target
    """
    df = pd.read_csv(filepath)
    logger.info(f"Loaded data with shape {df.shape}")
    
    if target_col in df.columns:
        y = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        X = df
        y = None
    
    return X, y


def extract_temporal_features(df, date_col='TransactionStartTime'):
    """Extract temporal features from transaction timestamps"""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['TransactionHour'] = df[date_col].dt.hour
    df['TransactionDay'] = df[date_col].dt.day
    df['TransactionMonth'] = df[date_col].dt.month
    df['TransactionYear'] = df[date_col].dt.year
    return df


def aggregate_features(df):
    """Aggregate transaction data to customer level"""
    def mode_func(x):
        return x.mode()[0] if not x.mode().empty else np.nan

    agg_funcs = {
        'TransactionId': 'count',
        'Amount': ['sum', 'mean', 'std', 'min', 'max'],
        'Value': ['sum', 'mean', 'std'],
        'TransactionHour': 'mean',
        'FraudResult': 'max',
        'ChannelId': mode_func,
        'ProductCategory': mode_func,
        'PricingStrategy': mode_func,
        'CurrencyCode': mode_func
    }
    
    # Check if columns exist before aggregating
    available_cols = df.columns.tolist()
    final_agg_funcs = {k: v for k, v in agg_funcs.items() if k in available_cols}
    
    customer_df = df.groupby('CustomerId').agg(final_agg_funcs)
    customer_df.columns = ['_'.join(col).strip() for col in customer_df.columns.values]
    customer_df.reset_index(inplace=True)
    
    # Rename columns
    rename_map = {
        'TransactionId_count': 'TransactionCount',
        'FraudResult_max': 'HasFraud',
        'Amount_sum': 'TotalAmount',
        'Amount_mean': 'AvgAmount',
        'Amount_std': 'StdAmount',
        'ChannelId_mode_func': 'PrimaryChannel',
        'ProductCategory_mode_func': 'PrimaryCategory',
        'PricingStrategy_mode_func': 'PrimaryPricing',
        'CurrencyCode_mode_func': 'PrimaryCurrency'
    }
    customer_df.rename(columns=rename_map, inplace=True)
    return customer_df


def handle_missing_values(df):
    """Handle missing values in customer data"""
    df = df.copy()
    cols_to_fill_0 = ['StdAmount', 'Value_std']
    for col in cols_to_fill_0:
        if col in df.columns:
            df[col] = df[col].fillna(0)
            
    numeric_features = ['TotalAmount', 'AvgAmount', 'TransactionCount', 'TransactionHour_mean']
    numeric_features = [col for col in numeric_features if col in df.columns]
    
    if numeric_features:
        imputer = SimpleImputer(strategy='median')
        df[numeric_features] = imputer.fit_transform(df[numeric_features])
        
    return df


def encode_categorical_features(df):
    """Encode categorical features"""
    df = df.copy()
    le = LabelEncoder()
    cols = ['PrimaryChannel', 'PrimaryCategory', 'PrimaryPricing', 'PrimaryCurrency']
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col + '_Encoded'] = le.fit_transform(df[col])
    return df


def normalize_features(df):
    """Normalize numerical features"""
    df = df.copy()
    scaler = StandardScaler()
    scaled_features = ['TotalAmount', 'AvgAmount', 'TransactionCount', 'TransactionHour_mean']
    scaled_features = [col for col in scaled_features if col in df.columns]
    
    if scaled_features:
        df[scaled_features] = scaler.fit_transform(df[scaled_features])
    return df


def woe_transformation(df, target_col, categorical_cols):
    """Calculate WoE and IV for categorical features"""
    df = df.copy()
    woe_df = pd.DataFrame()
    iv_dict = {}
    
    for col in categorical_cols:
        if col not in df.columns:
            continue
            
        # Calculate WoE
        grouped = df.groupby(col)[target_col].agg(['count', 'sum'])
        grouped.columns = ['total', 'bad']
        grouped['good'] = grouped['total'] - grouped['bad']
        
        # Avoid division by zero
        total_bad = grouped['bad'].sum()
        total_good = grouped['good'].sum()
        
        if total_bad == 0 or total_good == 0:
            continue
            
        grouped['bad_dist'] = (grouped['bad'] + 0.5) / total_bad
        grouped['good_dist'] = (grouped['good'] + 0.5) / total_good
        
        grouped['woe'] = np.log(grouped['good_dist'] / grouped['bad_dist'])
        grouped['iv'] = (grouped['good_dist'] - grouped['bad_dist']) * grouped['woe']
        
        # Map WoE back to dataframe
        woe_map = grouped['woe'].to_dict()
        woe_df[f"{col}_WoE"] = df[col].map(woe_map)
        
        # Calculate IV
        iv_dict[col] = grouped['iv'].sum()
        
    return woe_df, iv_dict


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Data processing module ready for import")
