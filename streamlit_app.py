"""
streamlit_app.py - Simple UI for Credit Risk Prediction

This app demonstrates:
1. A frontend (Streamlit) calling a backend API (FastAPI)
2. The model lives in FastAPI, not here
3. We only send HTTP requests and display results
4. MLflow integration for model monitoring and experiment tracking

Run with:
    streamlit run streamlit_app.py

Make sure FastAPI is running on http://localhost:8000 first!
"""

import requests
import streamlit as st
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime
import uuid

# ============================================================
# CONFIGURATION
# ============================================================
# Use secrets or env vars for production, fallback to localhost for dev
import os
API_URL = os.getenv("API_URL", "http://localhost:8001/predict")
DEFAULT_MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_REGISTRY_NAME = "credit-risk-model-best-model"

# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Risk Predictor (Alternative Data)")
st.markdown("Enter customer transaction details below to get a **risk prediction**.")
st.markdown("---")

# ============================================================
# MLFLOW SIDEBAR
# ============================================================
st.sidebar.title("🔍 MLOps Dashboard")
mlflow_uri = st.sidebar.text_input("MLflow Tracking URI", DEFAULT_MLFLOW_URI)
mlflow.set_tracking_uri(mlflow_uri)

# Connect to MLflow
client = None
try:
    client = MlflowClient()
    # Test connection
    client.search_experiments(max_results=1)
    st.sidebar.success("✅ Connected to MLflow")
except Exception as e:
    st.sidebar.error(f"❌ MLflow Connection Failed: {e}")
    client = None

if client:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📦 Model Registry")
    
    try:
        # Get latest versions of the model
        versions = client.get_latest_versions(MODEL_REGISTRY_NAME)
        if versions:
            latest_version = versions[0]
            st.sidebar.markdown(f"**Model:** `{MODEL_REGISTRY_NAME}`")
            st.sidebar.markdown(f"**Version:** `{latest_version.version}`")
            st.sidebar.markdown(f"**Stage:** `{latest_version.current_stage}`")
            
            # Show creation time
            ts = latest_version.creation_timestamp / 1000
            created_date = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M')
            st.sidebar.caption(f"Created: {created_date}")
        else:
            st.sidebar.warning(f"No versions found for {MODEL_REGISTRY_NAME}")
            
    except Exception as e:
        st.sidebar.warning(f"Could not fetch model info: {e}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("🧪 Recent Experiments")
    try:
        # List recent runs from the main experiment
        experiment = client.get_experiment_by_name("credit-risk-model")
        if experiment:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=5,
                order_by=["attribute.start_time DESC"]
            )
            
            for run in runs:
                run_name = run.data.tags.get("mlflow.runName", run.info.run_id[:7])
                status = run.info.status
                st.sidebar.text(f"{run_name} ({status})")
                if "roc_auc" in run.data.metrics:
                    st.sidebar.caption(f"AUC: {run.data.metrics['roc_auc']:.4f}")
        else:
            st.sidebar.info("Experiment 'credit-risk-model' not found.")
            
    except Exception as e:
        st.sidebar.error(f"Error listing runs: {e}")

# ============================================================
# INPUT FORM
# ============================================================

# Generate a random customer ID
if 'customer_id' not in st.session_state:
    st.session_state.customer_id = f"CUST_{uuid.uuid4().hex[:8].upper()}"

st.caption(f"Customer ID: {st.session_state.customer_id}")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("RFM Metrics")
    recency = st.number_input("Recency (Days since last txn)", min_value=0.0, value=5.0)
    frequency = st.number_input("Frequency (Txn count in window)", min_value=0.0, value=25.0)
    monetary = st.number_input("Monetary (Total value)", min_value=0.0, value=5000.0)
    transaction_count = st.number_input("Total Transaction Count", min_value=1, value=58)

with col2:
    st.subheader("Transaction Stats (Amount)")
    total_amount = st.number_input("Total Amount", value=10234.5)
    avg_amount = st.number_input("Average Amount", value=312.5)
    std_amount = st.number_input("Std Dev Amount", value=45.7)
    min_amount = st.number_input("Min Amount", value=10.0)
    max_amount = st.number_input("Max Amount", value=1500.0)

with col3:
    st.subheader("Transaction Stats (Value)")
    total_value = st.number_input("Total Value", value=10987.6)
    avg_value = st.number_input("Average Value", value=189.4)
    std_value = st.number_input("Std Dev Value", value=32.1)
    min_value = st.number_input("Min Value", value=5.0)
    max_value = st.number_input("Max Value", value=2000.0)

st.markdown("---")
col4, col5 = st.columns(2)

with col4:
    st.subheader("Ratios")
    debit_ratio = st.slider("Debit Ratio", 0.0, 1.0, 0.7)
    credit_ratio = st.slider("Credit Ratio", 0.0, 1.0, 0.3)

    st.subheader("Temporal Features")
    transaction_year_mean = st.number_input("Mean Year", value=2019.0)
    transaction_month_mean = st.slider("Mean Month", 1.0, 12.0, 6.0)
    transaction_day_mean = st.slider("Mean Day", 1.0, 31.0, 15.0)
    transaction_hour_mean = st.slider("Mean Hour", 0.0, 23.0, 12.0)
    transaction_dayofweek_mean = st.slider("Mean Day of Week", 0.0, 6.0, 3.0)
    weekend_transaction_ratio = st.slider("Weekend Ratio", 0.0, 1.0, 0.2)

with col5:
    st.subheader("Categorical Features")
    primary_channel = st.selectbox("Primary Channel", ["Channel_A", "Channel_B", "Channel_C", "UNKNOWN"])
    primary_category = st.selectbox("Primary Category", ["Category_X", "Category_Y", "Category_Z", "UNKNOWN"])
    primary_currency = st.selectbox("Primary Currency", ["USD", "EUR", "GBP", "UNKNOWN"])
    primary_pricing = st.selectbox("Primary Pricing", ["Standard", "Premium", "Discount", "UNKNOWN"])

# ============================================================
# PREDICTION & LOGGING
# ============================================================
st.markdown("---")

# Option to log inference to MLflow
log_inference = st.checkbox("Log this prediction to MLflow (Experiment: 'streamlit_inference')", value=True)

if st.button("🔮 Get Prediction", use_container_width=True):
    
    payload = {
        "customer_id": st.session_state.customer_id,
        "Recency": recency,
        "Frequency": frequency,
        "Monetary": monetary,
        "total_amount": total_amount,
        "avg_amount": avg_amount,
        "std_amount": std_amount,
        "min_amount": min_amount,
        "max_amount": max_amount,
        "transaction_count": int(transaction_count),
        "total_value": total_value,
        "avg_value": avg_value,
        "std_value": std_value,
        "min_value": min_value,
        "max_value": max_value,
        "debit_ratio": debit_ratio,
        "credit_ratio": credit_ratio,
        "transaction_year_mean": transaction_year_mean,
        "transaction_month_mean": transaction_month_mean,
        "transaction_day_mean": transaction_day_mean,
        "transaction_hour_mean": transaction_hour_mean,
        "transaction_dayofweek_mean": transaction_dayofweek_mean,
        "weekend_transaction_ratio": weekend_transaction_ratio,
        "primary_channel": primary_channel,
        "primary_category": primary_category,
        "primary_currency": primary_currency,
        "primary_pricing": primary_pricing
    }
    
    with st.spinner("Calling the model API..."):
        try:
            response = requests.post(API_URL, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
            
            # Display results
            st.markdown("### 📊 Prediction Result")
            res_col1, res_col2, res_col3 = st.columns(3)
            
            with res_col1:
                st.metric(
                    label="Default Probability",
                    value=f"{result['risk_probability']:.1%}"
                )
            
            with res_col2:
                risk = result['risk_category']
                if risk == "high_risk":
                    st.error(f"Risk Level: **{risk.upper()}** ⚠️")
                else:
                    st.success(f"Risk Level: **{risk.upper()}** ✅")
            
            with res_col3:
                st.metric(
                    label="Credit Score",
                    value=result['credit_score']
                )

            st.info(f"💡 Recommendation: Loan Amount ${result['recommended_amount']:,.2f} for {result['recommended_duration_months']} months")
            
            # Log to MLflow if requested
            if log_inference and client:
                try:
                    mlflow.set_experiment("streamlit_inference")
                    with mlflow.start_run(run_name=f"inference_{datetime.now().strftime('%H%M%S')}"):
                        # Log inputs
                        mlflow.log_params({k: v for k, v in payload.items() if k != 'customer_id'})
                        # Log output
                        mlflow.log_metric("risk_probability", result['risk_probability'])
                        mlflow.log_metric("credit_score", result['credit_score'])
                        mlflow.log_param("risk_category", result['risk_category'])
                        
                    st.toast("✅ Prediction logged to MLflow!", icon="📝")
                except Exception as e:
                    st.error(f"Failed to log to MLflow: {e}")

            with st.expander("See API details"):
                st.json(payload)
                st.json(result)
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API. Is FastAPI running on http://localhost:8000?")
        except requests.exceptions.HTTPError as e:
             st.error(f"❌ API Error: {e}")
             if e.response is not None:
                 st.json(e.response.json())
        except Exception as e:
            st.error(f"❌ Error: {e}")

# Footer
st.markdown("---")
st.caption("This UI calls the FastAPI model service and tracks experiments via MLflow.")
