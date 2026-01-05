import os
import sys
import streamlit as st
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from utils.data_loader import load_csv_safe
from utils.preprocessing import preprocess_data
from utils.model import run_regression
from utils.visualization import (
    kpi_cards,
    performance_trend_chart,
    actual_vs_predicted_chart,
    target_distribution,
    residual_plot
)
from utils.summary import generate_summary

st.set_page_config(
    page_title="AI Analytics Dashboard",
    layout="wide",
    page_icon="📊"
)

st.title("📊 AI-Powered Analytics Dashboard")

# ---------------- Sidebar ----------------
st.sidebar.header("📁 Upload Data")
file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if file:
    df = load_csv_safe(file)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    target = st.sidebar.selectbox("🎯 Select Target Column", numeric_cols)

    # ---------- KPIs ----------
    kpi_cards(df, target)

    # ---------- ML ----------
    X_train, X_test, y_train, y_test = preprocess_data(df, target)
    results = run_regression(X_train, X_test, y_train, y_test)

    # ---------- Charts ----------
    st.markdown("---")
    col1, col2 = st.columns([3, 2])

    with col1:
        performance_trend_chart(df, target)

    with col2:
        actual_vs_predicted_chart(y_test, results["y_pred"])

    st.markdown("---")
    target_distribution(df, target)
    residual_plot(y_test, results["y_pred"])

    # ---------- Metrics ----------
    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("MSE", round(results["mse"], 2))
    m2.metric("RMSE", round(results["rmse"], 2))
    m3.metric("R² Score", round(results["r2"], 2))

    st.success(
        f"📌 Sample Prediction (from test data): {round(results['future_prediction'], 2)}"
    )

    st.markdown("## 🧠 Executive Summary")
    st.info(generate_summary(df, target, results))

else:
    st.info("⬅ Upload a CSV file to start analysis")
