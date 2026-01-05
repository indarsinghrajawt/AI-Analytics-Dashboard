import os
import sys
import streamlit as st
import pandas as pd

# ---------------- PATH FIX ----------------
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

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Analytics Dashboard",
    layout="wide",
    page_icon="📊"
)

# ---------------- NEW PREMIUM THEME (CSS) ----------------
st.markdown("""
<style>
/* Global background */
.stApp {
  background: radial-gradient(1200px 600px at 10% 10%, #0b1220 0%, #020617 60%);
  color: #e5e7eb;
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #020617, #020617);
  border-right: 1px solid #1e293b;
}

/* Metric cards */
div[data-testid="metric-container"] {
  background: linear-gradient(180deg, #020617, #0b1220);
  border: 1px solid #1e293b;
  border-radius: 16px;
  padding: 18px 16px;
  box-shadow: 0 10px 30px rgba(0,0,0,.25);
}
div[data-testid="metric-container"] label {
  color: #9ca3af !important;
}
div[data-testid="metric-container"] div {
  color: #e5e7eb !important;
  font-size: 1.6rem;
}

/* Headings */
h1, h2, h3 {
  font-weight: 700;
  color: #e5e7eb;
}

/* Plotly background */
.js-plotly-plot .plotly {
  background: transparent !important;
}

/* Alerts */
.stAlert {
  border-radius: 14px;
  border: 1px solid #1e293b;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("📊 AI-Powered Analytics Dashboard")

# ---------------- SIDEBAR ----------------
st.sidebar.header("📁 Upload Data")
file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if file:
    df = load_csv_safe(file)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    target = st.sidebar.selectbox("🎯 Select Target Column", numeric_cols)

    # ---------- KPIs ----------
    st.subheader("📌 Key Metrics")
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
    st.info("⬅ Upload a CSV file from the sidebar to start analysis")
