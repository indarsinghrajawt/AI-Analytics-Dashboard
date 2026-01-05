import streamlit as st
import plotly.express as px
import pandas as pd


def kpi_cards(df, target):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("TOTAL VALUE (YTD)", int(df[target].sum()))
    c2.metric("AVERAGE SCORE", round(df[target].mean(), 2))
    c3.metric("BEST PERFORMANCE", int(df[target].max()))
    c4.metric("DATA POINTS", len(df))


def performance_trend_chart(df, target):
    st.subheader("📈 Performance Trend")

    trend_df = (
        df[[target]]
        .reset_index(drop=True)
        .assign(rolling_mean=lambda x: x[target].rolling(window=20).mean())
    )

    fig = px.line(
        trend_df,
        x=trend_df.index,
        y="rolling_mean",
        labels={
            "index": "Index",
            "rolling_mean": f"{target} (Rolling Avg)"
        }
    )

    fig.update_layout(
        template="plotly_dark",
        height=350
    )

    st.plotly_chart(fig, use_container_width=True)


def actual_vs_predicted_chart(y_test, y_pred):
    st.subheader("🔍 Actual vs Predicted")

    fig = px.scatter(
        x=y_test,
        y=y_pred,
        labels={"x": "Actual", "y": "Predicted"}
    )

    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)


def target_distribution(df, target):
    st.subheader("📊 Target Distribution")

    fig = px.histogram(df, x=target, nbins=20)
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)


def residual_plot(y_test, y_pred):
    st.subheader("🧮 Residual Error Analysis")

    residuals = y_test - y_pred
    fig = px.scatter(
        x=y_pred,
        y=residuals,
        labels={"x": "Predicted", "y": "Residual Error"}
    )

    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
