# =====================================================
# Model Monitoring - Streamlit + Evidently
# Proyecto Integrador Computación en la Nube
# Perfil: Data Scientist Junior
# =====================================================

import os
import time
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from evidently import Report
from evidently.presets import DataDriftPreset

from carga_datos import cargarDatos

st.set_page_config(
    page_title="Monitoreo del Modelo",
    layout="wide"
)

API_URL = "http://localhost:8000/predict"
MONITOR_LOG = "monitoring_log.csv"
TARGET = "Pago_atiempo"

@st.cache_data
def load_reference_data():
    df = cargarDatos()

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_ref, X_new, _, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_ref, X_new


X_ref, X_new = load_reference_data()

def get_predictions(X_batch: pd.DataFrame):
    payload = {
        "records": X_batch.to_dict(orient="records")
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()["predictions"]

    except Exception as e:
        st.error(f"❌ Error al consumir la API: {e}")
        return None

def log_predictions(X_batch: pd.DataFrame, preds: list):
    log_df = X_batch.copy()
    log_df["prediction"] = preds
    log_df["timestamp"] = pd.Timestamp.now()

    if os.path.exists(MONITOR_LOG):
        log_df.to_csv(MONITOR_LOG, mode="a", header=False, index=False)
    else:
        log_df.to_csv(MONITOR_LOG, index=False)

def generate_drift_report(reference_df, current_df):
    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference_df,
        current_data=current_df
    )
    return report

st.title("📊 Monitoreo del Modelo en Producción")

if os.path.exists(MONITOR_LOG):
    try:
        logged_data = pd.read_csv(MONITOR_LOG)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total de predicciones", len(logged_data))
        col2.metric("Predicción promedio", f"{logged_data['prediction'].mean():.3f}")
        col3.metric("Desviación estándar", f"{logged_data['prediction'].std():.3f}")
        col4.metric(
            "Tasa positiva (%)",
            f"{(logged_data['prediction'] > 0.5).mean() * 100:.1f}%"
        )

    except Exception:
        st.warning("⚠️ Error leyendo el log de monitoreo")

st.sidebar.header("Opciones de monitoreo")

sample_size = st.sidebar.slider(
    "Tamaño de muestra",
    min_value=50,
    max_value=500,
    value=200
)

if st.sidebar.button("🔄 Generar predicciones"):
    sample = X_new.sample(
        n=sample_size,
        random_state=int(time.time())
    )

    preds = get_predictions(sample)

    if preds is not None:
        log_predictions(sample, preds)
        st.success("✅ Predicciones registradas")
        st.rerun()

if os.path.exists(MONITOR_LOG):
    logged_data = pd.read_csv(MONITOR_LOG)

    tab1, tab2, tab3 = st.tabs(
        ["📈 Visualizaciones", "📊 Data Drift", "📂 Logs"]
    )

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            fig_hist = px.histogram(
                logged_data,
                x="prediction",
                nbins=20,
                title="Distribución de Predicciones"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            logged_data["timestamp"] = pd.to_datetime(logged_data["timestamp"])
            temporal_df = logged_data.groupby(
                logged_data["timestamp"].dt.floor("T")
            )["prediction"].mean().reset_index()

            fig_time = px.line(
                temporal_df,
                x="timestamp",
                y="prediction",
                title="Evolución Temporal de Predicciones"
            )
            st.plotly_chart(fig_time, use_container_width=True)

        st.subheader("🔍 Comparación de medias (Referencia vs Actual)")

        numeric_cols = X_ref.select_dtypes(include="number").columns[:4]
        comparison = []

        for col in numeric_cols:
            comparison.append({
                "Variable": col,
                "Referencia": X_ref[col].mean(),
                "Actual": logged_data[col].mean()
            })

        comp_df = pd.DataFrame(comparison)

        fig_comp = go.Figure()
        fig_comp.add_bar(x=comp_df["Variable"], y=comp_df["Referencia"], name="Referencia")
        fig_comp.add_bar(x=comp_df["Variable"], y=comp_df["Actual"], name="Actual")

        fig_comp.update_layout(
            title="Comparación de Medias",
            barmode="group"
        )

        st.plotly_chart(fig_comp, use_container_width=True)

    # -------------------------
    # TAB 2: Data Drift
    # -------------------------
    with tab2:
        st.subheader("📊 Detección de Data Drift")

        drift_report = generate_drift_report(
            X_ref,
            logged_data.drop(columns=["prediction", "timestamp"], errors="ignore")
        )

        try:
            st.components.v1.html(
                drift_report._repr_html_(),
                height=900,
                scrolling=True
            )
        except Exception:
            st.info("Reporte de drift generado correctamente")

    # -------------------------
    # TAB 3: Logs
    # -------------------------
    with tab3:
        st.subheader("📂 Historial de predicciones")

        show_rows = st.selectbox(
            "Mostrar últimas filas",
            [10, 25, 50, 100],
            index=0
        )

        st.dataframe(
            logged_data.tail(show_rows),
            use_container_width=True
        )

        st.download_button(
            "📥 Descargar log",
            logged_data.to_csv(index=False),
            "monitoring_log.csv",
            "text/csv"
        )

else:
    st.warning("⚠️ Aún no hay datos de monitoreo registrados")