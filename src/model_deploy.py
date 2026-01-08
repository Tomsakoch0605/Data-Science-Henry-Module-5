import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn

from carga_datos import cargarDatos
from ft_engineering import ft_engineering
from model_monitoring import generate_drift_report

class PredictionInput(BaseModel):
    bin_encoder_tipo_laboral: float
    poly_ohe_tipo_credito_9: float
    poly_ohe_tipo_credito_10: float
    poly_ohe_tendencia_ingresos_Decreciente: float
    poly_ohe_tendencia_ingresos_Estable: float
    capital_prestado: float
    plazo_meses: float
    edad_cliente: float
    salario_cliente: float
    total_otros_prestamos: float
    puntaje_datacredito: float
    cant_creditosvigentes: float
    huella_consulta: float
    saldo_total: float
    saldo_mora_codeudor: float
    creditos_sectorCooperativo: float
    creditos_sectorReal: float


class BatchPredictionInput(BaseModel):
    data: List[PredictionInput]

app = FastAPI(
    title="API de Predicción de Pago a Tiempo",
    description="API para predicción, evaluación y monitoreo del modelo XGBoost",
    version="1.1.0"
)

try:
    model = xgb.Booster()
    model.load_model("xgb_model.json")
    model_features = model.feature_names
except Exception as e:
    model = None
    print(f"❌ Error cargando el modelo: {e}")

@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "API de predicción de Pago a Tiempo activa"
    }

@app.post("/predict")
def predict_batch(input_data: BatchPredictionInput):
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Modelo no cargado"
        )

    try:
        input_list = [item.dict() for item in input_data.data]
        df = pd.DataFrame(input_list)

        # Asegurar orden de columnas
        df = df[model_features]

        dmatrix = xgb.DMatrix(df)
        probas = model.predict(dmatrix)

        threshold = 0.5
        predictions = [1 if p >= threshold else 0 for p in probas]

        return {
            "n_registros": len(predictions),
            "predictions": predictions,
            "probabilities": probas.tolist()
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

@app.get("/evaluation")
def evaluation():
    """
    Evalúa el modelo usando el dataset completo.
    Retorna métricas básicas.
    """
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Modelo no cargado"
        )

    try:
        df = cargarDatos()
        TARGET = "Pago_atiempo"

        X = df.drop(columns=[TARGET])
        y_true = df[TARGET]

        # Preprocesamiento
        preprocessor = ft_engineering(X)
        X_processed = preprocessor.fit_transform(X)

        dmatrix = xgb.DMatrix(X_processed)
        probas = model.predict(dmatrix)

        y_pred = [1 if p >= 0.5 else 0 for p in probas]

        accuracy = sum(y_pred == y_true) / len(y_true)

        return {
            "accuracy": round(accuracy, 4),
            "total_registros": len(y_true),
            "casos_no_pago": int(sum(pd.Series(y_pred) == 0))
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

@app.get("/monitor")
def monitor():
    """
    Ejecuta detección de Data Drift usando Evidently.
    """
    try:
        df = cargarDatos()
        TARGET = "Pago_atiempo"

        X = df.drop(columns=[TARGET])
        X_ref = X.sample(frac=0.8, random_state=42)
        X_cur = X.drop(X_ref.index)

        report = generate_drift_report(X_ref, X_cur)

        drift_dict = report.as_dict()
        dataset_drift = drift_dict["metrics"][0]["result"]["dataset_drift"]

        return {
            "dataset_drift_detected": dataset_drift,
            "n_features": len(X.columns)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

if __name__ == "__main__":
    uvicorn.run(
        "model_deploy:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
