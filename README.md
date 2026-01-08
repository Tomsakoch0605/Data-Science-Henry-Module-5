# 🏦 Predicción de Pago a Tiempo - MLOps Pipeline

Sistema de Machine Learning para predecir si un cliente pagará su crédito a tiempo, implementado con arquitectura MLOps completa incluyendo API REST, monitoreo de Data Drift y despliegue con Docker.

## 📋 Descripción del Proyecto

Este proyecto implementa un modelo XGBoost para clasificación binaria que predice la probabilidad de que un cliente cumpla con sus pagos de crédito. Incluye todo el ciclo de vida de un modelo en producción:

- **Análisis Exploratorio de Datos (EDA)**
- **Feature Engineering**
- **Entrenamiento del modelo**
- **API REST para predicciones**
- **Monitoreo en tiempo real con detección de Data Drift**
- **Contenerización con Docker**

## 🏗️ Arquitectura del Proyecto

```
proyecto/
│
├── Base_de_datos.xlsx          # Dataset original
├── carga_datos.py              # Módulo de carga de datos
├── comprension_eda.ipynb       # Notebook de análisis exploratorio
├── ft_engineering.py           # Pipeline de feature engineering
├── model_deploy.py             # API FastAPI para predicciones
├── model_monitoring.py         # Dashboard Streamlit de monitoreo
├── xgb_model.json              # Modelo XGBoost entrenado
├── Dockerfile                  # Configuración de contenedor
├── requirements.txt            # Dependencias del proyecto
└── README.md
```

## 📊 Dataset

El dataset contiene **10,763 registros** de solicitudes de crédito con las siguientes variables:

| Variable | Descripción |
|----------|-------------|
| `tipo_credito` | Categoría del crédito solicitado |
| `capital_prestado` | Monto del préstamo |
| `plazo_meses` | Duración del crédito en meses |
| `edad_cliente` | Edad del solicitante |
| `tipo_laboral` | Empleado / Independiente |
| `salario_cliente` | Ingreso mensual del cliente |
| `total_otros_prestamos` | Deudas previas del cliente |
| `puntaje_datacredito` | Score crediticio |
| `cant_creditosvigentes` | Número de créditos activos |
| `huella_consulta` | Consultas al buró de crédito |
| `saldo_total` | Saldo total de deudas |
| `saldo_mora_codeudor` | Mora del codeudor |
| `creditos_sectorCooperativo` | Créditos en cooperativas |
| `creditos_sectorReal` | Créditos en sector real |
| `tendencia_ingresos` | Creciente / Estable / Decreciente |
| **`Pago_atiempo`** | **Variable objetivo (1=Sí, 0=No)** |

## 🚀 Instalación

### Opción 1: Instalación Local

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/prediccion-pago-credito.git
cd prediccion-pago-credito

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt
```

### Opción 2: Docker

```bash
# Construir imagen
docker build -t prediccion-pago-api .

# Ejecutar contenedor
docker run -p 8000:8000 prediccion-pago-api
```

## 💻 Uso

### 1. API de Predicción

Iniciar el servidor:

```bash
uvicorn model_deploy:app --host 0.0.0.0 --port 8000 --reload
```

La API estará disponible en `http://localhost:8000`

#### Endpoints Disponibles

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/predict` | Predicción batch |
| GET | `/evaluation` | Métricas del modelo |
| GET | `/monitor` | Detección de Data Drift |

#### Ejemplo de Predicción

```python
import requests

url = "http://localhost:8000/predict"

payload = {
    "data": [
        {
            "bin_encoder_tipo_laboral": 1.0,
            "poly_ohe_tipo_credito_9": 0.0,
            "poly_ohe_tipo_credito_10": 0.0,
            "poly_ohe_tendencia_ingresos_Decreciente": 0.0,
            "poly_ohe_tendencia_ingresos_Estable": 1.0,
            "capital_prestado": 3500000.0,
            "plazo_meses": 12.0,
            "edad_cliente": 35.0,
            "salario_cliente": 4000000.0,
            "total_otros_prestamos": 1000000.0,
            "puntaje_datacredito": 750.0,
            "cant_creditosvigentes": 2.0,
            "huella_consulta": 3.0,
            "saldo_total": 50000.0,
            "saldo_mora_codeudor": 0.0,
            "creditos_sectorCooperativo": 0.0,
            "creditos_sectorReal": 1.0
        }
    ]
}

response = requests.post(url, json=payload)
print(response.json())
```

**Respuesta:**

```json
{
    "n_registros": 1,
    "predictions": [1],
    "probabilities": [0.847]
}
```

### 2. Dashboard de Monitoreo

Iniciar el dashboard:

```bash
streamlit run model_monitoring.py
```

El dashboard estará disponible en `http://localhost:8501`

**Características del Dashboard:**

- 📈 Visualización de distribución de predicciones
- 📊 Detección de Data Drift con Evidently
- 📉 Evolución temporal de predicciones
- 📋 Historial de predicciones (logs)
- ⬇️ Descarga de logs en CSV

### 3. Análisis Exploratorio

Abrir el notebook:

```bash
jupyter notebook comprension_eda.ipynb
```

## 🔧 Tecnologías Utilizadas

| Categoría | Tecnología |
|-----------|------------|
| **Lenguaje** | Python 3.10 |
| **ML Framework** | XGBoost |
| **API** | FastAPI + Uvicorn |
| **Monitoreo** | Streamlit + Evidently |
| **Visualización** | Plotly, Matplotlib, Seaborn |
| **Data** | Pandas, NumPy |
| **Contenedor** | Docker |
| **Validación** | Pydantic |

## 📦 Dependencias Principales

```txt
fastapi>=0.100.0
uvicorn>=0.23.0
xgboost>=2.0.0
pandas>=2.0.0
scikit-learn>=1.3.0
streamlit>=1.28.0
evidently>=0.4.0
plotly>=5.18.0
pydantic>=2.0.0
requests>=2.31.0
openpyxl>=3.1.0
```

## 📈 Métricas del Modelo

| Métrica | Valor |
|---------|-------|
| Accuracy | ~85% |
| Threshold | 0.5 |

## 🔄 Flujo de Trabajo MLOps

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Datos     │────▶│   Feature   │────▶│   Modelo    │
│   (Excel)   │     │ Engineering │     │  (XGBoost)  │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Dashboard  │◀────│  Monitoreo  │◀────│  API REST   │
│ (Streamlit) │     │ (Evidently) │     │  (FastAPI)  │
└─────────────┘     └─────────────┘     └─────────────┘
```

## 🐳 Despliegue con Docker

```bash
# Construir
docker build -t prediccion-pago-api .

# Ejecutar
docker run -d \
  --name api-prediccion \
  -p 8000:8000 \
  prediccion-pago-api

# Ver logs
docker logs -f api-prediccion

# Detener
docker stop api-prediccion
```

## 📝 Documentación de la API

Una vez iniciado el servidor, accede a la documentación interactiva:

- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el repositorio
2. Crea una rama (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -m 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 👤 Autor

**Alejandro Carrillo**

- GitHub: [@Tomsakoch0605](https://github.com/Tomsakoch0605)
- LinkedIn: [Alejandro Carrillo](https://www.linkedin.com/in/michel-alejandro-carrillo-vázquez-93658977)

---

⭐ Si este proyecto te fue útil, considera darle una estrella en GitHub.
