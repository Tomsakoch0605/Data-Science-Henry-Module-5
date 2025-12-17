from ft_engineering import ft_engineering
from carga_datos import cargarDatos

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split,
    KFold,
    ShuffleSplit,
    cross_val_score,
    learning_curve
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# =====================================================
# 1. Carga de datos
# =====================================================
df = cargarDatos()

TARGET = "Pago_atiempo"
X = df.drop(columns=[TARGET])
y = df[TARGET]

# =====================================================
# 2. Función de métricas
# =====================================================
def summarize_classification(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=0),
        "recall": recall_score(y_true, y_pred, pos_label=0),
        "f1_score": f1_score(y_true, y_pred, pos_label=0),
        "roc_auc": roc_auc_score(y_true, y_pred),
        "casosNoPagoAtiempo": np.sum(y_pred == 0)
    }

# =====================================================
# 3. Función build_model
# =====================================================
def build_model(model, data_params, test_frac=0.2):
    dataset = data_params["dataset"]
    x_cols = data_params["names_of_x_cols"]
    y_col = data_params["name_of_y_col"]

    X = dataset[x_cols]
    y = dataset[y_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_frac,
        random_state=42,
        stratify=y
    )

    preprocessor = ft_engineering(X_train)

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    pipeline = Pipeline(
        steps=[("model", model)]
    )

    pipeline.fit(X_train, y_train)

    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    train_metrics = summarize_classification(y_train, y_pred_train)
    test_metrics = summarize_classification(y_test, y_pred_test)

    # -------------------------------
    # Validación cruzada
    # -------------------------------
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=kfold,
        scoring="recall"
    )

    # -------------------------------
    # Curva de aprendizaje
    # -------------------------------
    train_sizes, train_scores, test_scores = learning_curve(
        pipeline,
        X_train,
        y_train,
        train_sizes=np.linspace(0.1, 1.0, 5),
        cv=ShuffleSplit(n_splits=10, test_size=0.2, random_state=42),
        scoring="recall",
        n_jobs=-1
    )

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_scores.mean(axis=1), label="Train")
    plt.plot(train_sizes, test_scores.mean(axis=1), label="CV")
    plt.title(f"Learning Curve - {model.__class__.__name__}")
    plt.xlabel("Training samples")
    plt.ylabel("Recall (No pago a tiempo)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        "train": train_metrics,
        "test": test_metrics,
        "cv_recall_mean": cv_scores.mean()
    }

# =====================================================
# 4. Modelos a entrenar
# =====================================================
models = {
    "LogisticRegression": LogisticRegression(
        solver="liblinear",
        class_weight="balanced"
    ),
    "LinearSVC": LinearSVC(
        class_weight="balanced",
        max_iter=1000
    ),
    "DecisionTree": DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=10,
        class_weight="balanced"
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=150,
        max_depth=7,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
}

data_params = {
    "dataset": df,
    "names_of_x_cols": X.columns,
    "name_of_y_col": TARGET
}

# =====================================================
# 5. Entrenamiento
# =====================================================
results = {}

for name, model in models.items():
    print(f"\nEntrenando modelo: {name}")
    results[name] = build_model(model, data_params)

# =====================================================
# 6. Resultados comparativos
# =====================================================
records = []

for model_name, model_results in results.items():
    for dataset, metrics in model_results.items():
        if isinstance(metrics, dict):
            for metric, value in metrics.items():
                records.append({
                    "Model": model_name,
                    "Dataset": dataset,
                    "Metric": metric,
                    "Score": value
                })

results_df = pd.DataFrame(records)

# =====================================================
# 7. Visualización comparativa
# =====================================================
plt.figure(figsize=(12, 6))
sns.barplot(
    data=results_df[results_df["Metric"] == "recall"],
    x="Model",
    y="Score",
    hue="Dataset",
    palette="cividis"
)
plt.title("Comparación de modelos – Recall No Pago a Tiempo")
plt.ylabel("Recall")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()