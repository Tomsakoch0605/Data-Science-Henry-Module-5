import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def prepare_data(df):

    data = df.copy()

    target = "Pago_atiempo"

    data = data.drop(columns=["fecha_prestamo"])

    data["puntaje_datacredito"] = data["puntaje_datacredito"].fillna(
        data["puntaje_datacredito"].median()
    )

    data["promedio_ingresos_datacredito"] = data[
        "promedio_ingresos_datacredito"
    ].fillna(data["promedio_ingresos_datacredito"].median())

    data["tendencia_ingresos"] = data["tendencia_ingresos"].fillna("Desconocido")

    X = data.drop(columns=[target])
    y = data[target]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, preprocessor