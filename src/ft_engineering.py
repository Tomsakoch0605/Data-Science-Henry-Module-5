import pandas as pd
from carga_datos import cargarDatos

from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


df = cargarDatos()

# Vista general del dataset (EDA básico)
df.info()
print(df.head())
print(df.describe(include="all"))


TARGET = "Pago_atiempo"

X = df.drop(TARGET, axis=1)
y = df[TARGET]


num_features = X.select_dtypes(include="number").columns
cat_features = X.select_dtypes(include="object").columns

print("Numeric features:")
print(num_features)

print("Categorical features:")
print(cat_features)


num_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="mean"))
    ]
)

cat_transformer = Pipeline(
    steps=[
        ("to_str", FunctionTransformer(lambda x: x.astype(str))),
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_features),
        ("cat", cat_transformer, cat_features)
    ]
)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


print("X_train procesado:")
print(X_train_processed)
print("Shape:", X_train_processed.shape)

print("X_test procesado:")
print(X_test_processed)
print("Shape:", X_test_processed.shape)


def ft_engineering(X_input=None):

    if X_input is None:
        X_input = df.drop(TARGET, axis=1)

    num_features = X_input.select_dtypes(include="number").columns
    cat_features = X_input.select_dtypes(include="object").columns

    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean"))
        ]
    )

    cat_transformer = Pipeline(
        steps=[
            ("to_str", FunctionTransformer(lambda x: x.astype(str))),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features),
            ("cat", cat_transformer, cat_features)
        ]
    )

    return preprocessor