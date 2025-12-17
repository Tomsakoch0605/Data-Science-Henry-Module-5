from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, auc_score
from xgboost import XGBClassifier


def train_and_evaluate_models(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor
):

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=150,
            random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False
        )
    }

    results = {}

    for model_name, model in models.items():

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ]
        )

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        results[model_name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred),
            "model": pipeline
        }

    return results