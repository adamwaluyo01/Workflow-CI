import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "namadataset_preprocessing"
ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)


def load_data():
    train_path = DATA_DIR / "train_preprocessed.csv"
    test_path = DATA_DIR / "test_preprocessed.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Train dataset not found: {train_path}")

    if not test_path.exists():
        raise FileNotFoundError(f"Test dataset not found: {test_path}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    if "churn" not in train.columns:
        raise ValueError("Column 'churn' not found in train_preprocessed.csv")

    if "churn" not in test.columns:
        raise ValueError("Column 'churn' not found in test_preprocessed.csv")

    X_train = train.drop(columns=["churn"])
    y_train = train["churn"]
    X_test = test.drop(columns=["churn"])
    y_test = test["churn"]

    return X_train, X_test, y_train, y_test


def save_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Churn", "Churn"],
    )
    display.plot(ax=ax, values_format="d")
    ax.set_title("Customer Churn Confusion Matrix - CI")

    output_path = ARTIFACT_DIR / "ci_confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)

    return output_path


def main():
    X_train, X_test, y_train, y_test = load_data()

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
    )

    active_run = mlflow.active_run()
    if active_run is not None:
        print(f"Using MLflow active run: {active_run.info.run_id}")
    else:
        print("No active run object detected, logging will use MLflow Project run from environment.")

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1_score": f1_score(y_test, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probas),
    }

    params = {
        "model_type": "RandomForestClassifier",
        "dataset": "customer_churn_preprocessed",
        "target": "churn",
        "n_estimators": 150,
        "max_depth": 8,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "class_weight": "balanced",
        "training_source": "github_actions_mlflow_project",
    }

    mlflow.log_params(params)
    mlflow.log_metrics(metrics)

    model_path = ARTIFACT_DIR / "customer_churn_model.joblib"
    joblib.dump(model, model_path)

    report_path = ARTIFACT_DIR / "classification_report.json"
    report_path.write_text(
        json.dumps(
            classification_report(
                y_test,
                preds,
                output_dict=True,
                zero_division=0,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    metrics_path = ARTIFACT_DIR / "ci_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    cm_path = save_confusion_matrix(y_test, preds)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
    )

    mlflow.log_artifact(model_path)
    mlflow.log_artifact(report_path)
    mlflow.log_artifact(metrics_path)
    mlflow.log_artifact(cm_path)

    print("CI training completed")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
