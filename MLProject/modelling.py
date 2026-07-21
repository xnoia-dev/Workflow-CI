"""
Kriteria 3 - Workflow CI: melatih ulang model klasifikasi rice variety lewat
MLflow Project, sehingga bisa dipanggil otomatis oleh GitHub Actions.

Dijalankan lewat MLflow Project (bukan langsung), contoh:
    mlflow run MLProject --env-manager=local
    mlflow run MLProject --env-manager=local -P n_estimators=200

Model hasil latihan disimpan ke folder lokal (--model_output) supaya langkah
berikutnya di CI bisa membangun Docker image dengan `mlflow models build-docker`
tanpa perlu menebak run id.
"""

import argparse
import json
import os
import shutil

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_COL = "is_osmancik"


def parse_args():
    parser = argparse.ArgumentParser(description="Training rice variety classifier")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--min_samples_split", type=int, default=5)
    parser.add_argument("--data_dir", type=str, default="rice_preprocessing")
    parser.add_argument("--model_output", type=str, default="artifacts/model")
    return parser.parse_args()


def resolve(path):
    """Path relatif selalu dihitung dari folder MLProject, bukan cwd pemanggil."""
    return path if os.path.isabs(path) else os.path.join(BASE_DIR, path)


def load_train_test(data_dir):
    data_dir = resolve(data_dir)
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    return X_train, X_test, y_train, y_test


def main():
    args = parse_args()

    X_train, X_test, y_train, y_test = load_train_test(args.data_dir)

    # Saat dipanggil lewat `mlflow run`, MLflow sudah membuat run aktif dan
    # start_run() di sini akan menempel ke run tersebut, bukan membuat run baru.
    with mlflow.start_run(run_name="random_forest_ci"):
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("min_samples_split", args.min_samples_split)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))

        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            random_state=42,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        metrics = {
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_precision": precision_score(y_test, y_pred, zero_division=0),
            "test_recall": recall_score(y_test, y_pred, zero_division=0),
            "test_f1": f1_score(y_test, y_pred, zero_division=0),
            "train_accuracy": accuracy_score(y_train, model.predict(X_train)),
        }

        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_train.head(5),
        )

        # Salinan lokal model + ringkasan metrik, dipakai langkah CI berikutnya
        # (upload artefak dan build Docker image).
        model_output = resolve(args.model_output)
        if os.path.exists(model_output):
            shutil.rmtree(model_output)
        os.makedirs(os.path.dirname(model_output), exist_ok=True)
        mlflow.sklearn.save_model(sk_model=model, path=model_output)

        metrics_path = os.path.join(os.path.dirname(model_output), "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")
        print(f"Model tersimpan di: {model_output}")


if __name__ == "__main__":
    main()
