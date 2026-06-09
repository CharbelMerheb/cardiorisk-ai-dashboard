from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "Processed_Datasets"
ARTIFACT_DIR = BASE_DIR / "models" / "artifacts"
RANDOM_STATE = 42


DATASETS = {
    "chd": {
        "path": DATA_DIR / "framingham_clean.csv",
        "target": "TenYearCHD",
    },
    "heart": {
        "path": DATA_DIR / "heart_clean.csv",
        "target": "HeartDisease",
    },
    "stroke": {
        "path": DATA_DIR / "stroke_clean_ML.csv",
        "target": "stroke",
    },
}


def load_dataset(name: str) -> tuple[pd.DataFrame, pd.Series]:
    config = DATASETS[name]
    frame = pd.read_csv(config["path"])
    target = config["target"]
    frame = frame.dropna().copy()
    frame = frame.replace({True: 1, False: 0})
    x = frame.drop(columns=[target])
    y = frame[target].astype(int)
    return x, y


def train_classical_model(name: str) -> Pipeline | RandomForestClassifier:
    x, y = load_dataset(name)
    if name == "heart":
        model = RandomForestClassifier(
            n_estimators=240,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
        model.fit(x, y)
        return model

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2500,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(x, y)
    return model


def train_neural_network(name: str) -> Pipeline:
    x, y = load_dataset(name)
    hidden_layers = {
        "chd": (32, 16),
        "heart": (32, 16),
        "stroke": (48, 24),
    }[name]
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                MLPClassifier(
                    hidden_layer_sizes=hidden_layers,
                    activation="relu",
                    alpha=0.001,
                    early_stopping=True,
                    max_iter=600,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(x, y)
    return model


def evaluate_model(name: str, model: object) -> float:
    x, y = load_dataset(name)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    model.fit(x_train, y_train)
    return roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    for name in DATASETS:
        classical = train_classical_model(name)
        neural = train_neural_network(name)
        joblib.dump(classical, ARTIFACT_DIR / f"{name}_ml.joblib")
        joblib.dump(neural, ARTIFACT_DIR / f"{name}_nn.joblib")
        classical_auc = evaluate_model(name, train_classical_model(name))
        neural_auc = evaluate_model(name, train_neural_network(name))
        print(f"{name:>6} | classical AUC: {classical_auc:.3f} | neural AUC: {neural_auc:.3f}")


if __name__ == "__main__":
    main()
