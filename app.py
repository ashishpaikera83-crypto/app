import os
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
DATA_PATH = BASE_DIR / "jk.csv"
EXPECTED_FEATURES = ["Area", "Item", "Element", "Year", "Unit"]


def get_model_options(data_df: pd.DataFrame) -> dict[str, list[str]]:
    return {
        "Area": [str(value) for value in data_df["Area"].dropna().unique().tolist()],
        "Item": [str(value) for value in data_df["Item"].dropna().unique().tolist()],
        "Element": [str(value) for value in data_df["Element"].dropna().unique().tolist()],
        "Unit": [str(value) for value in data_df["Unit"].dropna().unique().tolist()],
    }


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
    app.config["TEMPLATES_AUTO_RELOAD"] = True

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    with MODEL_PATH.open("rb") as handle:
        model = pickle.load(handle)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset file not found: {DATA_PATH}")
    data_df = pd.read_csv(DATA_PATH)

    model_options = get_model_options(data_df)

    @app.get("/")
    def index() -> str:
        return render_template("index.html", model_options=model_options)

    @app.post("/predict")
    def predict():
        try:
            data = request.get_json(silent=True) or {}
            payload = {feature: data.get(feature, "") for feature in EXPECTED_FEATURES}
            for key in ["Area", "Item", "Element", "Unit"]:
                if not payload.get(key):
                    payload[key] = ""
            try:
                payload["Year"] = float(payload.get("Year", 2020))
            except (TypeError, ValueError):
                payload["Year"] = 2020
            if payload.get("Unit") in (None, ""):
                payload["Unit"] = "Head"
            frame = pd.DataFrame([payload])
            prediction = model.predict(frame)[0]
            probability = None
            if hasattr(model, "predict_proba"):
                probability = float(model.predict_proba(frame)[0].max())

            actual_value = None
            exact_match = data_df[
                (data_df["Area"] == payload["Area"]) &
                (data_df["Item"] == payload["Item"]) &
                (data_df["Element"] == payload["Element"]) &
                (data_df["Year"] == payload["Year"]) &
                (data_df["Unit"] == payload["Unit"])
            ]
            if not exact_match.empty:
                actual_value = exact_match.iloc[0]["Value"]

            response = {
                "success": True,
                "prediction": str(prediction),
                "probability": probability,
            }
            if actual_value is not None:
                response["actual"] = str(actual_value)
                response["diff"] = str(float(prediction) - float(actual_value))
            return jsonify(response)
        except Exception as exc:
            return jsonify({"success": False, "error": str(exc)}), 400

    @app.get("/health")
    def health() -> Any:
        return jsonify({"status": "ok", "model": MODEL_PATH.name})

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
