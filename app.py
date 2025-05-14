from flask import Flask, render_template, request
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from model_utils.preprocess import load_and_preprocess
import os

app = Flask(__name__)

# Load models
ml_model = joblib.load("C:\\Users\\arunr\\OneDrive\\Desktop\\lung_cancer_prediction\\models\\ml_model.pkl")
dl_model = load_model("C:\\Users\\arunr\\OneDrive\\Desktop\\lung_cancer_prediction\\models\\dl_model.h5")
qml_model = joblib.load("C:\\Users\\arunr\\OneDrive\\Desktop\\lung_cancer_prediction\\models\\qml_model.pkl")
scaler_path = "C:\\Users\\arunr\\OneDrive\\Desktop\\lung_cancer_prediction\\models\\scaler.pkl"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Extract form data
        try:
            input_data = [
                int(request.form.get("AGE")),
                int(request.form.get("GENDER")),
                int(request.form.get("SMOKING")),
                int(request.form.get("YELLOW_FINGERS")),
                int(request.form.get("ANXIETY")),
                int(request.form.get("PEER_PRESSURE")),
                int(request.form.get("CHRONIC DISEASE")),
                int(request.form.get("FATIGUE")),
                int(request.form.get("ALLERGY")),
                int(request.form.get("WHEEZING")),
                int(request.form.get("ALCOHOL CONSUMING")),
                int(request.form.get("COUGHING")),
                int(request.form.get("SHORTNESS OF BREATH")),
                int(request.form.get("SWALLOWING DIFFICULTY")),
                int(request.form.get("CHEST PAIN"))
            ]
        except:
            return "Invalid input. Please enter valid integers."

        X = np.array([input_data])
        X_scaled = joblib.load(scaler_path).transform(X)

        # ML prediction
        ml_pred = ml_model.predict(X_scaled)[0]

        # DL prediction
        dl_pred = (dl_model.predict(X_scaled)[0][0] > 0.5).astype("int")

        # QML prediction (only use first 2 features)
        from model_utils.train_qml import quantum_features

        qml_input = quantum_features(X_scaled[:, :2])
        qml_pred = qml_model.predict(qml_input)[0]

        result = {
            "ML Prediction": "YES" if ml_pred else "NO",
            "DL Prediction": "YES" if dl_pred else "NO",
            "QML Prediction": "YES" if qml_pred else "NO"
        }

        return render_template("index.html", result=result)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
