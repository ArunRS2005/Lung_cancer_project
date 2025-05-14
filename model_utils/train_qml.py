import pennylane as qml
from pennylane import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
import joblib
from model_utils.preprocess import train_test_data


def quantum_features(x):
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit(inputs):
        qml.AngleEmbedding(inputs[:2], wires=[0, 1])
        qml.CNOT(wires=[0, 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(2)]

    return np.array([circuit(sample) for sample in x])


def train_qml_model(csv_path, model_path="qml_model.pkl"):
    X_train, X_test, y_train, y_test = train_test_data(csv_path)

    # Reduce features to 2 for simplicity (required by the quantum embedding above)
    X_train_small = X_train[:, :2]
    X_test_small = X_test[:, :2]

    X_train_encoded = quantum_features(X_train_small)
    X_test_encoded = quantum_features(X_test_small)

    clf = LogisticRegression()
    clf.fit(X_train_encoded, y_train)

    y_pred = clf.predict(X_test_encoded)
    acc = accuracy_score(y_test, y_pred)
    print("[QML] Accuracy:", acc)
    print(classification_report(y_test, y_pred))

    joblib.dump(clf, model_path)
    print(f"[QML] Model saved to {model_path}")


if __name__ == "__main__":
    train_qml_model("C:\\Users\\arunr\\OneDrive\\Desktop\\lung_cancer_prediction\\model_utils\\lungcancer.csv")
