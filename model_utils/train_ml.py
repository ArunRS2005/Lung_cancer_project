import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from preprocess import train_test_data


def train_ml_model(csv_path, model_path="ml_model.pkl"):
    X_train, X_test, y_train, y_test = train_test_data(csv_path)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("[ML] Accuracy:", acc)
    print(classification_report(y_test, y_pred))

    joblib.dump(clf, model_path)
    print(f"[ML] Model saved to {model_path}")


if __name__ == "__main__":
    train_ml_model("C:\\Users\\arunr\\OneDrive\\Desktop\\lung_cancer_prediction\\model_utils\\lungcancer.csv")
