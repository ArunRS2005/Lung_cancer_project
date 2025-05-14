import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report
from preprocess import train_test_data


def train_dl_model(csv_path, model_path="dl_model.h5"):
    X_train, X_test, y_train, y_test = train_test_data(csv_path)

    model = Sequential([
        Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=16, callbacks=[early_stop], verbose=0)

    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    acc = accuracy_score(y_test, y_pred)
    print("[DL] Accuracy:", acc)
    print(classification_report(y_test, y_pred))

    model.save(model_path)
    print(f"[DL] Model saved to {model_path}")


if __name__ == "__main__":
    train_dl_model("C:\\Users\\arunr\\OneDrive\\Desktop\\lung_cancer_prediction\\model_utils\\lungcancer.csv")
