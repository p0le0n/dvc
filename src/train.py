import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import json
import joblib

def train_model(input_path: str, metrics_path: str, model_path: str):
    # Загрузка данных
    df = pd.read_csv(input_path)
    X = df.drop(columns=["Species"])
    y = df["Species"]

    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Обучение модели
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Предсказания и метрики
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy}")

    # Сохранение метрик
    with open(metrics_path, "w") as f:
        json.dump({"accuracy": accuracy}, f)

    # Сохранение модели
    joblib.dump(model, model_path)
    print("Model and metrics saved.")

if __name__ == "__main__":
    train_model("data/Iris_processed.csv", "metrics.json", "models/iris_model.pkl")