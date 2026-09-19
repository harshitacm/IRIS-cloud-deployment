from pathlib import Path
import pickle

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def main():
    iris = load_iris()

    X = pd.DataFrame(
        iris.data,
        columns=[
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
        ],
    )
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Model Training Completed")
    print(f"Model Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    model_path = Path(__file__).resolve().parent / "iris_model.pkl"
    with model_path.open("wb") as file:
        pickle.dump(
            {
                "model": model,
                "target_names": iris.target_names,
                "accuracy": float(accuracy),
            },
            file,
        )

    print(f"Model saved successfully as {model_path.name}")


if __name__ == "__main__":
    main()
