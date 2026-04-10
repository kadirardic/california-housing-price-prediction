
import joblib
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


def main():
    model = joblib.load("./artifacts/model.pkl")
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    sample = X_test.iloc[0:1]
    prediction = model.predict(sample)
    print(f"Prediction: {prediction}")


if __name__ == '__main__':
    main()
