import argparse
import pandas as pd
from sklearn.metrics import mean_squared_error
import joblib

def evaluate_model(model_type):
    # Load preprocessed data
    X_test = pd.read_csv('X_test.csv')
    y_test = pd.read_csv('y_test.csv')

    # Load model
    model_name = f"model_{model_type}.joblib"
    model = joblib.load(model_name)

    # Predict and evaluate
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Model {model_type} - Mean Squared Error: {mse}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Choose model: A, B, C, or D")
    args = parser.parse_args()
    evaluate_model(args.model)
