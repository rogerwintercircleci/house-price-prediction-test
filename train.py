import argparse
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import joblib

def train_model(model_type):
    # Load preprocessed data
    X_train = pd.read_csv('X_train.csv')
    y_train = pd.read_csv('y_train.csv')

    # Choose model
    if model_type == "A":
        model = LinearRegression()
        model_name = "model_A.joblib"
    elif model_type == "B":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model_name = "model_B.joblib"
    elif model_type == "C":
        model = SVR()
        model_name = "model_C.joblib"
    elif model_type == "D":
        model = KNeighborsRegressor(n_neighbors=5)
        model_name = "model_D.joblib"
    else:
        raise ValueError("Invalid model type. Choose 'A', 'B', 'C' or 'D'.")

    # Train model
    model.fit(X_train, y_train)

    # Save trained model
    joblib.dump(model, model_name)
    print(f"{model_type} trained and saved as {model_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Choose model: A, B, C, or D")
    args = parser.parse_args()
    train_model(args.model)
