import joblib
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scripts.models import SimpleNN
from scripts.preprocess import load_and_preprocess_data

def evaluate_models():
    # First, we will load the data data and preprocessor
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data()

    X_test_dense = X_test.toarray() if hasattr(X_test, "toarray") else X_test

    # We will have two outputs. 
    # One output will come from the evaliation of a random forest
    # and the second output will come from the evaluation of a neural network

    # === Load and evaluate Random Forest ===
    rf_model = joblib.load("models/rf_model.pkl")
    y_pred_rf = rf_model.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)

    print("Random Forest Results:")
    print(f"  MSE: {mse_rf:.4f}")
    print(f"  R² : {r2_rf:.4f}")
    print("  Interpretation: The random forest model is predicting burned area with an average "
      f"error of about {mse_rf**0.5:.2f} hectares.\n"
      f"  It explains roughly {r2_rf * 100:.0f}% of the variation in the fire outcomes.\n")


    # === Load and evaluate Neural Network ===
    nn_model = SimpleNN(input_dim=X_test_dense.shape[1])
    nn_model.load_state_dict(torch.load("models/nn_model.pth"))
    nn_model.eval()

    X_test_tensor = torch.tensor(X_test_dense, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

    with torch.no_grad():
        y_pred_nn = nn_model(X_test_tensor).numpy().flatten()

    mse_nn = mean_squared_error(y_test, y_pred_nn)
    r2_nn = r2_score(y_test, y_pred_nn)

    print("Neural Network Results:")
    print(f"  MSE: {mse_nn:.4f}")
    print(f"  R² : {r2_nn:.4f}")
    print("  Interpretation: On average, the neural network's predictions are off by about "
      f"{mse_nn**0.5:.2f} hectares.\n"
      "  The R² score shows how well the model explains variation in burned area — "
      f"{r2_nn:.2f} means it's capturing about {r2_nn * 100:.0f}% of the variation.\n")


if __name__ == "__main__":
    evaluate_models()
