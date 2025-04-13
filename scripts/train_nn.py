# scripts/train_nn.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scripts.preprocess import load_and_preprocess_data

# Neural network definition
from scripts.models import SimpleNN

def train_nn():
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data()

    # Convert to torch tensors
    X_train_tensor = torch.tensor(X_train.toarray() if hasattr(X_train, "toarray") else X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.toarray() if hasattr(X_test, "toarray") else X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

    # Create DataLoader
    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

    # Initialize model
    model = SimpleNN(X_train_tensor.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(100):
        model.train()
        running_loss = 0.0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_dl):.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor)
        mse = criterion(predictions, y_test_tensor).item()
        r2 = 1 - (mse / torch.var(y_test_tensor).item())
        print(f"Neural Network MSE: {mse:.4f}")
        print(f"Neural Network R²: {r2:.4f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/nn_model.pth")
    print("Neural network model saved to models/nn_model.pth")

if __name__ == "__main__":
    train_nn()
