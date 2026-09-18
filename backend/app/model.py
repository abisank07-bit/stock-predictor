"""
model.py
LSTM model for stock price prediction, built with PyTorch.
"""

import torch
import torch.nn as nn
import numpy as np
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)


class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # take the output of the last timestep
        return self.fc(out)


def train_model(X, y, epochs=25, lr=0.001, batch_size=32):
    """Train an LSTM on prepared sequences and return the trained model + loss history."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)

    model = StockLSTM().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = []
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(loader)
        history.append(avg_loss)

    return model, history


def save_model(model, ticker: str):
    path = os.path.join(MODEL_DIR, f"{ticker}.pt")
    torch.save(model.state_dict(), path)
    return path


def load_model(ticker: str):
    path = os.path.join(MODEL_DIR, f"{ticker}.pt")
    if not os.path.exists(path):
        return None
    model = StockLSTM()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def predict_next_price(model, last_sequence, min_val, max_val):
    """
    last_sequence: the most recent `lookback` scaled closing prices, shape (lookback, 1)
    Returns the predicted next closing price in real (unscaled) terms.
    """
    model.eval()
    with torch.no_grad():
        x = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0)  # [1, lookback, 1]
        scaled_pred = model(x).item()

    # de-normalize back to real price
    real_price = scaled_pred * (max_val - min_val) + min_val
    return real_price


def confidence_score(history):
    """
    Simple confidence proxy: lower final training loss -> higher confidence.
    Returns a 0-100 score. Not a statistical guarantee, just a rough UI signal.
    """
    final_loss = history[-1]
    score = max(0, min(100, 100 * (1 - final_loss)))
    return round(score, 1)
