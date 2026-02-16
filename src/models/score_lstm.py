"""LSTM Score Predictor — Over-by-over score trajectory prediction.

Module B2: Uses sequence modelling to predict scores at each over.
This is the deep learning differentiator of the project.

Architecture:
    Input (seq_len=20, features=10) → LSTM(128) → LSTM(64) → Dense(32) → Dense(1)

Usage:
    from src.models.score_lstm import ScoreLSTM, OverByOverDataset

    dataset = OverByOverDataset(matches_data)
    model = ScoreLSTM()
    model = train_score_lstm(model, train_loader, val_loader)
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional

from src.config import settings
from src.utils.logger import logger


class OverByOverDataset(Dataset):
    """Dataset for over-by-over score prediction.

    Each sample is a sequence of per-over features for one innings,
    with the target being the final innings total.

    Per-over features (10 dimensions):
    [runs_this_over, wickets_this_over, extras, dot_balls,
     boundaries, current_run_rate, cumulative_score,
     wickets_fallen_total, phase_encoded, over_number]
    """

    def __init__(self, over_sequences: list[np.ndarray], targets: list[float],
                 max_overs: int = 20):
        self.max_overs = max_overs
        self.sequences = []
        self.targets = []

        for seq, target in zip(over_sequences, targets):
            # Pad if fewer than 20 overs
            n_features = seq.shape[1] if len(seq.shape) > 1 else 10
            padded = np.zeros((max_overs, n_features))
            actual_overs = min(len(seq), max_overs)
            padded[:actual_overs] = seq[:actual_overs]
            self.sequences.append(padded)
            self.targets.append(target)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor([self.targets[idx]]),
        )


class ScoreLSTM(nn.Module):
    """Bidirectional LSTM for cricket score prediction."""

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len=20, features=10).

        Returns:
            Predicted score tensor of shape (batch, 1).
        """
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use the final hidden state
        final_hidden = h_n[-1]  # (batch, hidden_dim)
        return self.fc(final_hidden)


def train_score_lstm(
    model: ScoreLSTM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 15,
    save_path: Optional[str] = None,
) -> ScoreLSTM:
    """Train the LSTM score predictor with early stopping.

    Args:
        model: ScoreLSTM instance.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        epochs: Maximum training epochs.
        lr: Learning rate.
        patience: Early stopping patience.
        save_path: Path to save best model checkpoint.

    Returns:
        Trained model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info(f"Training on device: {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=patience // 3, factor=0.5
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        val_mae = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                pred = model(X_val)
                val_loss += criterion(pred, y_val).item()
                val_mae += torch.abs(pred - y_val).mean().item()

        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch + 1}/{epochs}: "
                f"Train Loss={train_loss:.4f}, "
                f"Val Loss={val_loss:.4f}, "
                f"Val MAE={val_mae:.1f} runs"
            )

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    # Load best model
    if save_path:
        model.load_state_dict(torch.load(save_path, weights_only=True))

    logger.info(f"Training complete. Best Val Loss: {best_val_loss:.4f}")
    return model
