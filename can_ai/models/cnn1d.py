from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from torch.utils.data import DataLoader, TensorDataset


class _FeatureCNN1D(nn.Module):
    def __init__(self, n_features: int, n_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class CNN1DClassifier(ClassifierMixin, BaseEstimator):
    """A compact 1D-CNN classifier over each engineered CAN window feature vector."""

    def __init__(
        self,
        epochs: int = 8,
        batch_size: int = 256,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        dropout: float = 0.2,
        random_state: int = 42,
        device: str = "auto",
        verbose: bool = False,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    def _device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=False)
        X = X.astype(np.float32)
        y = y.astype(np.int64)

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        class_to_index = {label: idx for idx, label in enumerate(self.classes_)}
        y_index = np.array([class_to_index[label] for label in y], dtype=np.int64)

        device = self._device()
        model = _FeatureCNN1D(self.n_features_in_, len(self.classes_), self.dropout).to(device)

        counts = np.bincount(y_index, minlength=len(self.classes_)).astype(np.float32)
        weights = counts.sum() / (counts + 1e-6)
        weights = weights / weights.mean()
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        X_tensor = torch.tensor(X[:, None, :], dtype=torch.float32)
        y_tensor = torch.tensor(y_index, dtype=torch.long)
        loader = DataLoader(
            TensorDataset(X_tensor, y_tensor),
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.random_state),
        )

        model.train()
        for epoch in range(self.epochs):
            total = 0.0
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()
                total += float(loss.item())
            if self.verbose:
                print(f"1D CNN epoch {epoch + 1}/{self.epochs} loss={total / max(1, len(loader)):.4f}")

        self.model_ = model.to("cpu")
        self.model_.eval()
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "model_")
        X = check_array(X, accept_sparse=False).astype(np.float32)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")

        probs = []
        tensor = torch.tensor(X[:, None, :], dtype=torch.float32)
        loader = DataLoader(TensorDataset(tensor), batch_size=self.batch_size, shuffle=False)

        self.model_.eval()
        with torch.no_grad():
            for (xb,) in loader:
                logits = self.model_(xb)
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.vstack(probs)

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
