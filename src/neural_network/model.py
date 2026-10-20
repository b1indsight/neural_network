"""Forward propagation and gradient descent for binary classification."""

from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray


class TwoLayerNetwork:
    """A fully connected tanh hidden layer followed by a sigmoid output.

    Args:
        input_size: Number of features per sample.
        hidden_size: Number of neurons in the hidden layer.
        seed: Optional seed for reproducible initial weights.
    """

    def __init__(self, input_size: int, hidden_size: int = 8, *, seed: int | None = None):
        for name, value in (("input_size", input_size), ("hidden_size", hidden_size)):
            self._positive_integer(name, value)

        self.input_size = input_size
        self.hidden_size = hidden_size
        rng = np.random.default_rng(seed)
        self._weights1 = rng.normal(0, np.sqrt(1 / input_size), (input_size, hidden_size))
        self._bias1 = np.zeros(hidden_size)
        self._weights2 = rng.normal(0, np.sqrt(1 / hidden_size), (hidden_size, 1))
        self._bias2 = np.zeros(1)

    def predict_proba(self, features: ArrayLike) -> NDArray[np.float64]:
        """Return the probability of class 1 for each sample, with shape (n,)."""
        _, logits = self._forward(self._features(features))
        return self._sigmoid(logits).ravel()

    def predict(self, features: ArrayLike) -> NDArray[np.int64]:
        """Return binary labels using a probability threshold of 0.5."""
        return (self.predict_proba(features) >= 0.5).astype(np.int64)

    def loss(self, features: ArrayLike, targets: ArrayLike) -> float:
        """Return mean binary cross-entropy, computed directly from logits."""
        inputs = self._features(features)
        labels = self._targets(targets, inputs.shape[0])
        _, logits = self._forward(inputs)
        return self._cross_entropy(logits, labels)

    def fit(
        self,
        features: ArrayLike,
        targets: ArrayLike,
        *,
        epochs: int = 5000,
        learning_rate: float = 0.1,
    ) -> list[float]:
        """Train in place; return the loss before each full-batch update."""
        inputs = self._features(features)
        labels = self._targets(targets, inputs.shape[0])
        self._positive_integer("epochs", epochs)
        if (
            isinstance(learning_rate, bool)
            or not isinstance(learning_rate, Real)
            or not np.isfinite(learning_rate)
            or learning_rate <= 0
        ):
            raise ValueError("learning_rate must be a positive finite number")

        history = []
        for _ in range(epochs):
            hidden, logits = self._forward(inputs)
            history.append(self._cross_entropy(logits, labels))

            output_gradient = (self._sigmoid(logits) - labels) / inputs.shape[0]
            hidden_gradient = (output_gradient @ self._weights2.T) * (1 - hidden**2)
            # Compute the hidden gradient before updating the output weights.
            self._weights2 -= learning_rate * (hidden.T @ output_gradient)
            self._bias2 -= learning_rate * output_gradient.sum(axis=0)
            self._weights1 -= learning_rate * (inputs.T @ hidden_gradient)
            self._bias1 -= learning_rate * hidden_gradient.sum(axis=0)

        return history

    def _forward(
        self, features: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        hidden = np.tanh(features @ self._weights1 + self._bias1)
        return hidden, hidden @ self._weights2 + self._bias2

    @staticmethod
    def _sigmoid(logits: NDArray[np.float64]) -> NDArray[np.float64]:
        # logaddexp avoids overflow for very large positive or negative logits.
        return np.exp(-np.logaddexp(0.0, -logits))

    @staticmethod
    def _cross_entropy(logits: NDArray[np.float64], targets: NDArray[np.float64]) -> float:
        return float(np.mean(np.logaddexp(0.0, logits) - targets * logits))

    def _features(self, features: ArrayLike) -> NDArray[np.float64]:
        inputs = np.asarray(features, dtype=np.float64)
        if inputs.ndim != 2 or inputs.shape[0] == 0 or inputs.shape[1] != self.input_size:
            raise ValueError(
                f"features must have shape (n_samples, {self.input_size}), n_samples > 0"
            )
        if not np.isfinite(inputs).all():
            raise ValueError("features must contain only finite values")
        return inputs

    @staticmethod
    def _targets(targets: ArrayLike, sample_count: int) -> NDArray[np.float64]:
        labels = np.asarray(targets, dtype=np.float64)
        if labels.shape == (sample_count,):
            labels = labels.reshape(-1, 1)
        if labels.shape != (sample_count, 1):
            raise ValueError("targets must have shape (n_samples,) or (n_samples, 1)")
        if not np.isin(labels, (0, 1)).all():
            raise ValueError("targets must contain only binary labels 0 or 1")
        return labels

    @staticmethod
    def _positive_integer(name: str, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
