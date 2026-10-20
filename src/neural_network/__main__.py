"""Run a reproducible XOR training example with python -m neural_network."""

import numpy as np

from .model import TwoLayerNetwork


def main() -> None:
    features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    targets = np.array([0, 1, 1, 0])
    model = TwoLayerNetwork(input_size=2, hidden_size=8, seed=42)
    losses = model.fit(features, targets)

    print(f"Loss: {losses[0]:.6f} -> {model.loss(features, targets):.6f}")
    for inputs, probability in zip(features, model.predict_proba(features), strict=True):
        prediction = int(probability >= 0.5)
        print(f"{int(inputs[0])} XOR {int(inputs[1])} = {prediction} (p={probability:.4f})")


if __name__ == "__main__":
    main()
