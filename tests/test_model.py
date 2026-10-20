"""Verify gradients, learning behavior, and the public input contract."""

import unittest

import numpy as np

from neural_network import TwoLayerNetwork


class TwoLayerNetworkTests(unittest.TestCase):
    def setUp(self):
        self.features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        self.targets = np.array([0, 1, 1, 0])
        self.model = TwoLayerNetwork(2, seed=42)

    def test_learns_xor(self):
        initial_loss = self.model.loss(self.features, self.targets)
        history = self.model.fit(self.features, self.targets)

        self.assertEqual(len(history), 5000)
        self.assertEqual(history[0], initial_loss)
        self.assertLess(self.model.loss(self.features, self.targets), 0.02)
        np.testing.assert_array_equal(self.model.predict(self.features), self.targets)

    def test_probabilities_and_predictions_have_one_value_per_sample(self):
        probabilities = self.model.predict_proba(self.features)
        predictions = self.model.predict(self.features)

        self.assertEqual(probabilities.shape, (4,))
        self.assertEqual(predictions.shape, (4,))
        self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))
        np.testing.assert_array_equal(predictions, (probabilities >= 0.5).astype(int))

    def test_seed_reproduces_initial_predictions(self):
        same_seed = TwoLayerNetwork(2, seed=42)
        other_seed = TwoLayerNetwork(2, seed=7)
        expected = self.model.predict_proba(self.features)

        np.testing.assert_array_equal(same_seed.predict_proba(self.features), expected)
        self.assertFalse(np.array_equal(other_seed.predict_proba(self.features), expected))

    def test_column_targets_and_continued_training(self):
        self.model.fit(self.features, self.targets.reshape(-1, 1), epochs=10)
        previous_loss = self.model.loss(self.features, self.targets)
        history = self.model.fit(self.features, self.targets, epochs=10)

        self.assertEqual(history[0], previous_loss)
        self.assertLess(self.model.loss(self.features, self.targets), previous_loss)

    def test_backpropagation_matches_numerical_gradients(self):
        model = TwoLayerNetwork(2, hidden_size=3, seed=7)
        parameters = (model._weights1, model._bias1, model._weights2, model._bias2)
        originals = [parameter.copy() for parameter in parameters]
        numerical_gradients = []
        epsilon = 1e-5

        for parameter in parameters:
            gradient = np.empty_like(parameter)
            for index in np.ndindex(parameter.shape):
                original = parameter[index]
                parameter[index] = original + epsilon
                plus = model.loss(self.features, self.targets)
                parameter[index] = original - epsilon
                minus = model.loss(self.features, self.targets)
                parameter[index] = original
                gradient[index] = (plus - minus) / (2 * epsilon)
            numerical_gradients.append(gradient)

        learning_rate = 0.1
        model.fit(self.features, self.targets, epochs=1, learning_rate=learning_rate)
        for original, parameter, expected in zip(
            originals, parameters, numerical_gradients, strict=True
        ):
            np.testing.assert_allclose(
                (original - parameter) / learning_rate, expected, rtol=1e-5, atol=1e-8
            )

    def test_extreme_logits_remain_finite(self):
        model = TwoLayerNetwork(2, hidden_size=1, seed=42)
        model._weights1.fill(1)
        model._weights2.fill(1000)
        features = np.array([[1000, 1000], [-1000, -1000]])

        with np.errstate(over="raise", invalid="raise", divide="raise"):
            np.testing.assert_array_equal(model.predict_proba(features), [1, 0])
            self.assertAlmostEqual(model.loss(features, [0, 1]), 1000)

    def test_rejects_invalid_layer_sizes(self):
        for size in (0, -1, 1.5, True):
            with self.subTest(size=size):
                with self.assertRaisesRegex(ValueError, "input_size"):
                    TwoLayerNetwork(size)
                with self.assertRaisesRegex(ValueError, "hidden_size"):
                    TwoLayerNetwork(2, hidden_size=size)

    def test_rejects_invalid_features(self):
        for features in ([], [0, 1], [[0]], [[0, 1, 2]], [[np.nan, 0]], [[np.inf, 0]]):
            with self.subTest(features=features):
                with self.assertRaisesRegex(ValueError, "features"):
                    self.model.predict_proba(features)

    def test_rejects_invalid_targets(self):
        for targets in ([], [0], [[0, 1, 1, 0]], [0, 1, 0.5, 0], [0, 1, np.nan, 0]):
            with self.subTest(targets=targets):
                with self.assertRaisesRegex(ValueError, "targets"):
                    self.model.fit(self.features, targets, epochs=1)

    def test_rejects_invalid_training_options(self):
        for epochs in (0, -1, 1.5, True):
            with self.subTest(epochs=epochs):
                with self.assertRaisesRegex(ValueError, "epochs"):
                    self.model.fit(self.features, self.targets, epochs=epochs)
        for learning_rate in (0, -1, np.nan, np.inf, True, "fast"):
            with self.subTest(learning_rate=learning_rate):
                with self.assertRaisesRegex(ValueError, "learning_rate"):
                    self.model.fit(self.features, self.targets, learning_rate=learning_rate)


if __name__ == "__main__":
    unittest.main()
