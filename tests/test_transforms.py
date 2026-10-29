
import unittest
import numpy as np

from src.transforms import (
    geometric_adstock,
    hill_saturation,
    transform_channel,
    transform_channel_with_training_k
)


class TestMMMTransformations(unittest.TestCase):

    # Test 1: Existing Adstock carryover test
    def test_adstock_carryover(self):
        x = np.array([100, 0, 0], dtype=float)

        result = geometric_adstock(x, decay=0.5)

        np.testing.assert_allclose(
            result, [100, 50, 25]
        )

    # Test 2: Existing Saturation bounds test
    def test_saturation_bounds(self):
        x = np.array([0, 10, 1000], dtype=float)

        result = hill_saturation(
            x, half_sat=10
        )

        self.assertEqual(result[0], 0)
        self.assertAlmostEqual(result[1], 0.5)
        self.assertLess(result[2], 1)

    # Test 3: Invalid Adstock decay
    def test_invalid_decay(self):
        x = np.array([100, 50], dtype=float)

        with self.assertRaises(ValueError):
            geometric_adstock(x, decay=1.5)

    # Test 4: Combined transformation
    def test_combined_transformation(self):
        x = np.array([100, 50, 0], dtype=float)

        combined = transform_channel(
            x, decay=0.4, half_sat=100
        )

        separate = hill_saturation(
            geometric_adstock(x, decay=0.4),
            half_sat=100
        )

        np.testing.assert_allclose(
            combined, separate
        )

    # Test 5: Training-only parameter calculation
    def test_training_only_half_saturation(self):
        x = np.array(
            [100, 50, 75, 80, 900],
            dtype=float
        )

        result1, k1 = (
            transform_channel_with_training_k(
                x, decay=0.4, train_end=4
            )
        )

        modified_x = x.copy()
        modified_x[4] = 90000

        result2, k2 = (
            transform_channel_with_training_k(
                modified_x,
                decay=0.4,
                train_end=4
            )
        )

        self.assertAlmostEqual(k1, k2)

        np.testing.assert_allclose(
            result1[:4], result2[:4]
        )


if __name__ == "__main__":
    unittest.main()
