import numpy as np
from src.transforms import geometric_adstock, hill_saturation


def test_adstock_carryover():
    out = geometric_adstock(np.array([100, 0, 0]), 0.5)
    assert np.allclose(out, [100, 50, 25])


def test_saturation_bounds():
    out = hill_saturation(np.array([0, 10, 1000]), half_sat=10)
    assert out[0] == 0 and abs(out[1] - 0.5) < 1e-9 and out[2] < 1
