#!/usr/bin/env python3
"""Check compiled correction precision, dynamic directions, and alpha changes."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
from mlx import nn
import numpy as np

from bonsai_abliterate.ablation import Ablated


def test_projection():
    rng = np.random.default_rng(42)
    for dtype, tolerance in ((mx.float16, 2e-3), (mx.float32, 2e-6)):
        for rows in (1, 17):
            y = mx.array(rng.normal(size=(1, rows, 5120)).astype(np.float32)).astype(dtype)
            for alpha in (0.0, 1.0, 2.0):
                # Distinct directions must remain runtime inputs, not captured constants.
                for _ in range(2):
                    d = rng.normal(size=5120).astype(np.float32)
                    wrapped = Ablated(nn.Identity(), mx.array(d), alpha)
                    direction = np.asarray(wrapped._direction).astype(np.float64)
                    original = np.asarray(y).astype(np.float64)
                    expected = original - alpha * np.sum(
                        original * direction, axis=-1, keepdims=True
                    ) * direction
                    actual = wrapped(y)
                    assert actual.dtype == dtype
                    np.testing.assert_allclose(
                        np.asarray(actual).astype(np.float32), expected,
                        rtol=tolerance, atol=tolerance,
                    )
                    if alpha == 0:
                        np.testing.assert_array_equal(np.asarray(actual), np.asarray(y))
                    if alpha == 1:
                        residual = np.sum(np.asarray(actual).astype(np.float64) * direction, axis=-1)
                        assert np.max(np.abs(residual)) < tolerance


if __name__ == "__main__":
    test_projection()
    print("PASS: compiled correction preserves precision, direction, alpha, and dtype")
