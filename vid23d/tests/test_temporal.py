from __future__ import annotations

import unittest
import numpy as np

from vid23d.temporal import warp_with_flow


class TestTemporal(unittest.TestCase):
    def test_warp_identity_flow(self):
        img = np.linspace(0, 1, 64 * 48, dtype=np.float32).reshape(48, 64)
        flow = np.zeros((48, 64, 2), dtype=np.float32)
        warped = warp_with_flow(img, flow)
        self.assertTrue(np.allclose(img, warped, atol=1e-5))

