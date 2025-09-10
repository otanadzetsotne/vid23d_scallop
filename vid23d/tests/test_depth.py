from __future__ import annotations

import os
import unittest

import numpy as np

from vid23d.depth import DepthEstimator


class TestDepth(unittest.TestCase):
    def test_midas_predicts_disp(self):
        # Use a small synthetic image to reduce load
        img = np.zeros((64, 96, 3), dtype=np.uint8)
        img[:, :48] = (50, 100, 150)
        img[:, 48:] = (150, 100, 50)

        # Allow env to switch to a lighter MiDaS variant in CI if needed
        # but default loader will work in GPU env
        est = DepthEstimator(prefer_backend=os.environ.get("VID23D_DEPTH", "midas"))
        disp = est.predict_disp01(img)
        self.assertEqual(disp.shape[:2], img.shape[:2])
        self.assertTrue(np.isfinite(disp).all())
        self.assertGreaterEqual(float(disp.min()), 0.0)
        self.assertLessEqual(float(disp.max()), 1.0)

