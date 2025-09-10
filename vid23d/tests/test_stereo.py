from __future__ import annotations

import unittest
import numpy as np

from vid23d.stereo import synthesize_right, inpaint_frame, pack_sbs


class TestStereo(unittest.TestCase):
    def test_synthesize_and_pack(self):
        H, W = 48, 64
        left = np.zeros((H, W, 3), dtype=np.uint8)
        left[:, : W // 2, 0] = 255
        left[:, W // 2 :, 2] = 255
        disp = np.zeros((H, W), dtype=np.float32)
        disp[:, : W // 2] = 1.0
        right, hole = synthesize_right(left, disp, max_shift_px=8)
        self.assertEqual(right.shape, left.shape)
        out = inpaint_frame(right, hole)
        sbs = pack_sbs(left, out, mode="half")
        self.assertEqual(sbs.shape[0], H)
        self.assertEqual(sbs.shape[1], W)

