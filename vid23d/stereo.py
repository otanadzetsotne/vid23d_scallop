from __future__ import annotations

import numpy as np
import cv2


def synthesize_right(bgr_left: np.ndarray, disp01: np.ndarray, max_shift_px: int) -> tuple[np.ndarray, np.ndarray]:
    """Forward-warp left view into right view using z-buffer-like order.

    Returns (right_bgr, hole_mask_bool)
    """
    H, W, _ = bgr_left.shape
    shift = (disp01 * float(max_shift_px)).astype(np.float32)

    order = np.argsort(disp01.ravel())
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    xs_f = xs.ravel().astype(np.float32)
    ys_f = ys.ravel().astype(np.int32)
    shift_f = shift.ravel()

    x2 = np.floor(xs_f - shift_f).astype(np.int32)
    x2 = np.clip(x2, 0, W - 1)

    right = np.zeros_like(bgr_left)
    filled = np.zeros((H, W), dtype=np.bool_)

    src_idx = order
    tgt_x = x2[order]
    tgt_y = ys_f[order]

    right_flat = right.reshape(-1, 3)
    left_flat = bgr_left.reshape(-1, 3)
    filled_flat = filled.reshape(-1)

    tgt_flat_idx = (tgt_y * W + tgt_x)
    chunk = 2_000_000
    for i in range(0, len(src_idx), chunk):
        sl = slice(i, i + chunk)
        right_flat[tgt_flat_idx[sl]] = left_flat[src_idx[sl]]
        filled_flat[tgt_flat_idx[sl]] = True

    right = right_flat.reshape(H, W, 3)
    filled = filled_flat.reshape(H, W)
    hole = ~filled
    return right, hole


def inpaint_frame(bgr: np.ndarray, hole_mask: np.ndarray, use_lama: bool = False) -> np.ndarray:
    if not hole_mask.any():
        return bgr
    if use_lama:
        # Placeholder for LaMa integration (optional)
        pass
    hole_u8 = (hole_mask.astype(np.uint8)) * 255
    hole_u8 = cv2.dilate(hole_u8, np.ones((3, 3), np.uint8), iterations=1)
    out = cv2.inpaint(bgr, hole_u8, 3, cv2.INPAINT_TELEA)
    return out


def pack_sbs(left_bgr: np.ndarray, right_bgr: np.ndarray, mode: str = "half") -> np.ndarray:
    H, W, _ = left_bgr.shape
    if mode == "half":
        left_r = cv2.resize(left_bgr, (W // 2, H), interpolation=cv2.INTER_AREA)
        right_r = cv2.resize(right_bgr, (W // 2, H), interpolation=cv2.INTER_AREA)
        sbs = np.hstack([left_r, right_r])
    else:
        sbs = np.hstack([left_bgr, right_bgr])
    return sbs
