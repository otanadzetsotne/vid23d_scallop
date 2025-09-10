from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .config import DEVICE

try:
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    _HAVE_RAFT = True
except Exception:
    _HAVE_RAFT = False


class RaftWrapper:
    def __init__(self) -> None:
        if not _HAVE_RAFT:
            raise RuntimeError("torchvision RAFT is not available")
        self.weights = Raft_Large_Weights.C_T_V2
        self.model = raft_large(weights=self.weights, progress=True).to(DEVICE).eval()

    def flow(self, prev_bgr: np.ndarray, curr_bgr: np.ndarray) -> np.ndarray:
        prev_rgb = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        curr_rgb = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        t1 = torch.from_numpy(prev_rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        t2 = torch.from_numpy(curr_rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        # Use torchvision-provided transforms for the chosen weights if available
        try:
            preprocess = self.weights.transforms()
            t1 = preprocess(t1)
            t2 = preprocess(t2)
        except Exception:
            # Fallback to simple normalization
            t1 = (t1 - 0.5) / 0.5
            t2 = (t2 - 0.5) / 0.5

        with torch.no_grad():
            out = self.model(t1, t2)
        if isinstance(out, (list, tuple)):
            flow_up = out[-1]
        elif isinstance(out, dict):
            flow_up = next(iter(out.values()))
        else:
            flow_up = out

        if flow_up.ndim == 4 and flow_up.shape[1] == 2:
            flow = flow_up[0].permute(1, 2, 0).detach().cpu().numpy()
        elif flow_up.ndim == 4 and flow_up.shape[-1] == 2:
            flow = flow_up[0].detach().cpu().numpy()
        else:
            raise RuntimeError(f"Unexpected RAFT output shape: {tuple(flow_up.shape)}")
        return flow.astype(np.float32)


def warp_with_flow(tensor_2d: np.ndarray, flow: np.ndarray) -> np.ndarray:
    H, W = tensor_2d.shape
    grid_y, grid_x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    xs = (grid_x - flow[..., 0])
    ys = (grid_y - flow[..., 1])
    xs_norm = (xs / (W - 1)) * 2 - 1
    ys_norm = (ys / (H - 1)) * 2 - 1
    grid = np.stack([xs_norm, ys_norm], axis=-1)
    grid_t = torch.from_numpy(grid).float().unsqueeze(0).to(DEVICE)
    src = torch.from_numpy(tensor_2d).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    warped = F.grid_sample(src, grid_t, mode="bilinear", padding_mode="border", align_corners=False)
    return warped.squeeze().detach().cpu().numpy().astype(np.float32)
