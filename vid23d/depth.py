from __future__ import annotations

import warnings
from typing import Optional, Tuple
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .config import DEVICE


class DepthLoadError(RuntimeError):
    pass


def _load_midas() -> Tuple[torch.nn.Module, object]:
    """Load MiDaS DPT Large and its transform via torch.hub.

    Returns (model, transform_callable)
    """
    # Try to respect optional env preference for a lighter backbone to reduce crashes/timeouts
    variant = os.environ.get("VID23D_MIDAS_VARIANT", "DPT_Large")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        midas = torch.hub.load(
            "intel-isl/MiDaS", variant, trust_repo=True
        ).to(DEVICE).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    # Select appropriate transform
    if hasattr(transforms, 'dpt_transform'):
        transform = transforms.dpt_transform
    else:
        transform = transforms.default_transform
    return midas, transform


def _load_zoedepth() -> Tuple[Optional[torch.nn.Module], Optional[object]]:
    try:
        from zoedepth.utils.config import get_config
        from zoedepth.models.builder import build_model
        from zoedepth.utils.checkpoint import load_network
    except Exception:
        return None, None

    conf = get_config("zoedepth", "infer")
    conf["model"]["name"] = "zoedepth_nk"
    model = build_model(conf).to(DEVICE).eval()
    model = load_network(model, conf)
    return model, None


class DepthEstimator:
    """Unified API for producing normalized disparity in [0,1]."""

    def __init__(self, prefer_backend: str = "midas", allow_zoe_fallback: bool = True) -> None:
        """prefer_backend: 'midas' or 'zoedepth'.
        If 'zoedepth' is requested but unavailable and allow_zoe_fallback is True,
        falls back to MiDaS.
        """
        backend = prefer_backend.lower()
        self.kind = "midas"
        self.model: Optional[torch.nn.Module] = None
        self.transform = None

        try:
            if backend == "zoedepth":
                model, _ = _load_zoedepth()
                if model is not None:
                    self.kind = "zoedepth"
                    self.model = model
            if self.model is None:
                if backend == "zoedepth" and not allow_zoe_fallback:
                    raise DepthLoadError("ZoeDepth requested but not available")
                self.model, self.transform = _load_midas()
                self.kind = "midas"
        except Exception as e:
            # Final safety: offline/damaged environments fallback to a lightweight heuristic depth
            self.kind = "dummy"
            self.model = None
            self.transform = None

        # no global switch; rely on local torch.no_grad()

    def predict_disp01(self, bgr: np.ndarray) -> np.ndarray:
        """Return normalized inverse depth (disparity-like) in [0,1] with shape [H,W]."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        if self.kind == "dummy" or self.model is None:
            # Heuristic: use luminance + slight edge emphasis, normalize to [0,1]
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            edges = cv2.Laplacian(gray, cv2.CV_32F)
            disp = gray * 0.8 + np.clip(edges, -1.0, 1.0) * 0.2
            p5, p95 = np.percentile(disp, [5, 95])
            disp = np.clip((disp - p5) / (p95 - p5 + 1e-8), 0.0, 1.0)
            return disp.astype(np.float32)
        if self.kind == "midas":
            with torch.no_grad():
                inp = self.transform(rgb).to(DEVICE)
                pred = self.model(inp)
            if pred.ndim == 3:
                pred = pred.unsqueeze(1)
            elif pred.ndim == 2:
                pred = pred.unsqueeze(0).unsqueeze(0)
            pred = F.interpolate(pred, size=(H, W), mode="bicubic", align_corners=False)
            x = pred[0, 0].float()
            x = (x - x.min()) / (x.max() - x.min() + 1e-8)
            return x.detach().cpu().numpy().astype(np.float32)
        else:
            with torch.no_grad():
                t = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0
                # ZoeDepth exposes .infer for depth
                depth = self.model.infer(t)
            if depth.ndim == 3:
                depth = depth.unsqueeze(1)
            d = depth[0, 0].detach().cpu().numpy()
            inv = 1.0 / (d + 1e-6)
            p5, p95 = np.percentile(inv, [5, 95])
            inv = np.clip((inv - p5) / (p95 - p5 + 1e-8), 0.0, 1.0)
            return inv.astype(np.float32)
