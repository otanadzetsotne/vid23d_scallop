from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

from .ffmpeg_io import extract_frames_and_audio, list_frames, render_video_from_frames
from .depth import DepthEstimator
from .temporal import RaftWrapper, warp_with_flow
from .stereo import synthesize_right, inpaint_frame, pack_sbs


@dataclass
class PipelineConfig:
    input_video: str
    output_dir: str
    sbs_mode: str = "full"              # "half" or "full"
    max_shift_px: int = 32
    target_height: int = 0               # 0 = keep
    frame_stride: int = 1
    depth_backend: str = "midas"         # "midas" | "zoedepth"
    allow_zoe_fallback: bool = True
    use_raft_temporal: bool = True
    ema_alpha: float = 0.7
    use_lama_inpainting: bool = False
    keep_audio: bool = True
    audio_codec: str = "aac"             # "aac" | "copy"
    crf: int = 18
    pix_fmt: str = "yuv420p"

    # Optional outputs
    save_depth_video: bool = False
    save_depth_only: bool = False


def process_video(cfg: PipelineConfig) -> str:
    video_path = cfg.input_video
    work_dir = Path(cfg.output_dir)
    frames_dir = str(work_dir / "frames_src")
    sbs_dir = str(work_dir / "frames_sbs")
    # Choose audio container based on requested codec to avoid mux/extract failures
    audio_ext = ".mka" if cfg.audio_codec == "copy" else ".m4a"
    audio_path = str(work_dir / f"audio{audio_ext}")
    out_video = str(work_dir / (f"output_SBS_{'half' if cfg.sbs_mode=='half' else 'full'}.mp4"))
    depth_frames_dir = str(work_dir / "frames_depth")

    for d in [frames_dir, sbs_dir]:
        os.makedirs(d, exist_ok=True)
    if cfg.save_depth_video or cfg.save_depth_only:
        os.makedirs(depth_frames_dir, exist_ok=True)

    fps, has_audio = extract_frames_and_audio(
        video_path, frames_dir, audio_path,
        stride=cfg.frame_stride,
        target_height=cfg.target_height,
        keep_audio=cfg.keep_audio,
        audio_codec=cfg.audio_codec,
    )

    fpaths = list_frames(frames_dir)
    if len(fpaths) == 0:
        raise RuntimeError("No frames extracted from input video")

    depth_estimator = DepthEstimator(prefer_backend=cfg.depth_backend, allow_zoe_fallback=cfg.allow_zoe_fallback)
    raft: Optional[RaftWrapper] = None
    if cfg.use_raft_temporal:
        try:
            raft = RaftWrapper()
        except Exception as e:
            raft = None
            # Warn once if RAFT was requested but unavailable
            print(f"[vid23d] Warning: RAFT unavailable ({e}). Continuing without temporal smoothing.")

    prev_bgr: Optional[np.ndarray] = None
    prev_disp: Optional[np.ndarray] = None

    sbs_written = 0
    depth_written = 0
    for fp in tqdm(fpaths, desc="Processing frames"):
        bgr = cv2.imread(fp, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[vid23d] Warning: failed to read frame '{fp}', skipping.")
            continue
        disp = depth_estimator.predict_disp01(bgr)

        if raft is not None and prev_bgr is not None and prev_disp is not None:
            flow = raft.flow(prev_bgr, bgr)
            prev_disp_warped = warp_with_flow(prev_disp, flow)
            disp = cfg.ema_alpha * disp + (1.0 - cfg.ema_alpha) * prev_disp_warped
            disp = np.clip(disp, 0.0, 1.0)

        if cfg.save_depth_video or cfg.save_depth_only:
            depth_viz = (disp * 255.0).astype(np.uint8)
            depth_viz = cv2.applyColorMap(depth_viz, cv2.COLORMAP_INFERNO)
            if cv2.imwrite(os.path.join(depth_frames_dir, os.path.basename(fp)), depth_viz):
                depth_written += 1

        if not cfg.save_depth_only:
            right, hole = synthesize_right(bgr, disp, cfg.max_shift_px)
            right = inpaint_frame(right, hole, use_lama=cfg.use_lama_inpainting)
            sbs = pack_sbs(bgr, right, cfg.sbs_mode)
            if cv2.imwrite(os.path.join(sbs_dir, os.path.basename(fp)), sbs):
                sbs_written += 1

        prev_bgr = bgr
        prev_disp = disp

    # Render outputs
    if cfg.save_depth_video:
        if depth_written == 0:
            raise RuntimeError("No depth frames were produced; cannot render depth video.")
        render_video_from_frames(depth_frames_dir, str(work_dir / "depth_preview.mp4"), fps)

    if cfg.save_depth_only:
        output_path = str(work_dir / "depth_only.mp4")
        if depth_written == 0:
            raise RuntimeError("No depth frames were produced; cannot render depth-only video.")
        render_video_from_frames(depth_frames_dir, output_path, fps)
        return output_path

    if sbs_written == 0:
        raise RuntimeError("No SBS frames were produced; aborting render.")

    audio_for_mux = audio_path if (cfg.keep_audio and has_audio) else None
    render_video_from_frames(sbs_dir, out_video, fps, audio_path=audio_for_mux, audio_codec=cfg.audio_codec, crf=cfg.crf, pix_fmt=cfg.pix_fmt)
    return out_video
