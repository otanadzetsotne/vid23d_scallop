from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import imageio.v3 as iio

from vid23d.pipeline import PipelineConfig, process_video


def _write_synthetic_video(path: str, num_frames: int = 12, size: tuple[int, int] = (64, 48), fps: int = 12):
    writer = iio.get_writer(path, fps=fps)
    W, H = size
    for i in range(num_frames):
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        frame[:, :, 0] = (i * 20) % 255
        frame[:, :, 1] = (i * 40) % 255
        frame[:, :, 2] = (i * 60) % 255
        writer.append_data(frame)
    writer.close()


def test_full_pipeline_synthetic_video():
    with tempfile.TemporaryDirectory() as td:
        src = str(Path(td) / "input.mp4")
        _write_synthetic_video(src, num_frames=8, size=(96, 64), fps=8)
        out_dir = str(Path(td) / "out")
        os.makedirs(out_dir, exist_ok=True)

        cfg = PipelineConfig(
            input_video=src,
            output_dir=out_dir,
            sbs_mode="half",
            max_shift_px=8,
            target_height=0,
            frame_stride=1,
            depth_backend=os.environ.get("VID23D_DEPTH", "midas"),
            allow_zoe_fallback=True,
            use_raft_temporal=False,  # faster for tests; flow tested separately
            ema_alpha=0.7,
            use_lama_inpainting=False,
            keep_audio=False,
            audio_codec="aac",
            crf=28,
            pix_fmt="yuv420p",
            save_depth_video=False,
            save_depth_only=False,
        )
        out_path = process_video(cfg)
        assert os.path.isfile(out_path)

