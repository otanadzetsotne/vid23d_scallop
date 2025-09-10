from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import imageio.v3 as iio

from vid23d.ffmpeg_io import extract_frames_and_audio, render_video_from_frames, list_frames, FFmpegError


def _write_synthetic_video(path: str, num_frames: int = 12, size: tuple[int, int] = (64, 48), fps: int = 12):
    # Write simple RGB frames via imageio-ffmpeg backend
    writer = iio.get_writer(path, fps=fps)
    W, H = size
    for i in range(num_frames):
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        frame[:, :, 0] = (i * 20) % 255
        frame[:, :, 1] = (i * 40) % 255
        frame[:, :, 2] = (i * 60) % 255
        writer.append_data(frame)
    writer.close()


def test_extract_and_render_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        src = str(Path(td) / "input.mp4")
        _write_synthetic_video(src)

        frames_dir = str(Path(td) / "frames")
        audio_path = str(Path(td) / "audio.m4a")
        fps, has_audio = extract_frames_and_audio(src, frames_dir, audio_path, stride=1, target_height=0, keep_audio=False)
        assert fps > 0
        assert has_audio is False
        frames = list_frames(frames_dir)
        assert len(frames) > 0

        out = str(Path(td) / "out.mp4")
        render_video_from_frames(frames_dir, out, fps)
        assert os.path.isfile(out)

