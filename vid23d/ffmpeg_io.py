import os
import shutil
import subprocess
from pathlib import Path
from typing import Tuple, List


class FFmpegError(RuntimeError):
    pass


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise FFmpegError("ffmpeg is not installed or not in PATH")
    if shutil.which("ffprobe") is None:
        raise FFmpegError("ffprobe is not installed or not in PATH")


def run_cmd(cmd: List[str]) -> str:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise FFmpegError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}")
    return proc.stdout


def probe_fps(video_path: str) -> float:
    _require_ffmpeg()
    out = run_cmd([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path,
    ]).strip().splitlines()
    for line in out:
        if "/" in line:
            try:
                num, den = line.split("/")
                num_f, den_f = float(num), float(den)
                if den_f != 0:
                    return num_f / den_f
            except Exception:
                continue
        else:
            try:
                return float(line)
            except Exception:
                continue
    return 24.0


def extract_frames_and_audio(
    video_path: str,
    frames_dir: str,
    audio_path: str,
    stride: int = 1,
    target_height: int = 0,
    keep_audio: bool = True,
    audio_codec: str = "aac",
) -> Tuple[float, bool]:
    """Extract frames (PNG) and optionally audio from video.

    Returns (fps_for_render, has_audio).
    """
    _require_ffmpeg()
    if not os.path.isfile(video_path):
        raise FFmpegError(f"Input video not found: {video_path}")
    os.makedirs(frames_dir, exist_ok=True)
    fps_src = probe_fps(video_path)

    vf_parts: List[str] = []
    if target_height and target_height > 0:
        vf_parts.append(f"scale=-2:{int(target_height)}")
    if stride and stride > 1:
        vf_parts.append(f"fps=fps={fps_src/stride}")
    vf = ",".join(vf_parts) if vf_parts else f"fps=fps={fps_src}"

    run_cmd([
        "ffmpeg", "-y", "-i", video_path,
        "-vf", vf,
        os.path.join(frames_dir, "%06d.png"),
    ])

    has_audio = False
    if keep_audio:
        try:
            if audio_codec == "copy":
                # For stream copy, use Matroska container (.mka/.mkv) to avoid codec/container mismatch
                if not audio_path.lower().endswith((".mka", ".mkv")):
                    raise FFmpegError("For audio copy, use .mka/.mkv container to avoid muxing errors")
                run_cmd(["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "copy", audio_path])
            else:
                run_cmd(["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "aac", "-b:a", "192k", audio_path])
            has_audio = True
        except Exception:
            has_audio = False

    fps_out = fps_src / stride if stride and stride > 1 else fps_src
    return fps_out, has_audio


def render_video_from_frames(
    frames_dir: str,
    output_path: str,
    fps: float,
    audio_path: str | None = None,
    audio_codec: str = "aac",
    crf: int = 18,
    pix_fmt: str = "yuv420p",
) -> None:
    """Assemble video from frames, optionally mux audio.
    Frames must be named as %06d.png.
    """
    _require_ffmpeg()
    vpat = os.path.join(frames_dir, "%06d.png")
    if not os.path.isdir(frames_dir):
        raise FFmpegError(f"Frames directory not found: {frames_dir}")
    # Ensure at least one frame exists before invoking ffmpeg
    if not list(Path(frames_dir).glob("*.png")):
        raise FFmpegError(f"No frames found in directory: {frames_dir}")
    if audio_path and os.path.isfile(audio_path):
        cmd = [
            "ffmpeg", "-y", "-r", f"{fps:.6f}", "-i", vpat, "-i", audio_path,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-pix_fmt", pix_fmt, "-crf", str(crf),
            "-c:a", audio_codec,
            "-shortest",
            output_path,
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-r", f"{fps:.6f}", "-i", vpat,
            "-c:v", "libx264", "-pix_fmt", pix_fmt, "-crf", str(crf),
            output_path,
        ]
    run_cmd(cmd)


def list_frames(frames_dir: str) -> List[str]:
    p = Path(frames_dir)
    return sorted(str(x) for x in p.glob("*.png"))
