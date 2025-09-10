"""vid23d package: 2D -> 3D SBS video conversion pipeline.

Modules:
- ffmpeg_io: frame/audio extraction and final rendering via FFmpeg
- depth: MiDaS/ZoeDepth depth estimation
- temporal: RAFT-based temporal smoothing
- stereo: right-eye synthesis, inpainting, SBS packing
- pipeline: orchestration utilities
"""

__all__ = [
    "ffmpeg_io",
    "depth",
    "temporal",
    "stereo",
    "pipeline",
]
