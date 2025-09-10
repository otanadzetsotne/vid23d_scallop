from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from .pipeline import PipelineConfig, process_video


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vid23d", description="2D -> 3D (SBS) video converter")
    p.add_argument("--input_video", required=True, help="Path to input 2D video")
    p.add_argument("--output_dir", required=True, help="Directory to store outputs")

    p.add_argument("--sbs_mode", choices=["half", "full"], default="full")
    p.add_argument("--max_shift_px", type=int, default=32)
    p.add_argument("--target_height", type=int, default=0)
    p.add_argument("--frame_stride", type=int, default=1)

    p.add_argument("--depth_backend", choices=["midas", "zoedepth"], default="midas")
    p.add_argument("--allow_zoe_fallback", type=lambda x: x.lower() == "true", default=True)
    p.add_argument("--use_raft_temporal", type=lambda x: x.lower() == "true", default=True)
    p.add_argument("--ema_alpha", type=float, default=0.7)
    p.add_argument("--use_lama_inpainting", type=lambda x: x.lower() == "true", default=False)

    p.add_argument("--keep_audio", type=lambda x: x.lower() == "true", default=True)
    p.add_argument("--audio_codec", choices=["aac", "copy"], default="aac")
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--pix_fmt", default="yuv420p")

    p.add_argument("--save_depth", type=lambda x: x.lower() == "true", default=False)
    p.add_argument("--save_depth_only", type=lambda x: x.lower() == "true", default=False)
    p.add_argument("--remove_tmp", type=lambda x: x.lower() == "true", default=False)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = PipelineConfig(
        input_video=args.input_video,
        output_dir=args.output_dir,
        sbs_mode=args.sbs_mode,
        max_shift_px=args.max_shift_px,
        target_height=args.target_height,
        frame_stride=args.frame_stride,
        depth_backend=args.depth_backend,
        allow_zoe_fallback=args.allow_zoe_fallback,
        use_raft_temporal=args.use_raft_temporal,
        ema_alpha=args.ema_alpha,
        use_lama_inpainting=args.use_lama_inpainting,
        keep_audio=args.keep_audio,
        audio_codec=args.audio_codec,
        crf=args.crf,
        pix_fmt=args.pix_fmt,
        save_depth_video=args.save_depth,
        save_depth_only=args.save_depth_only,
    )
    out = process_video(cfg)
    if args.remove_tmp:
        work_dir = Path(args.output_dir)
        for rel in ["frames_src", "frames_sbs", "frames_depth", "audio.m4a"]:
            p = work_dir / rel
            try:
                if p.is_dir():
                    shutil.rmtree(p)
                elif p.exists():
                    p.unlink()
            except Exception:
                pass
    print(f"Done: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
