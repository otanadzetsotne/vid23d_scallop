## vid23d Scallop — 2D → 3D (SBS) Video Converter

Convert ordinary 2D video into stereoscopic SBS (Side‑by‑Side) video using neural depth estimation (MiDaS or optionally ZoeDepth), optional temporal smoothing (RAFT), depth‑based view synthesis, inpainting, and final FFmpeg rendering. Designed to produce SBS suitable for headsets like Quest.

### Requirements
- Python 3.10+
- FFmpeg installed and available in PATH
  - macOS: `brew install ffmpeg`
  - Ubuntu: `sudo apt-get update && sudo apt-get install -y ffmpeg`

### Install
```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# or install as a package
pip install .
```

Note on PyTorch wheels:
- Linux CUDA: requirements pin `torch`/`torchvision` to CUDA-enabled wheels. If you are on macOS or CPU-only Linux, install the appropriate wheels instead, for example:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Usage
```bash
# via Python
python main.py \
  --input_video /path/to/input.mp4 \
  --output_dir /path/to/output_dir \
  --sbs_mode full \
  --max_shift_px 32 \
  --target_height 0 \
  --frame_stride 1 \
  --depth_backend midas \
  --allow_zoe_fallback true \
  --use_raft_temporal true \
  --ema_alpha 0.7 \
  --use_lama_inpainting false \
  --keep_audio true \
  --audio_codec aac \
  --crf 18 \
  --pix_fmt yuv420p \
  --save_depth false \
  --save_depth_only false \
  --remove_tmp false

# or via console script
vid23d \
  --input_video /path/to/input.mp4 \
  --output_dir /path/to/output_dir
```

Key flags:
- `--sbs_mode`: `half` or `full` width packing
- `--max_shift_px`: horizontal disparity for nearest objects
- `--target_height`: 0 keeps original, >0 rescales by height
- `--frame_stride`: process every Nth frame to speed up
- `--depth_backend`: `midas` (default) or `zoedepth` (if installed)
- `--use_raft_temporal`: RAFT temporal smoothing, if torchvision RAFT available
- `--save_depth` / `--save_depth_only`: render depth previews

Outputs are placed under `--output_dir` and include `output_SBS_full.mp4` (or half), with optional `depth_preview.mp4`.

### Project structure
- `main.py`: CLI entry point
- `conf.py`: device selection (`cpu`/`mps`/`cuda`)
- `vid23d/`
  - `ffmpeg_io.py`: frame/audio extraction, render
  - `depth.py`: MiDaS/ZoeDepth inference
  - `temporal.py`: RAFT flow and warping
  - `stereo.py`: right-eye synthesis, inpainting, SBS packing
  - `pipeline.py`: orchestration and `process_video()`

### License
MIT — see `LICENSE`.

### Acknowledgements
MiDaS: `https://github.com/isl-org/MiDaS`
