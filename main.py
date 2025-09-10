from absl import flags, app
from pathlib import Path
import shutil
import sys

from vid23d.pipeline import PipelineConfig, process_video


FLAGS = flags.FLAGS


def main(argv):
    del argv
    cfg = PipelineConfig(
        input_video=FLAGS.input_video,
        output_dir=FLAGS.output_dir,
        sbs_mode=FLAGS.sbs_mode,
        max_shift_px=FLAGS.max_shift_px,
        target_height=FLAGS.target_height,
        frame_stride=FLAGS.frame_stride,
        depth_backend=FLAGS.depth_backend,
        allow_zoe_fallback=FLAGS.allow_zoe_fallback,
        use_raft_temporal=FLAGS.use_raft_temporal,
        ema_alpha=FLAGS.ema_alpha,
        use_lama_inpainting=FLAGS.use_lama_inpainting,
        keep_audio=FLAGS.keep_audio,
        audio_codec=FLAGS.audio_codec,
        crf=FLAGS.crf,
        pix_fmt=FLAGS.pix_fmt,
        save_depth_video=FLAGS.save_depth,
        save_depth_only=FLAGS.save_depth_only,
    )

    output_path = process_video(cfg)

    if FLAGS.remove_tmp:
        work_dir = Path(FLAGS.output_dir)
        for rel in ["frames_src", "frames_sbs", "frames_depth", "audio.m4a"]:
            p = work_dir / rel
            try:
                if p.is_dir():
                    shutil.rmtree(p)
                elif p.exists():
                    p.unlink()
            except Exception:
                pass

    print(f"Done: {output_path}")


if __name__ == '__main__':
    flags.DEFINE_string('input_video', None, 'Path to input 2D video')
    flags.DEFINE_string('output_dir', None, 'Directory to store outputs')

    flags.DEFINE_enum('sbs_mode', 'full', ['half', 'full'], 'SBS packing: half or full width')
    flags.DEFINE_integer('max_shift_px', 32, 'Max horizontal shift (pixels) for nearest objects')
    flags.DEFINE_integer('target_height', 0, 'Resize video to this height, 0 = keep original')
    flags.DEFINE_integer('frame_stride', 1, 'Process every Nth frame (1 = all)')

    flags.DEFINE_enum('depth_backend', 'midas', ['midas', 'zoedepth'], 'Depth model backend')
    flags.DEFINE_boolean('allow_zoe_fallback', True, 'Fallback to MiDaS if ZoeDepth unavailable')
    flags.DEFINE_boolean('use_raft_temporal', True, 'Use RAFT-based temporal smoothing')
    flags.DEFINE_float('ema_alpha', 0.7, 'EMA weight for current frame (0..1)')
    flags.DEFINE_boolean('use_lama_inpainting', False, 'Use LaMa inpainting (if available)')

    flags.DEFINE_boolean('keep_audio', True, 'Keep and mux original audio when possible')
    flags.DEFINE_enum('audio_codec', 'aac', ['aac', 'copy'], 'Audio codec for muxing')
    flags.DEFINE_integer('crf', 18, 'CRF for libx264')
    flags.DEFINE_string('pix_fmt', 'yuv420p', 'Pixel format for libx264')

    flags.DEFINE_boolean('save_depth', False, 'Also render depth preview video')
    flags.DEFINE_boolean('save_depth_only', False, 'Render only depth video (no SBS)')
    flags.DEFINE_boolean('remove_tmp', False, 'Remove temporary frames/audio after rendering')

    flags.mark_flag_as_required('input_video')
    flags.mark_flag_as_required('output_dir')

    app.run(main)
