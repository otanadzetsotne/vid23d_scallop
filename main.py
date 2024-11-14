import os
import pickle
import hashlib
from datetime import datetime

from tqdm import tqdm
from absl import flags, app

from frames import video_frames, video_generator
from img_depth import to_depth
from stereo import create_stereo_pair, concatenate_stereo_pair
from audio import extract_and_add_audio
from itertools import tee

def main(argv):
    current_datetime = datetime.now().strftime('%Y%m%d-%H%M')

    video_path = FLAGS.input_video
    depth_map_path = FLAGS.input_depth_map

    frames_list, fps = video_frames(video_path)
    frames_list, frames_list2 = tee(frames_list)

    if depth_map_path:
        depth_data, _ = video_frames(depth_map_path)
    else:
        depth_data = (to_depth(frame, FLAGS.depth_model) for frame in frames_list2)
    depth_data, depth_data2 = tee(depth_data)

    video_generators = []


    # Create stereo pairs and concatenate them
    stereo_pairs = (create_stereo_pair(frame, depth) for frame, depth in zip(frames_list, depth_data))
    stereo_frames = (concatenate_stereo_pair(left, right) for left, right in stereo_pairs)

    result_video_path_noaudio = os.path.join(FLAGS.output_dir, f'{current_datetime}-noaudio.mp4')
    result_video_path = os.path.join(FLAGS.output_dir, f'{current_datetime}.mp4')
    video_generators.append(video_generator(stereo_frames, result_video_path_noaudio, fps))

    if FLAGS.save_depth:
        depth_video_path = os.path.join(FLAGS.output_dir, f'{current_datetime}_depth_{FLAGS.depth_model}.mp4')
        video_generators.append(video_generator(depth_data2, depth_video_path, fps))
        # extract_and_add_audio(video_path, depth_video_path, depth_video_path)
    
    for _ in zip(*video_generators): pass # start process
    if not FLAGS.save_depth_only:
        extract_and_add_audio(video_path, result_video_path_noaudio, result_video_path)


if __name__ == '__main__':
    flags.DEFINE_string('input_video', None, 'Path to 2D video for processing', required=True)
    flags.DEFINE_string('output_dir', None, 'Path to directory to store resulting videos', required=True)
    flags.DEFINE_bool('save_depth', False, 'Create video of depth map')
    flags.DEFINE_bool('save_depth_only', False, 'Create video of depth map instead of a 3D Video')
    flags.DEFINE_string('input_depth_map', None, 'Path to depth video')
    flags.DEFINE_bool('remove_tmp', False, 'Remove temporary directory')
    flags.DEFINE_enum('depth_model', 'DPT_Large', ['DPT_Large', 'DPT_Hybrid', 'MiDaS_small'], 'Depth model type')
    FLAGS = flags.FLAGS
    app.run(main)
