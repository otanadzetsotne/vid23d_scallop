import os
import pickle
import hashlib
from datetime import datetime

from tqdm import tqdm
from absl import flags, app

from frames import video_frames, frames_to_vid
from img_depth import to_depth
from stereo import create_stereo_pair, concatenate_stereo_pair
from audio import extract_and_add_audio


def hash_file(filepath):
    # Initialize the SHA-512 hash object
    hash_object = hashlib.sha512()

    # Open the file in binary read mode and read it in 4096-byte blocks
    with open(filepath, 'rb') as file:
        for data_block in iter(lambda: file.read(4096), b""):
            hash_object.update(data_block)

    # Return the hexadecimal digest of the hash
    return hash_object.hexdigest()


def main(argv):
    current_datetime = datetime.now().strftime('%Y%m%d-%H%M')

    video_path = FLAGS.input_video
    video_hash_path = os.path.join(FLAGS.output_dir, hash_file(video_path))
    os.makedirs(video_hash_path, exist_ok=True)

    frames_list = video_frames(video_path)

    depth_file_path = os.path.join(video_hash_path, f'depth_{FLAGS.depth_model}.pickle')
    if os.path.exists(depth_file_path):
        with open(depth_file_path, 'rb') as file:
            depth_data = pickle.load(file)
    else:
        # Calculate depth for each frame using the specified depth model
        depth_data = [to_depth(frame, FLAGS.depth_model) for frame in tqdm(frames_list)]
        with open(depth_file_path, 'wb') as file:
            pickle.dump(depth_data, file)

    if FLAGS.save_depth:
        depth_video_path = os.path.join(FLAGS.output_dir, f'{current_datetime}_depth_{FLAGS.depth_model}.mp4')
        frames_to_vid(depth_data, depth_video_path)
        # extract_and_add_audio(video_path, depth_video_path, depth_video_path)

    # Create stereo pairs and concatenate them
    stereo_pairs = [create_stereo_pair(frame, depth) for frame, depth in tqdm(zip(frames_list, depth_data))]
    del depth_data  # Clear depth data from memory
    stereo_frames = [concatenate_stereo_pair(left, right) for left, right in tqdm(stereo_pairs)]
    del stereo_pairs  # Clear stereo pairs from memory

    result_video_path = os.path.join(FLAGS.output_dir, f'{current_datetime}.mp4')
    frames_to_vid(stereo_frames, result_video_path)
    # extract_and_add_audio(video_path, result_video_path, result_video_path)


if __name__ == '__main__':
    flags.DEFINE_string('input_video', None, 'Path to 2D video for processing', required=True)
    flags.DEFINE_string('output_dir', None, 'Path to directory to store resulting videos', required=True)
    flags.DEFINE_bool('save_depth', False, 'Create video of depth map')
    flags.DEFINE_bool('remove_tmp', False, 'Remove temporary directory')
    flags.DEFINE_enum('depth_model', 'DPT_Large', ['DPT_Large', 'DPT_Hybrid', 'MiDaS_small'], 'Depth model type')
    FLAGS = flags.FLAGS
    app.run(main)
