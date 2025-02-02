import os

import cv2
from tqdm import tqdm

from img_depth import to_depth
from stereo import create_stereo_pair, concatenate_stereo_pair

import itertools


def video_frames(video_path):
    video = cv2.VideoCapture(video_path)
    frames_quantity = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_count = video.get(cv2.CAP_PROP_FPS)
    progress = tqdm(total=frames_quantity)
    
    def get_frames():
        success = True
        while success:
            success, image = video.read()
            if success:
                progress.update(1)
                yield(image)

    return (get_frames(), fps_count)



def extract_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video = cv2.VideoCapture(video_path)
    print(f'Video loaded. FPS: {video.get(cv2.CAP_PROP_FPS)}')

    frames_quantity = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    progress = tqdm(total=frames_quantity)
    count = 0
    success = True
    frames = []
    while success:
        success, image = video.read()
        if success:
            frames.append((image, to_depth(image)))
            progress.update(1)

    stereo_frames = []
    for image, depth in tqdm(frames):
        stereo = create_stereo_pair(image, depth)
        stereo_frames.append(concatenate_stereo_pair(*stereo))

    for i, frame in enumerate(tqdm(stereo_frames)):
        cv2.imwrite(os.path.join(output_folder, f'frame_{i}.jpg'), frame)

    video.release()
    print(f'Extracted {count} frames', flush=True)


def concat_frames(input_folder, output_video_path, fps=30):
    images = [img for img in os.listdir(input_folder) if img.endswith('.jpg')]
    images.sort(key=lambda x: int(x.replace('frame_', '').replace('.jpg', '')))

    frame = cv2.imread(os.path.join(input_folder, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(input_folder, image)))

    cv2.destroyAllWindows()
    video.release()


def video_generator(frames, output_video_path, fps=30):
    if not frames:
        raise ValueError('Empty frames list')

    first_frame = next(frames)
    frame_height, frame_width = first_frame.shape[:2]
    frame_size = (frame_width, frame_height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    for frame in itertools.chain([first_frame], frames):
        if len(first_frame.shape) == 2:  # Greyscale image
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        video.write(frame)
        yield

    cv2.destroyAllWindows()
    video.release()


if __name__ == '__main__':
    VID_TEST_PATH = os.path.abspath('.output/test.mp4')
    VID_DEPTH_PATH = os.path.abspath('.output/test.mp4')
    FRAMES_PATH = os.path.abspath('frames')

    extract_frames(VID_TEST_PATH, FRAMES_PATH)
    concat_frames(FRAMES_PATH, VID_DEPTH_PATH)
