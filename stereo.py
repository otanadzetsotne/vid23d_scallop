import cv2
import numpy as np


def create_stereo_pair(image, depth_map, baseline_distance=0.01, depth_scale_factor=0.15):
    assert image.shape[:2] == depth_map.shape[:2], 'The image and depth map must be the same size.'

    depth_map = ((depth_map.astype(np.float32) ** 2) / depth_scale_factor) ** .5
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)

    height, width = image.shape[:2]
    x_coords = np.tile(np.arange(width), (height, 1)).astype(np.float32)
    y_coords = np.repeat(np.arange(height), width).reshape(height, width).astype(np.float32)

    displacement = (baseline_distance * (depth_map_normalized * width)).astype(np.float32)

    left_coords = np.stack((x_coords + displacement, y_coords), axis=-1)
    right_coords = np.stack((x_coords - displacement, y_coords), axis=-1)

    left_image = cv2.remap(image, left_coords, None, cv2.INTER_LINEAR)
    right_image = cv2.remap(image, right_coords, None, cv2.INTER_LINEAR)

    return left_image, right_image


def concatenate_stereo_pair(left_image, right_image, separator_width=0, separator_color=(0, 0, 255)):
    """
    Concatenates the left and right images of a stereo pair horizontally, with an optional colored separator.

    Parameters:
    - left_image, right_image: The left and right images to be concatenated.
    - separator_width: The width of the separator to be added between the images. Default is 0 (no separator).
    - separator_color: The color of the separator in BGR format. Default is blue.

    Returns:
    The concatenated image as a single numpy array.
    """
    # Ensure the two images have the same dimensions
    assert left_image.shape == right_image.shape, 'Images must be the same size.'

    # Concatenate images directly if no separator is needed
    if separator_width <= 0:
        return np.concatenate((left_image, right_image), axis=1)

    # Create the separator and concatenate with it if needed
    height = left_image.shape[0]
    separator = np.full((height, separator_width, 3), separator_color, dtype=np.uint8)
    concatenated_image = np.concatenate((left_image, separator, right_image), axis=1)

    return concatenated_image
