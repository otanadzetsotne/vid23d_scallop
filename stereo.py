import cv2
import numpy as np


def create_stereo_pair(image, depth_map, baseline_distance=0.005, depth_scale_factor=0.5):
    """
    Generates a stereo pair of images (left and right) from a single image and a depth map by simulating
    the parallax effect seen in stereo vision.

    Parameters:
    - image: The input image for which the stereo pair will be created.
    - depth_map: The depth map of the input image. Values represent depth intensity.
    - baseline_distance: The distance between the two virtual cameras. Default is 0.005.
    - depth_scale_factor: A scaling factor applied to the depth map to adjust the effect of depth. Default is 0.5.

    Returns:
    A tuple containing the left and right images of the stereo pair.
    """
    # Ensure the input image and depth map have the same dimensions
    assert image.shape[:2] == depth_map.shape[:2], 'The image and depth map must be the same size.'

    # Scale and normalize the depth map for processing
    depth_map = depth_map.astype(np.float32) / depth_scale_factor
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)

    height, width = image.shape[:2]

    # Create coordinate grids for the image pixels
    x_coords = np.tile(np.arange(width), (height, 1))
    y_coords = np.repeat(np.arange(height), width).reshape(height, width)

    # Calculate horizontal displacement based on depth
    displacement = (baseline_distance * (depth_map_normalized * width)).astype(int)

    # Apply displacement to get new coordinates for left and right images
    left_coords_x = np.clip(x_coords - displacement, 0, width - 1)
    right_coords_x = np.clip(x_coords + displacement, 0, width - 1)

    # Index the original image with new coordinates to create the stereo pair
    left_image = image[y_coords, left_coords_x]
    right_image = image[y_coords, right_coords_x]

    # Fill in the gaps caused by displacement with inpainting
    left_image_mask = (left_image == 0).all(axis=2)
    right_image_mask = (right_image == 0).all(axis=2)

    left_image = cv2.inpaint(left_image, left_image_mask.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
    right_image = cv2.inpaint(right_image, right_image_mask.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)

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
