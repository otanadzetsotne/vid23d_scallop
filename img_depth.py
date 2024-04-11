import os
from logging import getLogger, INFO

import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np

from conf import device


'''
Midas model docs
https://pytorch.org/hub/intelisl_midas_v2/
https://github.com/isl-org/MiDaS?tab=readme-ov-file
'''


logger = getLogger(__name__)
logger.setLevel(INFO)


# model_type = 'DPT_Large'      # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
# # model_type = 'DPT_BEiT_L_512'
# # model_type = 'DPT_SwinV2_L_384'
# # model_type = 'DPT_Hybrid'   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
# # model_type = 'MiDaS_small'  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)


models = {}


def init_models(model_type):
    global models

    if model_type not in models:
        # Load model
        midas = torch.hub.load('intel-isl/MiDaS', model_type)
        midas.to(device)
        midas.eval()
        logger.info('Model loaded')

        # Image transforms
        midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
        if model_type == 'DPT_Large' or model_type == 'DPT_Hybrid':
            transform = midas_transforms.dpt_transform
        elif model_type == 'DPT_SwinV2_L_384':
            transform = midas_transforms.swin384_transform
        elif model_type == 'DPT_BEiT_L_512':
            transform = midas_transforms.beit512_transform
        else:
            transform = midas_transforms.small_transform

        models[model_type] = midas, transform

    return models[model_type]


def to_depth(image, model_type):
    midas, transform = init_models(model_type)

    # Predict and resize to original resolution
    with torch.no_grad():
        input_batch = transform(image).to(device)

        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode='bicubic',
            align_corners=False,
        ).squeeze()

    return normalize_to_image_values(prediction.cpu().numpy())


def normalize_to_image_values(data, bit_depth=8):
    if bit_depth == 8:
        norm_data = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)
    elif bit_depth == 16:
        norm_data = ((data - data.min()) / (data.max() - data.min()) * 65535).astype(np.uint16)
    else:
        raise ValueError('Only bit depths of 8 or 16 are supported.')
    # norm_data = ((data - data.min()) / (data.max() - data.min())).astype(np.uint8)
    return norm_data


if __name__ == '__main__':
    IMG_TEST_PATH = os.path.abspath('.output/test.png')

    _, transform_ = init_models('DPT_Large')

    # Generate depth map
    img = cv2.imread(IMG_TEST_PATH)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    output = to_depth(img, 'MiDaS_small')

    # Show image
    plt.imshow(output)
    plt.show()
    logger.info('Done')
