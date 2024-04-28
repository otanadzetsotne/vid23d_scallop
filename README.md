# vid23d Scallop

## Introduction
`vid23d_scallop` is a Python application designed to transform 2D video frames into 3D stereo images using depth estimation and stereo pair generation. This project leverages deep learning models for depth perception and computer vision techniques to create enhanced visual experiences.

## Features
* Depth map estimation from 2D images.
* Generation of stereo image pairs from single frames.
* Video processing to extract frames and concatenate them back to video format.
* Audio extraction and merging with processed video.

## Prerequisites
Before you begin, ensure you have the following installed:
* Python 3.8 or higher
* OpenCV
* Torch
* tqdm
* moviepy
* matplotlib (optional, for debugging depth maps)

## Installation
Clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/vid23d_scallop.git
cd vid23d_scallop
```

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

## Usage
Configuring Your Device
Edit `conf.py` to configure the processing device (CPU, CUDA, MPS):

```python
import torch

device = 'cpu'
if torch.backends.mps.is_available():
    device = torch.device('mps')
if torch.cuda.is_available():
    device = torch.device('cuda')

print(f'Using device: {device}', flush=True)
```

## Processing Videos
Use **`main.py`** to process videos through the command line. The script supports various operations including frame extraction, depth map creation, and stereo pair generation.

```bash
python main.py --input_video path/to/your/video.mp4 --output_dir path/to/output
```

## Depth and Stereo Image Generation
`img_depth.py` and `stereo.py` are used internally to generate depth maps and stereo pairs, but can be used standalone for testing and development purposes.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your new features or fixes.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements
Our work uses code from [MiDaS](https://github.com/isl-org/MiDaS). We'd like to thank the authors for making this library available.
