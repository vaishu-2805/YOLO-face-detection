# YOLO Face Detection

This project uses YOLOv1/YOLOv2 models for real-time face detection.

## Features
- Detects faces in images and videos using YOLO models
- Supports multiple model weights: `yolov12l-face.pt`, `yolov12m-face.pt`, `yolov12n-face.pt`, `yolov12s-face.pt`
- Easy to use Python script (`main.py`)

## Requirements
- Python 3.7+
- PyTorch
- OpenCV

## Installation
1. Clone this repository:
   ```pwsh
   git clone https://github.com/yadavnikhil17102004/YOLO-face-detection.git
   cd YOLO-face-detection
   ```
2. Install dependencies:
   ```pwsh
   pip install torch opencv-python
   ```

## Usage
Run the main script with your desired model and input:
```pwsh
python main.py --weights yolov12s-face.pt --source path/to/image_or_video
```
- `--weights`: Path to the YOLO model weights
- `--source`: Path to the input image or video

## Model Weights
- `yolov12l-face.pt`
- `yolov12m-face.pt`
- `yolov12n-face.pt`
- `yolov12s-face.pt`

## Example
```pwsh
python main.py --weights yolov12s-face.pt --source sample.jpg
```

## License
This project is licensed under the MIT License.

## Author

[yadavnikhil17102004](https://github.com/yadavnikhil17102004)
## Acknowledgements
This repository is a fork of the original project by [vaishu-2805](https://github.com/vaishu-2805/YOLO-face-detection).
Customizations and improvements have been made in this version.
