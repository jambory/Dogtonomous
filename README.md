# Dogtonomous

Created by Coby Wilcox, M.S. Student at SFSU.

Dogtonomous is an open-source library dedicated to creating an autonomous device for training dogs. It employs a real-time, 3-deep model stack to detect dogs, estimate their poses, and classify their behaviors.

Initially developed for Raspberry Pi as an edge device, the project is now transitioning to the **Nvidia Jetson Orin Nano** to better handle the computational demands of multi-stage AI inference.

****
![ezgif-35f69ca656feab8a](https://github.com/user-attachments/assets/b9f8da58-cf94-4cd6-8f0b-4c502f4da76b)

## 3-Deep Model Stack

The inference pipeline is structured as a sequential stack of three specialized models:

1.  **Detector:** A fine-tuned **YOLO11n** model that identifies the dog and generates a bounding box whenever it appears on screen.
2.  **Pose Estimator:** A fine-tuned **HRNet** pose estimation model. It utilizes the bounding box from the detector to estimate keypoints on the dog's body. This implementation is based on **DeepLabCut (DLC)** but has been decoupled into `dlc_implementation/` to avoid the overhead of the full DLC library.
3.  **Behavior Classifier:** A simple neural network that processes the estimated keypoints to classify the dog's current behavior (e.g., *sit*, *down*, *paw*).

## Project Structure

### `models/`
Contains the base abstractions for models and the `ModelStack` orchestrator.
- **Detectors:** YOLO-based implementations for dog detection.
- **Pose:** Integration with the DLC-derived HRNet model.
- **Classifiers:** Torch-based neural networks for behavior prediction from keypoints.

### `dlc_implementation/`
A standalone extraction of the DeepLabCut (DLC) inference logic. This allows for lightweight HRNet pose estimation without requiring the full DLC environment, making it more suitable for edge devices.

### `video/`
A module built on OpenCV for efficient video processing. It supports:
- **Live Feed:** Real-time processing from camera devices.
- **Pre-Recorded:** Processing and evaluation of video files.
- **Platform Support:** Includes legacy support for Raspberry Pi (`cap_type="rp"`) and standard OpenCV capture for Jetson/PC.

## Usage Example

The `ModelStack` allows for easy orchestration of the three models:

```python
from models.modelstack import ModelStack
from models.detectors.detector_yolo import DetectorYOLO
from models.pose.pose_dlc import PoseDLC
from models.classifiers.classifier_torch1m import ClassifierTorch1Model
from video.livefeed import LiveFeed

# Initialize the stack
stack = ModelStack([
    DetectorYOLO(detector_path),
    PoseDLC(pose_path, pose_config_path),
    ClassifierTorch1Model(classifier_model)
])

# Run on a live feed
live = LiveFeed(video=0, modelstack=stack)
live.run()
```

## Hardware Transition
While the codebase still contains options for Raspberry Pi, the project is currently focused on the **Nvidia Jetson Orin Nano**. The Raspberry Pi's processing power was insufficient for real-time execution of the full 3-deep model stack, whereas the Jetson Orin Nano provides the necessary GPU acceleration for smooth inference.
