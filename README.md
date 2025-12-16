# Dogtonomous

Created by Coby Wilcox, M.S. Student at SFSU.

Open source library for creating an autonomous device to train your dog! Built upon a top-down pose estimation model made with Deeplabcut, YOLO11s and, with a simple neural network for classifying the poses.

****

## Structure

### Video

The video module allows for easy processing of live or prerecorded videos for inference, data-prep, and analysis. Meant to provide clean focused implementations of opencv-python video processors without much of the boilerplate.

**Base**

The base super class that all video classes will generally inherit from. 

Functionality:
- Read frames.
- Get frame dimensions.
- Store frame number.
- Display current frame.

**Live**

The live video class, can take in a ModelStack to do inference with, or simply display a live feed of video.

Functionality:

- Perform inference from ModelStack.
- Optionally record live video and model outputs.
- Optionally visualize outputs of model inference.

**PreRecorded**

The video class when reading from a video file. Allows for inference, evaluation, or quickly accessing specific frames and displaying them.

Functionality:

- Move to a specific frame in a video.
- Perform inference on the video optionally visualizing the outputs.
- With given ground truth labels perform evaluation on the video.



****

### Inference

The inference module allows for quick and easy model stack implementations and experimentation. When combined with videoprocessing, allows users to see exactly what their models are predicting. The inference module implementations are meant to be as lightweight as possible to allow for efficient live prediction.

### Evaluation

Made to extend the inference module into a easy solution for objective evaluation of each model in a model stack. 

### Models

Meant to store the underlying structure of the models used for easy calling and analysis. The models classes themselves contain much of the implementations of things like inference and evaluation to allow for easy changing to the structure of the model inputs and outputs without breaking the loops. 

To avoid to much overhead the model classes have defined modes similar to PyTorch model implementations. You can change the model's mode on the fly and therefore, extend what it can do.

