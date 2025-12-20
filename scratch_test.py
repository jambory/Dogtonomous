from video.prerecorded import PreRecorded
from models.modelstack import ModelStack
from models.pose import *
from models.detectors import *
from models.classifiers import *
import dotenv
import torch
import os

dotenv.load_dotenv()

vid_path = os.environ.get('TEST_VIDEO')
detector_path = os.environ.get('DETECTOR_MODEL_PATH')
pose_path = os.environ.get('POSE_MODEL_PATH')
pose_config_path = os.environ.get('POSE_MODEL_CONFIG_PATH')
classifier_path = os.environ.get('CLASSIFIER_MODEL_PATH')

# Loading classifier model

classifier_model = KeypointMLPDeeper(num_keypoints=30, out_dim=3)
classifier_model.load_state_dict(torch.load(classifier_path, weights_only=True))

modelstack = ModelStack([DetectorYOLO(detector_path), PoseDLC(path=pose_path, model_config_path=pose_config_path,device='mps'),
                         ClassifierTorch1Model(model=classifier_model,device='mps')])

prerecorded = PreRecorded(vid_path, modelstack, 'Model Testing')
prerecorded.run()