from models.pose.pose_estimator import PoseEstimator
from dlc_implementation.inference import PoseInferenceRunner
import yaml
import numpy as np
from typing import Tuple, List

class PoseDLC (PoseEstimator):

    def __init__(self, path, model_config_path ,conf_thresh:float=0.5,skeleton:List[Tuple]|None=None,device='cpu'):
        super().__init__(skeleton=skeleton)
        if 0 > conf_thresh > 1:
            raise ValueError(f"Parameter `conf_thresh` must be set between 0-1, received: {conf_thresh}")
        self.conf_thresh = conf_thresh
        self.context['path'] = path

        with open(model_config_path, 'r') as file:
            self.model_config = yaml.safe_load(file)

        self.context['bodyparts'] = self.model_config['metadata']['bodyparts']

        self.model = PoseInferenceRunner(model_config=self.model_config,
                                         model_snapshot_path=path,
                                         device=device)

    def prepare_inputs(self, inputs, frame):
        conf, x1, y1, x2, y2 = inputs[0]
        context = {
            'bboxes': np.array([[x1, y1, x2 - x1, y2 - y1]], dtype=np.float32),
            'bbox_scores': np.array([conf], dtype=np.float32)
        }

        pose_inputs = frame, context
        return pose_inputs

    def predict(self, inputs):
        img, context = inputs
        pose_predictions = self.model.inference(context, img)
        return pose_predictions

    def prepare_outputs(self, output):
        return output[0]['bodyparts']