from models.classifiers.classifier import Classifier
from torch import nn
import torch
from typing import Tuple,List

class ClassifierTorch1Model(Classifier):

    def __init__(self, model:nn.Module, conf_thresh:float=0.5,labels:Tuple|List=('sit','down','paw'),
                 normalize:bool=True,device:str="cpu"):
        super().__init__()
        self.context = {'model_type': str(type(model))}
        self.conf_thresh = conf_thresh
        self.device = device
        model.to(device)
        model.eval()
        self.model = model
        self.normalize = normalize
        self.labels = labels

    def prepare_inputs(self, inputs, frame):
        keypoints, mask = self._prepare_keypoints(inputs[0],fill_value=0.0)
        if self.normalize:
            h, w = frame.shape[:2]
            keypoints[:, 0] /= w
            keypoints[:, 1] /= h

        return keypoints.to(self.device), mask.to(self.device)

    def _prepare_keypoints(self, np_keypoints, fill_value=0.0):
        arr = torch.from_numpy(np_keypoints).float()  # (K, 3)
        coords = arr[:, :2]  # (K, 2)
        conf = arr[:, 2]  # (K,)

        mask = (conf > self.conf_thresh).float()  # (K,)

        coords_masked = coords.clone()
        coords_masked[mask == 0] = fill_value

        return coords_masked, mask

    def predict(self, inputs):
        kp, mask = inputs
        classifier_outputs = self.model(kp.unsqueeze(0), mask.unsqueeze(0))
        classifier_outputs = torch.sigmoid(classifier_outputs)

        return classifier_outputs.squeeze(0).tolist()

    def prepare_outputs(self, output):
        return output



