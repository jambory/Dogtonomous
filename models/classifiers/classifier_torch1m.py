from models.classifiers.classifier import Classifier
from torch import nn
import torch
from typing import Tuple,List
import numpy as np

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
        keypoints = inputs[0]
        if self.normalize:
            keypoints = self.translate_around_center(6, keypoints)
        keypoints, mask = self._prepare_keypoints(keypoints,fill_value=0.0)

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

    def translate_around_center(self, center_bodypart_idx, kp_array):
        center_body_part_kp = kp_array[center_bodypart_idx][:2]
        center_body_part_conf = kp_array[center_bodypart_idx][2]
        if np.isnan(center_body_part_kp).any() or (center_body_part_conf < self.conf_thresh):
            center_body_part_kp = np.nanmean(kp_array[:, :2], axis=0)

        kp_array[:, :2] = kp_array[:, :2] - center_body_part_kp

        min_h, max_h = np.nanmin(kp_array[:, 1]), np.nanmax(kp_array[:, 1])
        min_w, max_w = np.nanmin(kp_array[:, 0]), np.nanmax(kp_array[:, 0])
        h = max_h - min_h
        w = max_w - min_w

        kp_array[:, 0] /= w
        kp_array[:, 1] /= h

        return kp_array



