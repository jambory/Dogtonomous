from models.model import Model
from typing import Tuple, List

class PoseEstimator (Model):

    def __init__(self, skeleton:bool=True):
        self.model_type = 'pose'
        self.context = {}
        self.skeleton = curr_skeleton if skeleton else None
        self.conf_thresh: float

    def prepare_inputs(self, inputs, frame):
        """
        Abstract method for preparing inputs for model.
        """
        raise NotImplemented

    def predict(self, inputs):
        """
        Abstract method for predict on inputs of model.
        """
        raise NotImplemented

    def prepare_outputs(self, output):
        """
        Abstract method for formatting/preparing outputs of model.
        """
        raise NotImplemented

# Skeleton of newbodyparts pose project
curr_skeleton = [
            (0,1),
            (1,2),
            (2,26),
            (26,6),
            (26,13),

            (3,4),
            (4,5),
            (5,27),
            (27,6),
            (27,13),

            (6,25),
            (25,7),
            (7,8),
            (8,9),
            (9,11),
            (8,10),
            (10,12),

            (13,24),
            (6,23),

            (24,28),
            (28,14),
            (23,29),

            (15,16),
            (16,17),
            (17,18),
            (18,28),
            (22,28),
            (18,14),

            (19,20),
            (20,21),
            (21,22)
        ]