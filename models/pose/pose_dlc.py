from models.pose.pose_estimator import PoseEstimator

class PoseDLC (PoseEstimator):

    def __init__(self, path, conf_thresh:float=0.5):
        super().__init__()
        if 0 > conf_thresh > 1:
            raise ValueError(f"Parameter `conf_thresh` must be set between 0-1, received: {conf_thresh}")
        self.conf_thresh = conf_thresh
        self.context['path'] = path

        self.model =
