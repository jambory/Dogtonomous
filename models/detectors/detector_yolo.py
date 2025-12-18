from models.detectors.detector import Detector
from ultralytics import YOLO
import numpy as np
from pathlib import Path

class DetectorYOLO(Detector):

    def __init__(self, path:str|Path, conf_thresh:float=0.5):
        super().__init__()
        if  0 > conf_thresh > 1:
            raise ValueError(f"Parameter `conf_thresh` must be set between 0-1, received: {conf_thresh}")
        self.conf_thresh= conf_thresh
        self.context['path'] = path

        self.model = YOLO(path)

    def predict(self, inputs):
        # Predict with yolo model, using tracking since it should be on a continuous video.
        outputs = self.model.track(inputs, persist=True)
        return outputs

    def prepare_outputs(self, output):
        outputs = []

        # Extract bounding boxes, classes, and confidence values
        r = output[0]
        for box, cls, conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf):
            conf = conf.item()
            if conf > 0.5:
                x1, y1, x2, y2 = box.astype(np.float32)
                outputs.append(np.array([conf, x1, y1, x2, y2]))

        return outputs

if __name__ == "__main__":
    import dotenv
    import os

    dotenv.load_dotenv()
    path = os.environ.get('DETECTOR_MODEL_PATH')
    detector = DetectorYOLO(path=path)