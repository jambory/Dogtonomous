import cv2
import numpy as np
import os
from typing import List
from models.modelstack import ModelStack
from models.classifiers.classifier import Classifier
from models.detectors.detector import Detector
from models.pose.pose_estimator import PoseEstimator

class Base:
    """

    """
    def __init__(self, video: str|int, name: str|None = None, modelstack:ModelStack|None = None, cap_type='cv'):
        self.frame_n: int = -1 # Initialize the frame number to -1 to signify no frames have been read yet.

        if cap_type == "cv":
            self.cap: cv2.VideoCapture = cv2.VideoCapture(video)
            self.width: int = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height: int = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps: int = int(self.cap.get(cv2.CAP_PROP_FPS))
        elif cap_type == "rp":
            from picamera2 import Picamera2
            self.cap = Picamera2()

            config = self.cap.create_video_configuration({"format": "XRGB8888", "size":(640,480)})
            self.cap.configure(config)

            size = config["main"]["size"]
            self.width, self.height = size

            controls = self.cap.camera_controls
            _, _,max_fd = controls["FrameDurationLimits"]
            self.fps = 1_000_000 / max_fd
            self.cap.start()
        else:
            raise ValueError(f"No capture type `{cap_type}`")
        
        self.cap_type = cap_type

        if name is None:
            if type(video) == int:
                self.name = f"Live Feed Device: {video}"
            else:
                self.name = os.path.basename(video)
        else:
            self.name = name

        self.modelstack = modelstack

    def read(self):
        if self.cap_type == "cv":
            ret, frame = self.cap.read()
            if not ret:
                print("No frame detected...")
                return None
        elif self.cap_type == "rp":
            frame = self.cap.capture_array()
        self.frame_n += 1
        return frame

    def display_frame(self,frame:np.ndarray|None,w_plt=False):
        if frame is None:
            raise Exception("No frame to be displayed...")
        if w_plt:
            import matplotlib.pyplot as plt
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(6, 4))
            plt.imshow(frame_rgb)
            plt.axis("off")
            plt.title(f"{self.name} f:{self.frame_n}")
            plt.show()
        else:
            cv2.imshow(winname=self.name, mat=frame)

    def release(self):
        """
        Release video resources and close OpenCV windows.
        """
        if self.cap_type=="cv":
            self.cap.release()
        else:
            self.cap.stop()
        cv2.destroyAllWindows()

    def process_frame(self, frame: np.ndarray):
        if self.modelstack is None:
            return [None]
        else:
            outputs = self.modelstack.run(frame)
            return outputs

    def visualize(self, frame, outputs):
        if self.modelstack is None:
            return
        for output,model in zip(outputs,self.modelstack.models):
            if output is None:
                break
            else:
                match model.model_type:
                    case 'detector':
                        self.visualize_detection(frame, model, output)
                    case 'pose':
                        self.visualize_pose(frame, model, output)
                    case 'classifier':
                        self.visualize_classification(frame, model, output)


    @staticmethod
    def visualize_detection(frame:np.ndarray, model:Detector, output:List):
        """
        Draw YOLO bounding boxes and confidence values on the frame.

        Args:
            frame (np.ndarray): Image to draw on.
            output (list): List of detections [conf, x1, y1, x2, y2].

        Returns:
            np.ndarray: Frame with bounding boxes drawn.
        """
        for detector_prediction in output:
            conf = detector_prediction[0]
            if conf < model.conf_thresh:
                continue
            x1, y1, x2, y2 = detector_prediction[1:].astype(int)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, f"Conf: {conf:.3f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    @staticmethod
    def visualize_pose(frame: np.ndarray, model: PoseEstimator ,output: List):
        bodypart_predictions = output[0]
        conf_thresh = model.conf_thresh
        for bodypart, predictions in zip(model.context["bodyparts"], bodypart_predictions):
            x, y, conf = predictions
            if conf >= conf_thresh:
                cv2.circle(frame, (int(x), int(y)), radius=5, color=(0, 0, 255), thickness=5)
                cv2.putText(
                    frame, bodypart,
                    (int(x + 5), int(y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2
                )

        if model.skeleton is not None:
            drawn = {}
            for connection in model.skeleton:
                bodypart1, bodypart2 = connection
                if (bodypart1 in drawn) and (bodypart2 in drawn):
                    x1, y1 = drawn[bodypart1]
                    x2, y2 = drawn[bodypart2]
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 50, 255), thickness=1)

    @staticmethod
    def visualize_classification(frame: np.ndarray, model: Classifier, output: List):
        """
                        Display predicted behaviors (e.g., sit/down/paw) on the frame.

                        Args:
                            output (list[tuple[str, list]]): List of (behavior_name, prediction).
                            frame (np.ndarray): Image to draw on.

                        Returns:
                            np.ndarray: Frame annotated with classifier outputs.
                        """
        display_text = "P: " + ",".join(
            f"{model_type.capitalize()}:{prediction if type(prediction) == float else prediction[0]:.1f}" for
            model_type, prediction in zip(model.labels, output)
        )
        height, width, _ = frame.shape
        if width < 1000:
            font_scale = 1
            thickness = 1
        else:
            font_scale = 2
            thickness = 4
        cv2.putText(
            frame, display_text,
            (int(width * 0.045), int(height * 0.1)),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            (0, 0, 0), thickness, cv2.LINE_AA
        )


if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()
    vid_path = os.environ.get('TEST_VIDEO')
    base = Base(vid_path)
    frame = base.read()
    base.display_frame(frame)
    base.release()
