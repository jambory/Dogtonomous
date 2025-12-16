import cv2
import numpy as np
import os

class Base:
    """

    """
    def __init__(self, video: str|int, name: str|None = None):
        self.cap: cv2.VideoCapture = cv2.VideoCapture(video)
        self.frame_n: int = -1 # Initialize the frame number to -1 to signify no frames have been read yet.
        # self.frame: cv2.Mat | np.ndarray | None  = None

        self.width: int = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height: int = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps: int = int(self.cap.get(cv2.CAP_PROP_FPS))
        if name is None:
            if type(video) == int:
                self.name = f"Live Feed Device: {video}"
            else:
                self.name = os.path.basename(video)
        else:
            self.name = name

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            print("No frame detected...")
            return None
        self.frame_n += 1
        return frame

    def display_frame(self,frame:cv2.Mat|None,w_plt=False):
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
        self.cap.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()
    vid_path = os.environ.get('TEST_VIDEO')
    base = Base(vid_path)
    frame = base.read()
    base.display_frame(frame)
    base.release()
