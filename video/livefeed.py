from video.base import Base
import cv2
from models.modelstack import ModelStack

class LiveFeed(Base):
    """

    """
    def __init__(self, video: int, modelstack: ModelStack|None=None, name:str|None=None, cap_type:str="cv"):
        if type(video)!=int:
            raise Exception(f'`video` must be an integer e.g. 0,1,2.. Got: {video}')
        super().__init__(video=video, name=name, modelstack=modelstack, cap_type=cap_type)
        self.device=video

    def run(self):
        """
        Begin processing the video stream frame-by-frame.

        Press 'q' to quit the video window.
        """
        while True:
            frame = self.read()
            if frame is None:
                print("No videofeed detected...")
                break
            outputs = self.process_frame(frame)
            self.visualize(frame, outputs)
            self.display_frame(frame)
            if cv2.waitKey(int(1000 / self.fps)) & 0xFF == ord('q'):
                break
        self.release()

if __name__ == "__main__":
    import dotenv
    import os

    dotenv.load_dotenv()

    live = LiveFeed(0)
    live.record("test_vid.mp4")