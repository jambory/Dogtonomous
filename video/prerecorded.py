from video.base import Base
import cv2
from typing import List
from models.modelstack import ModelStack

class PreRecorded (Base):

    def __init__(self, video: str, modelstack: ModelStack|None=None,name: str|None=None):
        super().__init__(video=video, name=name, modelstack=modelstack)
        self.n_total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.models = modelstack

    def set_frame(self,frame_n:int):
        if (frame_n<0) or (frame_n>=self.n_total_frames):
            raise Exception(f"Invalid frame_n: {frame_n}")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_n)
        self.frame_n = frame_n

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
    vid_path = os.environ.get('TEST_VIDEO')

    prerecorded = PreRecorded(vid_path)
    prerecorded.run()

