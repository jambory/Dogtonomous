from video.base import Base
import cv2
from typing import List

class LiveFeed(Base):
    """

    """
    def __init__(self, video: int, modelstack: List|None=None, name:str|None=None):
        if type(video)!=int:
            raise Exception(f'`video` must be an integer e.g. 0,1,2.. Got: {video}')
        super().__init__(video=video, name=name)
        self.models = modelstack
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
            frame,output = self.process_frame(frame)
            self.display_frame(frame)
            if cv2.waitKey(int(1000 / self.fps)) & 0xFF == ord('q'):
                break
        self.release()

    def process_frame(self, frame: cv2.Mat):
        if self.models is None:
            return frame, None
        else:
            # TODO: implement model stack and allow frames to be processed.
            return frame, None

    def record(self, file_name: str|None=None):
        if file_name is None:
            import time
            if self.name.startswith("Live Feed Device:"):
                file_name = f"dev{self.device}_{time.time()}.mp4"
            else:
                file_name = f"{self.name}_{time.time()}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(file_name, fourcc, self.fps, (self.width, self.height))
        while True:
            frame = self.read()
            if frame is None:
                print("No videofeed detected...")
                break
            frame,output = self.process_frame(frame)
            self.display_frame(frame)
            writer.write(frame)
            if cv2.waitKey(int(1000 / self.fps)) & 0xFF == ord('q'):
                break
        writer.release()
        self.release()

if __name__ == "__main__":
    import dotenv
    import os

    dotenv.load_dotenv()

    live = LiveFeed(0)
    live.record("test_vid.mp4")