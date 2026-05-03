import os
import cv2
import base64
import numpy as np
import socketio
import uvicorn
from fastapi import FastAPI
from models.detectors.detector_yolo import DetectorYOLO
from models.pose.pose_dlc import PoseDLC
from dotenv import load_dotenv

# Load paths from .env
load_dotenv()
DETECTOR_PATH = os.environ.get('DETECTOR_MODEL_PATH')
POSE_MODEL_PATH = os.environ.get('POSE_MODEL_PATH')
POSE_CONFIG_PATH = os.environ.get('POSE_MODEL_CONFIG_PATH')
device = os.environ.get('DEVICE')

# Initialize Models
print("Loading models into memory...")
detector = DetectorYOLO(path=DETECTOR_PATH, device=device) # Change to 'cuda' if you have a GPU
pose_estimator = PoseDLC(path=POSE_MODEL_PATH, model_config_path=POSE_CONFIG_PATH, device=device)

# Setup FastAPI and Socket.IO
app = FastAPI()
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
combined_asgi_app = socketio.ASGIApp(sio, app)

@sio.event
async def connect(sid, environ):
    print(f"Client connected: {sid}")

@sio.event
async def process_frame(sid, data):
    """
    Receives a frame, runs inference, and returns results.
    data format: {'frame_id': int, 'image': 'base64_string'}
    """
    frame_id = data['frame_id']
    
    # 1. Decode Image
    img_data = base64.b64decode(data['image'])
    nparr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 2. Run Detector
    raw_detections = detector.predict(frame)
    detections = detector.prepare_outputs(raw_detections)

    results = {
        'frame_id': frame_id,
        'detections': [],
        'poses': []
    }

    if len(detections) > 0:
        # For simplicity in the prototype, we process the first detected object
        det = detections[0] # [conf, x1, y1, x2, y2]
        results['detections'].append(det.tolist())

        # 3. Run Pose Estimation
        pose_inputs = pose_estimator.prepare_inputs(detections, frame)
        raw_pose = pose_estimator.predict(pose_inputs)
        pose_data = pose_estimator.prepare_outputs(raw_pose)
        
        # pose_data is usually a dict of {bodypart: [x, y, conf]}
        results['poses'].append(pose_data)

    # 4. Send back results
    await sio.emit('inference_results', results, to=sid)

@sio.event
async def disconnect(sid):
    print(f"Client disconnected: {sid}")

if __name__ == "__main__":
    # Run the server
    uvicorn.run(combined_asgi_app, host="0.0.0.0", port=8000)
