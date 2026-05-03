import cv2
import socketio
import base64
import time
import asyncio
import numpy as np
from video.livefeed import LiveFeed

# Configuration
SERVER_URL = "http://localhost:8000" # Change to your PC's IP when running on Pi
FRAME_RATE = 10 # Target FPS
SEND_INTERVAL = 1.0 / FRAME_RATE

sio = socketio.AsyncClient()
frame_id_counter = 0

@sio.event
async def connect():
    print("Connected to Cloud Server")

@sio.event
async def inference_results(data):
    # Calculate RTT based on the timestamp we sent
    rtt = (time.time() - data['timestamp_sent']) * 1000
    print(f"Frame {data['frame_id']} | Latency: {rtt:.2f}ms | Detections: {len(data['detections'])}")

async def send_frames():
    global frame_id_counter
    
    # Use the project's LiveFeed class with 'rp' capture type
    # For local testing on PC, you might need to change 'rp' back to 'cv'
    try:
        vid = LiveFeed(0, cap_type="rp")
    except Exception as e:
        print(f"Failed to initialize RP camera: {e}. Falling back to CV.")
        vid = LiveFeed(0, cap_type="cv")

    print(f"Camera initialized. Resolution: {vid.width}x{vid.height} @ {vid.fps} FPS")

    while True:
        start_time = time.time()
        
        frame = vid.read()
        if frame is None:
            break

        # 1. Pre-process: Resize to 640x640 for YOLO if not already
        if vid.width != 640 or vid.height != 640:
            frame_resized = cv2.resize(frame, (640, 640))
        else:
            frame_resized = frame

        # 2. Encode to JPEG
        _, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 70])
        
        # 3. Convert to Base64
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')

        # 4. Send to Server
        payload = {
            'frame_id': frame_id_counter,
            'image': jpg_as_text,
            'timestamp_sent': time.time()
        }
        
        if sio.connected:
            await sio.emit('process_frame', payload)
            frame_id_counter += 1

        # Control FPS
        elapsed = time.time() - start_time
        sleep_time = max(0, SEND_INTERVAL - elapsed)
        await asyncio.sleep(sleep_time)

    vid.release()

async def main():
    try:
        await sio.connect(SERVER_URL)
        await send_frames()
    except Exception as e:
        print(f"Connection Error: {e}")
    finally:
        await sio.wait()

if __name__ == "__main__":
    asyncio.run(main())
