import cv2
import socketio
import base64
import time
import asyncio

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
    rtt = (time.time() - data['timestamp_sent']) * 1000
    print(f"Frame {data['frame_id']} | Latency: {rtt:.2f}ms | Detections: {len(data['detections'])}")

async def send_frames():
    global frame_id_counter
    cap = cv2.VideoCapture(0) # Open Webcam
    
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Pre-process: Resize to 640x640 for YOLO
        frame_resized = cv2.resize(frame, (640, 640))

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

    cap.release()

async def main():
    await sio.connect(SERVER_URL)
    await send_frames()
    await sio.wait()

if __name__ == "__main__":
    asyncio.run(main())
