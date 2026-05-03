# Cloud-Inference Implementation Plan: Dogtonomous

This document outlines the strategy for moving the YOLO and DeepLabCut (DLC) inference pipeline from a local environment to AWS, using a Raspberry Pi as the edge capture device.

## 1. Architectural Overview
The system follows a **Thin Client / Thick Server** model.
*   **Edge (Raspberry Pi):** Captures video, resizes/compresses frames, and sends them via WebSocket.
*   **Cloud (AWS EC2):** Receives frames, runs GPU-accelerated inference, and returns JSON metadata.

## 2. Infrastructure Requirements (AWS)
*   **Instance Type:** `g4dn.xlarge` (NVIDIA T4 GPU) or `g5.xlarge` (NVIDIA A10G).
*   **AMI:** Deep Learning OSS AMI (Ubuntu 22.04).
*   **Networking:** Port 8000/8080 open for WebSocket traffic.
*   **Location:** Deploy in the AWS Region geographically closest to the physical location of the Pi (e.g., `us-west-2` for West Coast).

## 3. Communication Protocol: WebSockets
To balance low latency with implementation simplicity, we will use **WebSockets** via `FastAPI` and `python-socketio`.

### Data Schema (Upstream)
```json
{
    "frame_id": 12345,
    "timestamp": 1625091234.56,
    "data": "base64_encoded_jpeg_string"
}
```

### Data Schema (Downstream)
```json
{
    "frame_id": 12345,
    "detections": [...],
    "pose_keypoints": [...],
    "latency_ms": 45.2
}
```

## 4. Implementation Phases

### Phase 1: AWS Environment Setup
1.  Launch G4dn instance and install dependencies (`torch`, `ultralytics`, `opencv`).
2.  Port the existing `DetectorYOLO` and `PoseDLC` classes to the cloud environment.
3.  Test local inference on the GPU using a sample video file to establish a performance baseline.

### Phase 2: Cloud Inference Server (`cloud_server.py`)
1.  Build a FastAPI application.
2.  Load models into GPU memory on startup (Warm-up).
3.  Implement a WebSocket endpoint that:
    *   Decodes incoming JPEG bytes.
    *   Runs `DetectorYOLO.predict()`.
    *   Passes boxes to `PoseDLC.predict()`.
    *   Returns the results immediately.

### Phase 3: Raspberry Pi Client (`pi_client.py`)
1.  Initialize camera using `OpenCV`.
2.  Implement a loop targeting **10 FPS** to ensure network stability.
3.  Resize frames to 640x640 (standard YOLO input) before encoding to JPEG.
4.  Track "Round-Trip Time" (RTT) for every frame to monitor real-time latency.

## 5. Latency Mitigation Strategies
*   **JPEG Compression:** Use a quality setting of 50-70 to reduce payload size without sacrificing detection accuracy.
*   **Resolution Scaling:** Only send the resolution required by the model (640px), not the full camera resolution.
*   **Async Processing:** Ensure the Pi captures the *next* frame while waiting for the *previous* result to arrive (Pipelining).

## 6. Success Metrics
*   **Target Latency:** < 150ms total round-trip.
*   **Consistency:** Jitter (variation in latency) < 20ms.
*   **Accuracy:** Cloud results must match local inference results within a 1% confidence margin.

## 7. Security Considerations
*   **Authentication:** Use a simple API Key/Token in the WebSocket header to prevent unauthorized use of the GPU instance.
*   **Data Retention:** Raw video frames should be processed in memory and never written to disk on the AWS instance to maintain privacy.
