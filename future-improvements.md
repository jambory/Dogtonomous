# Future Performance Improvements

This document outlines architectural and logic-level optimizations to further increase the inference speed of the Dogtonomous model stack on the Nvidia Jetson Orin Nano.

## 1. Multiprocessing Pipeline (Producer-Consumer)
**Current State:** Sequential execution (`Read -> Detect -> Pose -> Classify -> Display`). The GPU/CPU sits idle during I/O or other stages.
**Improvement:** Implement a multi-stage pipeline using Python's `multiprocessing`.
- **Process A:** Frame acquisition (GStreamer/OpenCV) and resizing.
- **Process B:** Detector + Pose Estimation (GPU-intensive).
- **Process C:** Behavior Classification and Visualization.
**Goal:** Parallelize I/O and inference to maximize hardware utilization.

## 2. Asynchronous GPU Preprocessing
**Current State:** Image cropping and normalization happen on the CPU via NumPy/OpenCV.
**Improvement:** Move preprocessing to the GPU.
- Use **Nvidia VPI (Vision Programming Interface)** or **`torchvision.transforms`** on CUDA tensors.
- Perform the "Top-Down Crop" logic directly in GPU memory to avoid redundant CPU-GPU copies.
**Goal:** Reduce the CPU bottleneck and minimize memory transfer latency.

## 3. Conditional Pose Estimation (Logic Gating)
**Current State:** The Pose Estimator (the most expensive model) runs on every single frame.
**Improvement:** Use temporal consistency to skip redundant inferences.
- Only run the Pose Estimator every $N$ frames.
- Use a lightweight tracker (like BoT-SORT) to update the bounding box in "skip" frames.
- Gate the Pose Estimator: only re-run if the dog's bounding box moves significantly or if the confidence of the previous pose was low.
**Goal:** Drastically reduce the duty cycle of the HRNet model.

## 4. Unified Memory & Pinned Buffers
**Current State:** Standard PyTorch/NumPy memory management.
**Improvement:** Optimize for Jetson's unified memory architecture.
- Use **Pinned Memory** (`tensor.pin_memory()`) for faster host-to-device transfers.
- Utilize zero-copy buffers where possible to allow the GPU to access CPU-allocated frames without an explicit copy.
**Goal:** Minimize synchronization overhead and memory bandwidth usage.
