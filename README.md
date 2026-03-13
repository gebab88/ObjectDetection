# ObjectDetection

Real-time object detection pipeline for video files and webcams, based on YOLO models. Supports three inference backends: **OpenCV DNN**, **ONNX Runtime**, and **OpenVINO** – with optional hardware acceleration via CoreML (macOS) and XNNPACK (ARM/Linux).

---

## Features

- **Multi-Backend**: Switch between OpenCV DNN, ONNX Runtime and OpenVINO via a single enum
- **Multi-Source**: Process video files (`.mp4`, `.mov`, …) or live webcam streams
- **Hardware Acceleration**: CoreML (Apple Silicon / AMD Radeon via Metal) and XNNPACK (ARM Linux)
- **YOLO Support**: YOLO v12 (standard output format) and YOLO v26 (end-to-end, no NMS required)
- **Video Output**: Processed frames with bounding boxes are written to `output.avi`
- **Performance Timing**: Built-in high-resolution timer for total execution time

---

## Supported Platforms

| Platform        | OpenCV DNN | ONNX Runtime | OpenVINO |
|-----------------|:----------:|:------------:|:--------:|
| macOS (x86/ARM) | ✅          | ✅ /w CoreML   | ✅        |
| Linux (aarch64) | ✅          | ✅ /W XNNPACK  | ❌ (not supported) |

---

## Requirements

- **CMake** ≥ 3.10
- **C++17** compiler (clang++ or g++)
- **OpenCV** ≥ 4.x (with `opencv_dnn`, `opencv_tracking`)
- **ONNX Runtime** (local installation, see below)
- **OpenVINO** 2025.x *(macOS only)*

### Install OpenCV (macOS)
```bash
brew install opencv
```

### Install OpenCV (Linux / ARM)
```bash
sudo apt install libopencv-dev
```

### ONNX Runtime

Download the prebuilt binaries for your platform from the [official releases](https://github.com/microsoft/onnxruntime/releases) and place them in `onnxruntime/` inside the project root:

```
ObjectDetection/
└── onnxruntime/
    ├── include/
    │   └── onnxruntime/core/session/
    │       ├── onnxruntime_cxx_api.h
    │       ├── cpu_provider_factory.h
    │       └── coreml_provider_factory.h   # macOS only
    ├── build/MacOS/Release/
    │   └── libonnxruntime.dylib            # macOS
    └── lib/
        └── libonnxruntime.so               # Linux / ARM
```

### OpenVINO (Intel only)

Install OpenVINO 2025.x to `/opt/intel/openvino_2025.x.x/`. The CMakeLists.txt expects this path by default.

```bash
# Adjust the path in CMakeLists.txt if your installation differs:
set(OpenVINO_DIR "/opt/intel/openvino_2025.x.x/runtime/cmake")
```

---

## Build

```bash
git clone <repo-url>
cd ObjectDetection

mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -- -j$(nproc)
```

The resulting binary is `ObjectDetection` inside the `build/` directory.

---

## Configuration

All runtime parameters are currently set as constants at the top of `src/main.cpp`:

```cpp
const float SCORE_THRESHOLD  = 0.4f;      // Minimum confidence score
const float NMS_THRESHOLD    = 0.5f;      // Non-Maximum Suppression threshold (OpenCV backend only)
const cv::Size2f MODELSHAPE  = {640, 640};// Model input resolution

const int  FILE_OR_WEBCAM    = 1;         // 1 = video file, 2 = webcam
const bool SHOW_FRAMES       = false;     // Display frames in a window during processing

const std::string VIDEO_INPUT_FILE  = "./test.mov";
const std::string CLASS_NAMES_FILE  = "./coco.txt";
const std::string MODEL_FILE        = "./yolo26x.onnx";

const bool USE_CUDA = false;              // CUDA backend (OpenCV DNN only)

FRAMEWORK backend = FRAMEWORK::ONNXRuntime; // OpenCV | ONNXRuntime | OpenVINO
```

---

## Usage

Place your model file (`.onnx`) and the class names file in the project directory, then run:

```bash
cd build
./ObjectDetection
```

The processed video is saved to `output.avi` in the working directory.

### Supported Models

| Model file        | Backend compatibility              | Notes                          |
|-------------------|-------------------------------     |--------------------------------|
| `yolo12m.onnx`    | OpenCV DNN, ONNX Runtime, OpenVINO | Standard YOLO output (NMS required) |
| `yolo26m.onnx`    | ONNX Runtime, OpenVINO, no OpenCV DNN | End-to-end, NMS built-in     |
| `yolo26x.onnx`    | ONNX Runtime, OpenVINO, no OpenCV DNN | End-to-end, larger variant, NMS built-in     |

> **Note:** The OpenCV DNN backend only supports `yolo12m.onnx`. Using `yolo26*.onnx` with OpenCV DNN will fall into the unimplemented `else` branch.

---

## Project Structure

```
ObjectDetection/
├── CMakeLists.txt
├── coco.txt                  # COCO class names (80 classes)
├── coco.yaml                 # COCO dataset config
├── include/
│   ├── Detection.hpp         # Abstract base class
│   ├── Detection_OpenCV.hpp
│   ├── Detection_ORT.hpp
│   ├── Detection_OpenVINO.hpp
│   ├── VideoHandler.hpp
│   ├── Tracker.hpp
│   ├── Timer.hpp
│   └── Framework.hpp         # Backend enum
└── src/
    ├── main.cpp
    ├── Detection.cpp
    ├── Detection_OpenCV.cpp
    ├── Detection_ORT.cpp
    ├── VideoHandler.cpp
    ├── Tracker.cpp
    ├── Timer.cpp
    └── macos/
        └── Detection_OpenVINO.cpp
```

---

## Class Overview

| Class               | Responsibility                                               |
|---------------------|--------------------------------------------------------------|
| `Detection`         | Abstract base class; loads class names, holds shared state   |
| `Detection_OpenCV`  | OpenCV DNN backend; CUDA support optional                    |
| `Detection_ORT`     | ONNX Runtime backend; CoreML (macOS) and XNNPACK (ARM) EP   |
| `Detection_OpenVINO`| OpenVINO backend *(macOS only)*                              |
| `VideoHandler`      | Opens video/webcam, crops to square, writes output video     |
| `Timer`             | High-resolution wall-clock timer using `std::chrono`         |
| `Tracker`           | (In progress) OpenCV object tracker wrapper                  |

---

## Known Limitations / TODO

- **Linux + OpenVINO**: Not yet supported. OpenVINO is only compiled on macOS.
- **Tracker**: The `Tracker` module is stubbed out and not integrated into the main loop.
- **Configuration**: Parameters are currently hardcoded in `main.cpp`; CLI argument support is planned.
- **OpenCV DNN + yolo26**: The detection branch for this combination is not yet implemented (`// in progress`).

---

## License

See [LICENSE](LICENSE) for details.
