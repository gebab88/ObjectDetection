# ObjectDetection

Real-time object detection pipeline for video files and webcams, based on YOLO models. Supports three inference backends: **OpenCV DNN**, **ONNX Runtime**, and **OpenVINO** – with optional hardware acceleration via CoreML (macOS) and XNNPACK (ARM/Linux).

---

## Features

- **Multi-Backend**: Switch between OpenCV DNN, ONNX Runtime and OpenVINO via a single enum
- **Multi-Source**: Process video files (`.mp4`, `.mov`, …) or live webcam streams
- **Hardware Acceleration**: CoreML (Apple Silicon / Neural Engine) and XNNPACK (ARM Linux)
- **YOLO Support**: YOLO v12 (standard output format) and YOLO v26 (end-to-end, no NMS required)
- **Video Output**: Processed frames with bounding boxes are written to the configured output file
- **Performance Timing**: Built-in high-resolution timer for total execution time

---

## Supported Platforms

| Platform        | OpenCV DNN | ONNX Runtime | OpenVINO |
|-----------------|:----------:|:------------:|:--------:|
| macOS (x86/ARM) | ✅          | ✅ / CoreML on Apple Silicon | ✅        |
| Linux (aarch64) | ✅          | ✅ / XNNPACK   | ❌ (not supported) |

---

## Requirements

- **CMake** ≥ 3.10
- **C++17** compiler (clang++ or g++)
- **OpenCV** ≥ 4.x (with `opencv_dnn`, `opencv_tracking`)
- **ONNX Runtime** *(optional, required for the ONNX Runtime backend)*
- **OpenVINO** 2025.x *(optional, macOS only)*

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

Tests are built when GoogleTest is installed. To let CMake download GoogleTest
into the build directory, configure with `-DFETCH_GTEST=ON`.

### Test Matrix

CI runs the following build-and-test matrix via GitHub Actions:

| Job | OS | Coverage |
|-----|----|----------|
| `linux-minimal` | Ubuntu | OpenCV-only build without optional backends or YAML |
| `linux-yaml` | Ubuntu | YAML config enabled, optional inference backends disabled |
| `linux-default-discovery` | Ubuntu | Default optional backend discovery with YAML available |
| `macos-minimal` | macOS | OpenCV-only build without optional backends or YAML |
| `macos-yaml` | macOS | YAML config enabled, optional inference backends disabled |
| `macos-default-discovery` | macOS | Default optional backend discovery with YAML available |

### Apple Neural Engine / NPU

On Apple Silicon, use the ONNX Runtime backend with CoreML enabled:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_ONNX_RUNTIME=ON \
  -DENABLE_ORT_COREML=ON \
  -DENABLE_OPENVINO=OFF
cmake --build build
```

Then select ONNX Runtime in `config.yaml`:

```yaml
backend:
  framework: "ONNXRuntime"
```

The CoreML Execution Provider is configured with `CPUAndNeuralEngine` and
`MLProgram`. CoreML decides operator placement, so this enables the Apple Neural
Engine path but does not guarantee that every model operator runs on the NPU.

---

## Configuration

Runtime parameters are loaded from `config.yaml` when yaml-cpp is available:

```yaml
detection:
  score_threshold: 0.4
  nms_threshold: 0.5

model:
  model_file: "yolo26x.onnx"
  class_names_file: "coco.txt"
  model_width: 640
  model_height: 640

backend:
  framework: "ONNXRuntime"   # OpenCV | OpenVINO | ONNXRuntime
  use_cuda: false

input:
  source: "VIDEO"            # WEBCAM | VIDEO
  webcam_index: 0
  video_input_file: "test.mov"
  video_output_file: "output.mp4"

display:
  show_frames: false
```

If yaml-cpp or `config.yaml` is not available, the program uses built-in
defaults that match the backends compiled into the binary.

---

## Usage

Place your model file (`.onnx`) and the class names file in the project directory, then run:

```bash
cd build
./ObjectDetection
```

The processed video is saved to `input.video_output_file` from `config.yaml`.
If the requested container/codec cannot be opened, the writer tries compatible
fallback codecs and may use a new `.avi` fallback filename.

### Supported Models

| Model file        | Backend compatibility              | Notes                          |
|-------------------|-------------------------------     |--------------------------------|
| `yolo12m.onnx`    | OpenCV DNN, ONNX Runtime, OpenVINO | Standard YOLO output (NMS required) |
| `yolo26m.onnx`    | ONNX Runtime, OpenVINO, no OpenCV DNN | End-to-end, NMS built-in     |
| `yolo26x.onnx`    | ONNX Runtime, OpenVINO, no OpenCV DNN | End-to-end, larger variant, NMS built-in     |

> **Note:** The OpenCV DNN backend only supports `yolo12m.onnx`. YOLO26 models require ONNX Runtime or OpenVINO in the current build.

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
