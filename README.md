# ObjectDetection

Real-time object detection pipeline for video files and webcams, based on YOLO models. Supports three inference backends: **OpenCV DNN**, **ONNX Runtime**, and **OpenVINO** – with optional hardware acceleration via CoreML (macOS) and XNNPACK (ARM/Linux).

---

## Features

- **Multi-Backend**: Switch between OpenCV DNN, ONNX Runtime and OpenVINO via a single enum
- **Multi-Source**: Process video files (`.mp4`, `.mov`, …) or live webcam streams
- **Hardware Acceleration**: CoreML (Apple Silicon / Neural Engine) and XNNPACK (ARM Linux)
- **YOLO Support**: YOLOE open-vocabulary exports, YOLO v12 raw output and YOLO v26 end-to-end output
- **Video Output**: Processed frames with bounding boxes are written to the configured output file
- **Performance Timing**: Built-in high-resolution timer for total execution time

---

## Supported Platforms

| Platform              | OpenCV DNN | ONNX Runtime | OpenVINO |
|-----------------------|:----------:|:------------:|:--------:|
| macOS (Apple Silicon) | ✅          | ✅ / CoreML   | ❌ auto-disabled |
| macOS (Intel)         | ✅          | ✅            | ✅ when installed |
| Linux (Intel x86_64)  | ✅          | ✅            | ✅ when installed |
| Linux (aarch64)       | ✅          | ✅ / XNNPACK  | ❌ auto-disabled |

---

## Requirements

- **CMake** ≥ 3.10
- **C++17** compiler (clang++ or g++)
- **OpenCV** ≥ 4.x (with `opencv_dnn`, `opencv_tracking`)
- **ONNX Runtime** *(optional, required for the ONNX Runtime backend)*
- **OpenVINO** 2025.x *(optional, enabled by CMake only when an Intel CPU is detected)*

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

OpenVINO is only enabled automatically when CMake detects an Intel CPU
(`GenuineIntel`). On Apple Silicon, ARM Linux, and non-Intel CPUs, CMake
disables `ENABLE_OPENVINO` and does not search for the package.

Install OpenVINO 2025.x and make its CMake package discoverable through
`CMAKE_PREFIX_PATH` or `OpenVINO_DIR`.

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DOpenVINO_DIR=/opt/intel/openvino_2025.x.x/runtime/cmake
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

When ONNX Runtime and local model assets are available, CMake also enables a
real inference test. It runs `Detection_ORT` on `test.mov` and verifies that the
model draws bounding-box output on at least one frame. The test auto-selects
`yoloe-11l-seg.onnx` when present, otherwise `yolo26m.onnx`/`yolo26x.onnx`.
Disable it with `-DENABLE_INFERENCE_TESTS=OFF` or override the inputs with:

```bash
cmake .. \
  -DINFERENCE_TEST_MODEL=/path/to/model.onnx \
  -DINFERENCE_TEST_CLASSES=/path/to/classes.txt \
  -DINFERENCE_TEST_VIDEO=/path/to/video.mov
```

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

The CoreML Execution Provider is configured with `MLProgram` and a local
`.ort_coreml_cache` directory for compiled CoreML artifacts. By default the
provider lets CoreML use all available compute units. To force one supported
CoreML mode for diagnostics, run with `ORT_COREML_COMPUTE_UNITS=CPUAndGPU` or
`ORT_COREML_COMPUTE_UNITS=CPUAndNeuralEngine`. CoreML decides operator
placement, so this enables GPU and Apple Neural Engine paths but does not
guarantee that every model operator runs on the NPU.

---

## Configuration

Runtime parameters are loaded from `config.yaml` when yaml-cpp is available:

```yaml
detection:
  score_threshold: 0.4
  nms_threshold: 0.5

model:
  model_file: "yoloe-11l-seg.onnx"
  class_names_file: "yoloe_classes.txt"
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

## YOLOE Export

YOLOE uses a class vocabulary that is embedded into the exported model. Edit
`yoloe_classes.txt`, then export the ONNX file and matching runtime labels:

```bash
python3 -m pip install ultralytics
python3 export.py
```

The default export writes `yoloe-11l-seg.onnx` and keeps `yoloe_classes.txt`
as the runtime label file. Re-run the export whenever the class list changes.
Use `python3 export.py --nms` only if you want NMS embedded into the ONNX graph;
without it, this C++ application performs NMS after inference.

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
| `yoloe*.onnx`     | OpenCV DNN, ONNX Runtime, OpenVINO | Open-vocabulary classes baked in during export; raw YOLO output or end-to-end output |
| `yolo12m.onnx`    | OpenCV DNN, ONNX Runtime, OpenVINO | Standard YOLO output (NMS required) |
| `yolo26m.onnx`    | ONNX Runtime, OpenVINO, no OpenCV DNN | End-to-end, NMS built-in     |
| `yolo26x.onnx`    | ONNX Runtime, OpenVINO, no OpenCV DNN | End-to-end, larger variant, NMS built-in     |

> **Note:** YOLOE class labels must match the classes used by `export.py`.

---

## Project Structure

```
ObjectDetection/
├── CMakeLists.txt
├── coco.txt                  # COCO class names (80 classes)
├── yoloe_classes.txt         # YOLOE export/runtime classes
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

- **OpenVINO**: Automatically disabled unless CMake detects an Intel CPU.
- **Tracker**: The `Tracker` module is stubbed out and not integrated into the main loop.
- **OpenCV DNN + yolo26**: The detection branch for this combination is not yet implemented (`// in progress`).

---

## License

See [LICENSE](LICENSE) for details.
