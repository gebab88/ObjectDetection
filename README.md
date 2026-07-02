# ObjectDetection

Real-time object detection pipeline for video files and webcams, based on YOLO models. Supports three inference backends: **OpenCV DNN**, **ONNX Runtime**, and **OpenVINO** – with optional hardware acceleration via CoreML (macOS) and XNNPACK (ARM/Linux).

---

## Features

- **Multi-Backend**: Switch between OpenCV DNN, ONNX Runtime and OpenVINO via `config.yaml` (a detector factory validates the choice against the model format)
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

Homebrew:
```bash
brew install opencv
```

Or MacPorts (installs under `/opt/local`, which CMake already searches):
```bash
sudo port install opencv4
```

### Install OpenCV (Linux / ARM)
```bash
sudo apt install libopencv-dev
```

### ONNX Runtime

You can either use the **prebuilt binaries** or **build ONNX Runtime from source**.
Either way, CMake discovers it through `ORT_DIR` (default: `onnxruntime/` in the
project root; override with `-DORT_DIR=/path/to/onnxruntime`). It expects the
headers under `<ORT_DIR>/include` and the shared library under
`<ORT_DIR>/build/MacOS/Release` (macOS) or `<ORT_DIR>/lib` (macOS and Linux/ARM).

#### Option A — prebuilt binaries

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

#### Option B — build from source

Clone and build the shared library (see the upstream [build instructions](https://onnxruntime.ai/docs/build/) for all options):

```bash
git clone --recursive https://github.com/microsoft/onnxruntime
cd onnxruntime
# Linux / ARM:
./build.sh --config Release --build_shared_lib --parallel --skip_tests
# macOS (adds the CoreML execution provider):
./build.sh --config Release --build_shared_lib --parallel --skip_tests --use_coreml
```

A source checkout already ships the public headers under
`include/onnxruntime/core/...`, and the build writes the shared library to
`build/<Platform>/Release/`. Make both discoverable in one of two ways:

- **Point `ORT_DIR` at the source checkout** (simplest on macOS, whose expected
  layout matches the source tree 1:1):
  ```bash
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DORT_DIR=/path/to/onnxruntime
  ```
  On macOS this finds `build/MacOS/Release/libonnxruntime.dylib` and the headers
  directly. On **Linux/ARM** the library lands in `build/Linux/Release/`, which is
  *not* searched — symlink it into `<ORT_DIR>/lib/` first (see the next option).

- **Assemble the documented `onnxruntime/` layout** by symlinking (or copying) the
  source artifacts into place. From the project root, with `SRC` pointing at your
  ONNX Runtime source checkout:
  ```bash
  mkdir -p onnxruntime/include onnxruntime/lib
  ln -s "$SRC/include/onnxruntime" onnxruntime/include/onnxruntime
  # macOS:
  ln -s "$SRC/build/MacOS/Release/libonnxruntime.dylib" onnxruntime/lib/
  # Linux / ARM:
  ln -s "$SRC/build/Linux/Release/libonnxruntime.so"    onnxruntime/lib/
  ```
  Then configure as usual (the default `ORT_DIR` already points at `onnxruntime/`).

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
real inference test. It runs `Detection_ORT` on every video found in `tests/`
(`tests/*.MOV`, `*.mov`, `*.mp4`) — one parameterized test case per video — and
verifies that the model draws bounding-box output on at least one frame. The test
auto-selects `yoloe-11l-seg.onnx` when present, otherwise `yolo26m.onnx`/`yolo26x.onnx`.

> **Note:** The inference tests require you to place one or more test videos in
> the `tests/` directory (e.g. `tests/my_clip.MOV`). These videos are **not**
> tracked in Git, so you must add your own. If `tests/` contains no video, CMake
> skips the inference test (a message says so at configure time); the rest of the
> build and the unit tests are unaffected.

Disable the test with `-DENABLE_INFERENCE_TESTS=OFF`, or override the inputs with:

```bash
cmake .. \
  -DINFERENCE_TEST_MODEL=/path/to/model.onnx \
  -DINFERENCE_TEST_CLASSES=/path/to/classes.txt \
  -DINFERENCE_TEST_VIDEO=/path/to/video.mov   # single-video override
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

The CoreML Execution Provider uses the `MLProgram` model format and caches the
compiled CoreML model in a local `.ort_coreml_cache/` directory, so the
(slow) first-run compilation is reused on later runs.

By default, CoreML is allowed to use **all** compute units — CPU, GPU and the
Apple Neural Engine (ANE) — and picks where to run each operator itself. You can
restrict which units it may use via the `ORT_COREML_COMPUTE_UNITS` environment
variable, mainly for diagnostics:

```bash
ORT_COREML_COMPUTE_UNITS=CPUAndGPU            ./build/ObjectDetection  # CPU + GPU only
ORT_COREML_COMPUTE_UNITS=CPUAndNeuralEngine   ./build/ObjectDetection  # CPU + ANE only
```

> **Note:** This variable sets which compute units CoreML *may* use, not where
> each operator actually runs. CoreML still decides operator placement, and any
> operator it cannot map to the GPU/ANE falls back to the CPU. So even
> `CPUAndNeuralEngine` does not guarantee the whole model runs on the Neural
> Engine — it only enables that path.

---

## Configuration

Runtime parameters are loaded from `config.yaml` when yaml-cpp is available. A
documented template is version-controlled as `config.example.yaml` – copy it to
`config.yaml` and adjust:

```bash
cp config.example.yaml config.yaml
```

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
  video_input_file: "tests/test_mov1.MOV"
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
./build/ObjectDetection                       # auto-find config.yaml (./ then ../)
./build/ObjectDetection --config path/to/config.yaml
```

Command-line options:

| Option | Description |
|--------|-------------|
| `-c`, `--config <path>` | Path to `config.yaml`. If omitted, it is searched in the current directory and one level up (so running from `build/` finds the repo-root config). |
| `-h`, `--help` | Show usage. |

**Path resolution:** relative paths inside `config.yaml` (`model_file`,
`class_names_file`, `video_input_file`, `video_output_file`) are resolved
against the config file's directory, so the program can be started from any
working directory (e.g. `build/`). Absolute paths are used as-is.

The processed video is saved to `input.video_output_file` from `config.yaml`.
If the requested container/codec cannot be opened, the writer tries compatible
fallback codecs and may use a new `.avi` fallback filename.

### Supported Models

The pipeline supports three YOLO output formats, detected from the model
**filename** (`detectModelFormat`): **YOLO12** (raw grid output), **YOLO26**
(end-to-end with built-in NMS), and **YOLOE** (open-vocabulary; raw or
end-to-end depending on the export).

| Model file        | Backend compatibility              | Notes                          |
|-------------------|-------------------------------     |--------------------------------|
| `yoloe*.onnx`     | OpenCV DNN, ONNX Runtime, OpenVINO | Open-vocabulary classes baked in during export; raw YOLO output or end-to-end output |
| `yolo12m.onnx`    | OpenCV DNN, ONNX Runtime, OpenVINO | Standard YOLO output (NMS required) |
| `yolo26m.onnx`    | ONNX Runtime, OpenVINO, no OpenCV DNN | End-to-end, NMS built-in     |
| `yolo26x.onnx`    | ONNX Runtime, OpenVINO, no OpenCV DNN | End-to-end, larger variant, NMS built-in     |

> **Note:** YOLOE class labels must match the classes used by `export.py`.

#### Tested with

The pipeline has been run with the following exported models:

| Model                    | Format | Notes                                   |
|--------------------------|--------|-----------------------------------------|
| `yolo26m.onnx`           | YOLO26 | End-to-end                              |
| `yolo26x.onnx`           | YOLO26 | End-to-end, larger variant              |
| `yoloe-11l-seg.onnx`     | YOLOE  | Open-vocabulary                         |
| `yoloe-26m-seg.onnx`     | YOLOE  | Open-vocabulary                         |
| `yoloe-26m-seg-pf.onnx`  | YOLOE  | Prompt-free; covered by the automated inference test |
| `yoloe-26l-seg-pf.onnx`  | YOLOE  | Prompt-free                             |
| `yoloe-26x-seg-pf.onnx`  | YOLOE  | Prompt-free, larger variant             |
| `yoloe-26x-seg.onnx`     | YOLOE  | Open-vocabulary, larger variant         |

The `-seg` models are segmentation exports, but only their bounding boxes are
used (see Known Limitations). `-pf` denotes the prompt-free YOLOE variant.

---

## Project Structure

```
ObjectDetection/
├── CMakeLists.txt
├── config.example.yaml       # Sample runtime config (copy to config.yaml)
├── export.py                 # YOLOE ONNX export helper
├── coco.txt                  # COCO class names (80 classes)
├── coco.yaml                 # COCO dataset config
├── yoloe_classes.txt         # YOLOE export/runtime classes
├── yoloe_pf_classes.txt      # YOLOE prompt-free runtime classes
├── include/
│   ├── Detection.hpp         # Abstract base class
│   ├── Detection_OpenCV.hpp
│   ├── Detection_ORT.hpp
│   ├── Detection_OpenVINO.hpp
│   ├── DetectorFactory.hpp   # Backend selection + createDetector()
│   ├── VideoHandler.hpp
│   ├── FrameCrop.hpp         # Centered-square crop helper
│   ├── YoloPostprocess.hpp   # Decoders (raw / end-to-end) + drawing
│   ├── config_loader.hpp     # Config struct + YAML loading + path resolution
│   ├── Tracker.hpp
│   ├── Timer.hpp
│   ├── Framework.hpp         # Backend / model-format enums + helpers
│   └── Source.hpp            # Input source enum
├── src/
│   ├── main.cpp
│   ├── Detection.cpp
│   ├── Detection_OpenCV.cpp
│   ├── Detection_ORT.cpp
│   ├── DetectorFactory.cpp
│   ├── VideoHandler.cpp
│   ├── Tracker.cpp
│   ├── Timer.cpp
│   └── openvino/
│       └── Detection_OpenVINO.cpp
└── tests/
    ├── test_framework.cpp    # Unit tests (format, decoders, config, ...)
    ├── test_inference.cpp    # Real ONNX inference test over tests/ videos
    └── *.MOV                 # Local, git-ignored test videos
```

---

## Class Overview

| Class               | Responsibility                                               |
|---------------------|--------------------------------------------------------------|
| `Detection`         | Abstract base class; loads class names, holds shared state   |
| `Detection_OpenCV`  | OpenCV DNN backend; CUDA support optional                    |
| `Detection_ORT`     | ONNX Runtime backend; CoreML (macOS) and XNNPACK (ARM) EP   |
| `Detection_OpenVINO`| OpenVINO backend *(Intel CPUs only; auto-disabled otherwise)* |
| `DetectorFactory`   | `createDetector()` builds the configured backend; `supportedBackendsFor()` encodes format↔backend compatibility (free functions) |
| `YoloPostprocess`   | Header-only decoders (raw / end-to-end + `decodeByFormat` dispatch), NMS and box/label drawing |
| `VideoHandler`      | Opens video/webcam, crops to square, writes output video     |
| `Timer`             | High-resolution wall-clock timer using `std::chrono`         |
| `Tracker`           | **Stub** – thin `track()` wrapper around an OpenCV tracker; not wired into the pipeline (see Known Limitations) |

---

## Known Limitations / TODO

- **Segmentation**: The `-seg` YOLOE models are segmentation exports, but the
  pipeline is detection-only — it draws bounding boxes and labels and discards
  the mask coefficients. The decoders (`YoloPostprocess.hpp`) intentionally skip
  the mask fields. Rendering the instance masks is left as future work.
- **OpenVINO**: Automatically disabled unless CMake detects an Intel CPU.
- **Tracker**: `src/Tracker.cpp` is an intentional stub. It exposes a single
  free function, `track(frame, tracker, trackingBox)`, that updates an OpenCV
  tracker and draws the tracked box, but nothing calls it — the main loop runs
  per-frame detection only, without object tracking across frames. The file is
  compiled so the stub stays build-checked; integrating it (instantiating a
  tracker, seeding it from a detection, and calling `track()` in the frame loop)
  is left as future work.
- **OpenCV DNN + yolo26**: Not supported. OpenCV's DNN module cannot load the
  end-to-end YOLO26 graph, so `main()` does not offer the OpenCV backend for
  YOLO26 models. Use the ONNX Runtime or OpenVINO backend instead.
- **CI coverage of optional backends** *(deferred, YAGNI)*: CI currently builds
  only the OpenCV-based configurations, so the ONNX Runtime / OpenVINO backends
  and the real inference test are exercised locally but not in CI. A CI job that
  installs ONNX Runtime (and OpenVINO on an Intel runner) to run the inference
  test, plus stricter warnings (`-Wall -Wextra`), is intentionally left as a TODO
  and will be added when the need arises.

---

## License

See [LICENSE](LICENSE) for details.

---

## Disclaimer

This README was generated with the help of artificial intelligence. Its contents
have not been fully reviewed by a human. **No warranty is given for the accuracy
or completeness of the information provided.**
