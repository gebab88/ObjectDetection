# ObjectDetection

Real-time object detection pipeline for video files and webcams, based on YOLO models. Supports five inference backends: **OpenCV DNN**, **ONNX Runtime**, **OpenVINO**, **Hailo**, and **Core AI** – with optional hardware acceleration via CoreML (macOS), XNNPACK (ARM/Linux), Apple Core AI (GPU / Neural Engine on Apple Silicon), and HAILO10H HEF models.

---

## Features

- **Multi-Backend**: Switch between OpenCV DNN, ONNX Runtime, OpenVINO, Hailo and Core AI via `config.yaml` (a detector factory validates the choice against the model format)
- **Multi-Source**: Process video files (`.mp4`, `.mov`, …) or live webcam streams
- **Hardware Acceleration**: CoreML (Apple Silicon / Neural Engine), XNNPACK (ARM Linux), HailoRT and Core AI (GPU / Neural Engine, macOS 27+ / Xcode 27, built automatically when available)
- **YOLO / HEF Support**: YOLOE open-vocabulary exports, YOLO v12 raw output, YOLO v26 end-to-end or raw output and Hailo HEF models
- **Video Output**: Processed frames with bounding boxes are written to the configured output file
- **Audio Passthrough**: The sound of an input video file is copied unchanged into the output (requires the FFmpeg libraries, see [Audio Passthrough](#audio-passthrough))
- **Performance Timing**: Built-in high-resolution timer for total execution time

---

## Supported Platforms

| Platform              | OpenCV DNN | ONNX Runtime | OpenVINO | Hailo | Core AI |
|-----------------------|:----------:|:------------:|:--------:|:-----:|:------:|
| macOS (Apple Silicon) | ✅          | ✅ / CoreML   | ❌ auto-disabled | ❌ | ✅ (auto, macOS 27.0+ / Xcode 27) |
| macOS (Intel)         | ✅          | ✅            | ✅ when installed | ❌ | ❌ |
| Linux (Intel x86_64)  | ✅          | ✅            | ✅ when installed | ❌ | ❌ |
| Linux (aarch64)       | ✅          | ✅ / XNNPACK  | ❌ auto-disabled | ✅ Raspberry Pi only, when installed | ❌ |

---

## Quick Start

The steps from a fresh checkout to a processed video. Each step links to the
section with the details and the optional backends.

1. **Install the build dependencies** (see [Requirements](#requirements)).
   yaml-cpp is needed to read `config.yaml`; FFmpeg copies the audio track into
   the output.
   ```bash
   # macOS (Homebrew)
   brew install cmake opencv yaml-cpp ffmpeg pkg-config
   # macOS (MacPorts, installs under /opt/local, which CMake already searches)
   sudo port install cmake opencv4 yaml-cpp ffmpeg pkgconfig
   # Linux / Raspberry Pi OS
   sudo apt install build-essential cmake pkg-config libopencv-dev libyaml-cpp-dev \
                    libavformat-dev libavcodec-dev libavutil-dev
   ```

2. **Add ONNX Runtime** (recommended: OpenCV DNN alone cannot run YOLO26 models).
   Unpack a [prebuilt release](https://github.com/microsoft/onnxruntime/releases)
   into `onnxruntime/` in the project root, or build it from source (see
   [ONNX Runtime](#onnx-runtime)). Skip this step for Core AI or Hailo only.

3. **Build** (out of source, in `build/`):
   ```bash
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build --parallel
   ```
   The configure output lists which backends were enabled and why others were
   skipped (e.g. `ONNX Runtime backend enabled: ...`,
   `Core AI backend disabled: <reason>`).

4. **Get a model** and put it in the project root. Models are not part of the
   repository. Export one with Ultralytics:
   ```bash
   python3 -m pip install ultralytics
   # ONNX (ONNX Runtime, OpenVINO):
   yolo export model=yolo26m.pt format=onnx imgsz=640
   # Apple Core AI (Apple Silicon, macOS 27+), FP16:
   yolo export model=yolo26m.pt format=coreai imgsz=640 quantize=16
   ```
   YOLO26 and YOLO12 models use the COCO labels in `coco.txt`. For YOLOE see
   [YOLOE Export](#yoloe-export), for Hailo a precompiled `.hef` from the Hailo
   Model Zoo (see [Requirements](#requirements)).

5. **Create the configuration** from the template and set the model, the
   backend and the input (see [Configuration](#configuration)):
   ```bash
   cp config.example.yaml config.yaml
   ```
   ```yaml
   model:
     model_file: "yolo26m.onnx"       # or "yolo26m.aimodel" with CoreAI
     class_names_file: "coco.txt"
   backend:
     framework: "ONNXRuntime"         # OpenCV | OpenVINO | ONNXRuntime | CoreAI | Hailo
   input:
     source: "VIDEO"                  # or "WEBCAM" (with webcam_index)
     video_input_file: "my_video.mp4"
   output:
     video_output_file: "output.mp4"
   ```
   Relative paths are resolved against the directory of `config.yaml`.

6. **Run** (see [Usage](#usage)):
   ```bash
   ./build/ObjectDetection                  # finds config.yaml in ./ or ../
   ./build/ObjectDetection --config path/to/config.yaml
   ```
   The video with the detected boxes is written to `video_output_file`. Ctrl+C
   stops early and still writes the frames processed so far.

7. **Optional: run the tests** (needs GoogleTest, or configure with
   `-DFETCH_GTEST=ON`; see [Build](#build)):
   ```bash
   ctest --test-dir build --output-on-failure
   ```

---

## Requirements

- **CMake** ≥ 3.10
- **C++17** compiler (clang++ or g++)
- **OpenCV** ≥ 4.x (with `opencv_dnn`, `opencv_tracking`)
- **yaml-cpp** *(optional, required to load `config.yaml`; without it built-in defaults are used)*
- **ONNX Runtime** *(optional, required for the ONNX Runtime backend)*
- **OpenVINO** 2025.x *(optional, enabled by CMake only when an Intel CPU is detected)*
- **FFmpeg libraries** *(optional: `libavformat`, `libavcodec`, `libavutil` via pkg-config, required for audio passthrough)*
- **HailoRT** *(optional, Raspberry Pi with 64-bit Linux only, required for `.hef` models)*
- **Core AI** *(optional, Apple Silicon only, requires macOS 27.0+ / Xcode 27 and the Swift interop layer)*

Hailo/HEF support is compiled only when CMake detects a Raspberry Pi and finds
HailoRT. Native builds detect the board
through the Linux device tree. For cross-compilation, set
`-DTARGET_RASPBERRY_PI=ON -DENABLE_HAILO=ON` alongside your Linux/aarch64 toolchain.
Other targets disable Hailo, including its shared HEF/backend enum entries.
Use `-DENABLE_HAILO=OFF` to omit it on a Raspberry Pi as well.

A HEF must be compiled for the connected chip (Hailo Model Zoo:
`ModelZoo/Compiled/<version>/hailo10h/` for a Hailo-10H). Otherwise startup
fails with a message naming the chips the HEF is for and the connected one.

The Hailo backend keeps up to four frames in flight (`run_async`), so the chip
works on the next frames while the CPU reads, decodes and writes the others.
On a Raspberry Pi 5 with Hailo-10H this reaches the chip's own throughput on
`tests/vtest.avi`: about 78 FPS with `yolov8m.hef` and 59 FPS with
`yolo26m_hailo10h.hef`.

Core AI support (`ENABLE_CORE_AI`, default ON) is compiled only on Apple Silicon
when `swiftc`, a macOS SDK that ships `CoreAI.framework` (Xcode 27+) and macOS 27+
are available; otherwise CMake skips it with a status message explaining why.

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
# Optional, for audio passthrough (usually already pulled in by libopencv-dev):
sudo apt install libavformat-dev libavcodec-dev libavutil-dev pkg-config
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

### Audio Passthrough

OpenCV's `cv::VideoWriter` writes images only, so without extra work the
output video has no sound. When CMake finds the FFmpeg libraries
(`libavformat`, `libavcodec`, `libavutil` via pkg-config) it builds
`src/AudioMux.cpp` and defines `HAVE_AUDIO_PASSTHROUGH`; the configure step
prints `Audio passthrough enabled` or the reason it is disabled.

- After the last frame (also after Ctrl+C) the program copies the audio track of
  the input **video file** into the output, without re-encoding, and cuts it at
  the end of the processed video. Webcam runs stay silent.
- The audio keeps its timing relative to the first video frame. For iPhone
  clips (AAC) an `.mp4`/`.mov` output keeps the audio exactly as in the source,
  including the AAC encoder priming. AVI cannot store that, so an `.avi` output
  starts with the audio packet that covers t = 0 (offset below 1 ms for iPhone clips).
- When the source has several audio tracks (e.g. spatial audio), the default
  track is used if the output container can hold it, otherwise the first one that fits.
- If the source has no audio, or copying fails, the video is kept without sound
  and a message is printed; the run itself does not fail.
- The output is written to `<name>.audio-tmp.<ext>` next to the output first and
  then replaces it.

Disable it with `-DENABLE_AUDIO_PASSTHROUGH=OFF`. On macOS the FFmpeg libraries
come with Homebrew's or MacPorts' OpenCV (`ffmpeg` package); make sure
`pkg-config` is installed.

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

On a Raspberry Pi with HailoRT, CMake also builds `hailo_tests_runner` when
`.hef` files lie in the project root. For every HEF it runs the first 40 frames
of `tests/vtest.avi` on the accelerator and checks that frames processed with
several in flight are pixel-identical to frames processed one at a time, and
that people are detected. Without a connected Hailo device or the video, the
tests are skipped. Optional inputs:

```bash
cmake .. \
  -DHAILO_TEST_VIDEO=/path/to/video.avi \
  -DHAILO_TEST_INCOMPATIBLE_HEF=/path/to/hailo8_model.hef  # checks the error for a HEF of another chip
```

### Core AI Backend (Apple platforms)

The Core AI backend (Apple's on-device inference framework for Apple Silicon,
iOS 27.0+ / macOS 27.0+, Xcode 27) is compiled into the binary automatically when
the toolchain supports it. A regular build is enough:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

- `ENABLE_CORE_AI` defaults to ON. CMake then compiles the Swift interop layer
  (`swift/CoreAIInterop.swift`) and the backend (`src/Detection_CoreAI.cpp`) and
  defines `HAVE_CORE_AI`, but only when all of these hold:
  - Apple Silicon (other platforms and CPU architectures disable it automatically),
  - `swiftc` is found,
  - the macOS SDK provides `CoreAI.framework` (Xcode 27+; older SDKs would fail
    with `no such module 'CoreAI'`),
  - the host runs macOS 27+, or `CMAKE_OSX_DEPLOYMENT_TARGET` is set to 27.0+
    explicitly (to build macOS 27 binaries on an older host).
- When a requirement is missing, the configure step prints
  `Core AI backend disabled: <reason>` and the build continues without it. The
  setting is not forced off, so the backend is picked up on the next configure
  once the toolchain or OS is upgraded.
- A build that includes the backend raises the macOS deployment target to 27.0.
- Pass `-DENABLE_CORE_AI=OFF` to leave the backend out explicitly.
- To select the backend at runtime, set the framework in the `backend:` section of
  `config.yaml`:

```yaml
backend:
  framework: "CoreAI"
```

- **Fallback:** when the backend was not compiled into the binary or fails during
  initialisation (missing `.aimodel` asset, Swift runtime missing, OS version below
  macOS 27.0, …), `createDetector` falls back to the ONNX Runtime backend — which
  itself has a CoreML / CPU fallback. A hard failure would regress the existing
  ONNX Runtime path.

  Because the ONNX Runtime backend consumes ONNX-format assets, a Core AI selection
  with an `.aimodel` model file is mapped to its ONNX counterpart
  (`yolo26x.aimodel` → `yolo26x.onnx`) before the fallback hands the model to the ONNX
  Runtime backend, so the fallback actually gets a loadable model.

- **Model asset:** the Core AI backend runs an Apple Core AI `.aimodel` asset (the
  successor of the CoreML `.mlmodel` format; an ONNX file such as `yolo26x.onnx` cannot
  be loaded by it). Point `model_file` in `config.yaml` at the asset:

```yaml
model:
  model_file: "yolo26x.aimodel"   # Apple Core AI asset; use "yolo26x.onnx" for ONNX Runtime
  class_names_file: "coco.txt"
  model_width: 640
  model_height: 640
```

  The asset is produced with the Ultralytics Core AI export:

```bash
yolo export model=yolo26x.pt format=coreai imgsz=640 quantize=16
```

  This writes `yolo26x.aimodel` (static graph exported at `imgsz`, raw YOLO26 head
  output `(1, 84, 8400)` with host-side NMS). `quantize=16` produces an **FP16**
  asset (float16 weights, input and output; the backend converts the frame and the
  output). It is strongly recommended: on an M5 Pro `yolo26x` runs at about 6.5 ms
  per frame instead of 27 ms, and over a 1025-frame test video 99.7 % of the FP32
  detections were found again (mean box IoU 0.997); the rest were objects with a
  score right at the threshold. Leave out `quantize=16` for an FP32 asset. Keep the asset next to the ONNX files in
  the repository root and make sure the filename keeps the `yolo26` substring so the
  model format is detected correctly.

  For smaller assets, `export_coreai.py` exports with INT8 or INT4 weights
  (weight-only quantization through coreai-torch; activations, input and output
  stay FP16, so the backend runs them unchanged). The Ultralytics export itself
  offers only FP32 and FP16:

```bash
python3 export_coreai.py --weights yolo26x.pt --quantize 8   # writes yolo26x_q8.aimodel
python3 export_coreai.py --weights yolo26x.pt --quantize 4   # writes yolo26x_q4.aimodel
```

  Measured with `yolo26x` on an M5 Pro (accuracy from `yolo val` on coco128, timing
  from the app on the 1025-frame test video):

  | Weights | Size | Time per frame | mAP50 | mAP50-95 |
  |---|---|---|---|---|
  | FP16 | 107 MB | 6.4 ms | 0.817 | 0.661 |
  | INT8 | 54 MB | 6.7 ms | 0.818 | 0.660 |
  | INT4 | 29 MB | 12.5 ms | 0.771 | 0.591 |

  INT8 keeps the accuracy at half the size but is not faster; INT4 loses accuracy
  and runs slower on the GPU. FP16 stays the default for speed.

  > **Note:** the ONNX Runtime backend consumes ONNX-format assets and cannot run
  > `.aimodel` packages. The automatic fallback therefore maps an `.aimodel`
  > selection to its ONNX counterpart (e.g. `yolo26x.aimodel` → `yolo26x.onnx`) and
  > lets the ONNX Runtime backend load that instead.

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

There is no Core AI job yet: GitHub-hosted runners do not offer a macOS 27 / Xcode 27
image. The macOS jobs build with `ENABLE_CORE_AI` at its default, so they verify that
CMake skips the backend cleanly when the SDK lacks `CoreAI.framework`.

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
compiled CoreML model in a `.ort_coreml_cache/` directory next to the model
file, so the (slow) first-run compilation is reused on later runs regardless of
the working directory. If that directory cannot be created (e.g. a read-only
model location), the program warns and runs without a persistent cache.

CoreML only takes operators whose tensor shapes are fully known. Many ONNX
exports declare a dynamic batch size (and sometimes dynamic height/width), which
would leave the whole model on the CPU. The program therefore pins such dynamic
input dimensions to batch 1 and the configured `model_width`/`model_height`
before the session is created (`[Provider] Pinned dynamic input dimension ...`).
For `yolo26x.onnx` on an M5 Pro this is the difference between about 310 ms
(CPU) and 33 ms (GPU via CoreML) per frame.

With CoreML the backend also keeps two frames in flight, each on its own worker
thread, so the next inference starts while the previous frame is decoded and
drawn (`yolo26x.onnx`: 32.9 → 31.7 ms per frame; a third frame gains nothing).
CPU-only sessions already use all cores per frame and stay synchronous.

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

#### Core AI and the Apple Neural Engine

On Apple Silicon the **Apple Neural Engine** is used in two ways, depending on
which backend is compiled in and selected:

- **ONNX Runtime + CoreML** (default-capable, see above): the CoreML Execution
  Provider targets the Neural Engine and caches compiled models in
  `.ort_coreml_cache/` next to the model file.
- **Core AI** (built automatically when available, see [Core AI Backend](#core-ai-backend-apple-platforms)):
  Apple's newer **Core AI** framework (WWDC 2026, iOS 27.0+ / macOS 27.0+, Apple
  Silicon, Xcode 27) runs inference through the Swift interop layer (`AIModel` /
  `InferenceFunction` / `InferenceValue`, ahead-of-time compiled `.aimodel`
  assets, unified CPU/GPU/Neural Engine runtime). The existing ONNX Runtime /
  CoreML path remains the default and stays available for older targets and
  platforms.

The Core AI backend keeps several frames in flight, each on its own worker
thread, and runs them on fixed compute units ("lanes"). By default there are six
lanes: GPU, GPU, Neural Engine, twice. The GPU gets two frames for each one on
the Neural Engine because it is about twice as fast, and every lane appears twice
so that one frame's preprocessing and decoding overlap the other's inference.
Measured with `yolo26x` on an M5 Pro (ms per frame, whole test video):

| Lanes (`COREAI_COMPUTE_UNIT`) | FP16 asset | FP32 asset |
|---|---|---|
| GPU + Neural Engine (default, `gpu+neural_engine`) | **6.5** | 27.1 |
| GPU only (`gpu`, four lanes) | 9.0–9.5 | 27.3 |
| one frame at a time on Core AI's choice (before) | – | 33.3 |

FP32 models barely use the Neural Engine (its specialization runs mostly on the
GPU), so the Neural Engine lanes mainly pay off with FP16 assets. The lanes can
be changed with the `COREAI_COMPUTE_UNIT` environment variable; operators a unit
cannot run still fall back to the others:

```bash
COREAI_COMPUTE_UNIT=gpu           ./build/ObjectDetection  # four GPU lanes
COREAI_COMPUTE_UNIT=neural_engine ./build/ObjectDetection  # four Neural Engine lanes
COREAI_COMPUTE_UNIT=all           ./build/ObjectDetection  # four lanes, Core AI picks
COREAI_COMPUTE_UNIT=cpu           ./build/ObjectDetection  # two CPU-only lanes
```

> **Known Core AI issue:** with GPU and Neural Engine lanes running at the same
> time, Core AI (macOS 27.2) occasionally returns an all-zero output for a GPU
> inference of an FP16 model (about 1 in 1000). The backend detects this and
> reruns the frame (`[CoreAI] Rerunning an all-zero output ...`, printed once).
> It does not happen with a single unit (`COREAI_COMPUTE_UNIT=gpu`).

When the Core AI backend is not compiled in (see the requirements above) or fails
during initialisation, the application falls back to the ONNX Runtime backend.

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
  framework: "ONNXRuntime"   # OpenCV | OpenVINO | ONNXRuntime | CoreAI | Hailo (case-sensitive)
  use_cuda: false

input:
  source: "VIDEO"            # WEBCAM | VIDEO
  webcam_index: 0
  video_input_file: "tests/test_mov1.MOV"

output:
  video_output_file: "output.mp4"

display:
  show_frames: false
```

If yaml-cpp or `config.yaml` is not available, the program uses built-in
defaults that match the backends compiled into the binary. Missing or malformed
keys fall back to these defaults, but an unknown `backend.framework` or
`input.source` name (e.g. a typo such as `CoreAi` or `Video`) aborts at startup
with the list of valid names instead of switching to another backend or source. Likewise, if inference fails while processing
frames (for example because `model_width`/`model_height` do not match the
model), the program prints the reason and aborts instead of writing a video
without detections.

To run detection through the Apple **Core AI** backend instead of ONNX Runtime,
set:

```yaml
backend:
  framework: "CoreAI"
```

The framework name is validated against the compiled backend set at startup and
the factory reports a clear error when the chosen backend cannot handle the model
format. On Apple Silicon the Core AI backend is compiled in automatically when the
toolchain supports it (see [Core AI Backend](#core-ai-backend-apple-platforms));
when it is not compiled in or fails during initialisation, the factory falls back
to the ONNX Runtime backend automatically.

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

Place your model file (`.onnx`) and the class names file in the project directory, then run
(the class names file lists one name per line in class-id order; blank lines and
lines starting with `#` are ignored):

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

The processed video is saved to `output.video_output_file` from `config.yaml`
(the legacy `input.video_output_file` location is still honoured).
If the requested container/codec cannot be opened, the writer tries compatible
fallback codecs and may use a new `.avi` fallback filename.

### Supported Models

The pipeline supports three YOLO output formats plus Hailo HEFs and Apple Core AI
assets, detected from the model **filename** (`detectModelFormat`): **YOLO12**
(raw grid output), **YOLO26** (end-to-end with built-in NMS, or raw with host-side
NMS depending on the export), and **YOLOE**
(open-vocabulary; raw or end-to-end depending on the export). Files ending in
`.hef` use the **HEF** format and require the Hailo backend. Apple Core AI assets
ending in `.aimodel` (YOLO26 end-to-end exports, e.g. via Ultralytics'
`format="coreai"`, or raw-head exports) use the **YOLO26** format and are served by
the Core AI backend on Apple Silicon, which dispatches raw and end-to-end outputs
via the shared decoders.

| Model file        | Backend compatibility              | Notes                          |
|-------------------|-------------------------------     |--------------------------------|
| `yoloe*.onnx`     | OpenCV DNN, ONNX Runtime, OpenVINO | Open-vocabulary classes baked in during export; raw YOLO output or end-to-end output |
| `yolo12m.onnx`    | OpenCV DNN, ONNX Runtime, OpenVINO | Standard YOLO output (NMS required) |
| `yolo26m.onnx`    | ONNX Runtime, OpenVINO, no OpenCV DNN | End-to-end, NMS built-in     |
| `yolo26x.onnx`    | ONNX Runtime, OpenVINO, no OpenCV DNN | End-to-end, larger variant, NMS built-in     |
| `*.hef`           | Hailo                               | HailoRT model, either with on-chip NMS (e.g. yolov8m) or a raw YOLO26 head decoded on the CPU (e.g. yolo26m) |
| `yolo26m.aimodel` | Core AI                            | Apple Silicon, macOS 27.0+ / Xcode 27; built automatically when the toolchain supports it, YOLO26 export via Core AI (end-to-end or raw-head, dispatched by output shape) |

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
| `nanodet_repvgg_a12_hailo10h.hef` | HEF | HailoRT backend |
| `yolo26m_hailo10h.hef`   | HEF    | Raw YOLO26 head, decoded on the CPU     |

The `-seg` models are segmentation exports, but only their bounding boxes are
used (see Known Limitations). `-pf` denotes the prompt-free YOLOE variant.

---

## Project Structure

```
ObjectDetection/
├── CMakeLists.txt
├── config.example.yaml       # Sample runtime config (copy to config.yaml)
├── export.py                 # YOLOE ONNX export helper
├── export_coreai.py          # Core AI export with FP16, INT8 or INT4 weights
├── coco.txt                  # COCO class names (80 classes)
├── coco.yaml                 # COCO dataset config
├── yoloe_classes.txt         # YOLOE export/runtime classes
├── yoloe_pf_classes.txt      # YOLOE prompt-free runtime classes
├── include/
│   ├── Detection.hpp         # Abstract base class
│   ├── Detection_OpenCV.hpp
│   ├── Detection_ORT.hpp
│   ├── Detection_OpenVINO.hpp
│   ├── Detection_Hailo.hpp
│   ├── Yolo26Decoder.hpp     # Decoder for raw YOLO26 heads of Hailo HEFs
│   ├── HailoOutput.hpp       # Hailo NMS parsing, scaling and drawing
│   ├── HailoMessages.hpp     # User-facing Hailo error messages
│   ├── DetectionPipeline.hpp # Frame loop keeping several frames in flight
│   ├── AsyncFrameWriter.hpp  # Encodes the output video on its own thread
│   ├── FrameSlots.hpp        # Worker thread + buffers per frame in flight (Core AI, ORT/CoreML)
│   ├── DetectorFactory.hpp   # Backend selection + createDetector()
│   ├── VideoHandler.hpp
│   ├── AudioMux.hpp          # Audio passthrough into the output video (FFmpeg)
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
│   ├── Detection_Hailo.cpp
│   ├── Yolo26Decoder.cpp
│   ├── HailoOutput.cpp
│   ├── HailoMessages.cpp
│   ├── DetectionPipeline.cpp
│   ├── AsyncFrameWriter.cpp
│   ├── DetectorFactory.cpp
│   ├── VideoHandler.cpp
│   ├── AudioMux.cpp
│   ├── Tracker.cpp
│   ├── Timer.cpp
│   └── openvino/
│       └── Detection_OpenVINO.cpp
└── tests/
    ├── test_framework.cpp    # Unit tests (format, decoders, config, ...)
    ├── test_inference.cpp    # Real ONNX inference test over tests/ videos
    ├── test_audio_mux.cpp    # Audio passthrough tests (built with FFmpeg libraries)
    ├── test_yolo26_decoder.cpp     # Raw YOLO26 head decoding (Hailo)
    ├── test_hailo_output.cpp       # Hailo NMS parsing, scaling and drawing
    ├── test_hailo_messages.cpp     # Hailo error messages
    ├── test_hailo_inference.cpp    # End-to-end tests on the Hailo accelerator
    ├── test_detection_pipeline.cpp # Frame order and frames in flight
    ├── test_async_frame_writer.cpp # Output writer thread
    ├── test_frame_slots.cpp        # Worker slots for frames in flight
    └── *.MOV                 # Local, git-ignored test videos
```

---

## Class Overview

| Class               | Responsibility                                               |
|---------------------|--------------------------------------------------------------|
| `Detection`         | Abstract base class; loads class names, holds shared state; `detect_async()` / `max_frames_in_flight()` for pipelined backends (default: synchronous `detect()`) |
| `Detection_OpenCV`  | OpenCV DNN backend; CUDA support optional                    |
| `Detection_ORT`     | ONNX Runtime backend; CoreML (macOS) and XNNPACK (ARM) EP   |
| `Detection_OpenVINO`| OpenVINO backend *(Intel CPUs only; auto-disabled otherwise)* |
| `Detection_Hailo`   | HailoRT backend for `.hef` models; async inference with up to four frames in flight |
| `Detection_CoreAI`  | Core AI backend for YOLO26 on Apple Silicon (macOS 27+, Swift interop layer; falls back to ONNX Runtime) |
| `detection_pipeline`| `run()` reads, detects and delivers frames in order, with up to `max_frames_in_flight()` frames in progress |
| `AsyncFrameWriter`  | Writes the output frames on its own thread (bounded queue), so encoding overlaps with reading and detection |
| `yolo26`            | `decode()` turns the quantized raw YOLO26 head outputs of a HEF into boxes (no ONNX postprocess model needed) |
| `hailo_output`      | Parses Hailo NMS output, scales YOLO26 results to the frame and draws both with the shared `drawDetection()` |
| `DetectorFactory`   | `createDetector()` builds the configured backend; `supportedBackendsFor()` encodes format↔backend compatibility (free functions) |
| `YoloPostprocess`   | Header-only decoders (raw / end-to-end + `decodeByFormat` dispatch), NMS and box/label drawing |
| `VideoHandler`      | Opens video/webcam, crops to square, writes output video; `finish_output()` closes it and adds the source audio |
| `audio_mux`         | `copyAudioTrack()` remuxes the source's audio track into the finished output video (stream copy, FFmpeg) |
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
- **Raw Hailo YOLO26**: HEFs without on-chip NMS must contain a YOLO26
  detection head: per scale one 4-channel box output and one class output with
  the same grid, UINT16 with a single quantization scale each (as in the Model
  Zoo's `yolo26m`). The outputs are matched by shape, not by name, so other
  YOLO26 sizes with this layout should work too; only `yolo26m` is tested.
  Other raw heads (e.g. YOLOv8 with DFL box outputs) are rejected at startup.
- **CI coverage of optional backends** *(deferred, YAGNI)*: CI builds only the
  OpenCV-based configurations. The ONNX Runtime backend expects a local install
  under `onnxruntime/`, which does not exist on the CI runners, so
  `src/Detection_ORT.cpp` — the main inference path — is never even **compiled**
  in CI: a broken include or an API change after an ONNX Runtime update keeps CI
  green. The real inference test additionally needs model and video assets,
  which are git-ignored, so a CI job could only restore compile coverage, not
  test coverage. Such a job would download an ONNX Runtime release into
  `onnxruntime/` (the CMake search is platform-independent, so a plain x86-64
  Linux runner works); it is intentionally left unbuilt until the need arises.
  The same applies to OpenVINO on an Intel runner and to stricter warnings
  (`-Wall -Wextra`).

---

## License

See [LICENSE](LICENSE) for details.

---

## Disclaimer

This README was generated with the help of artificial intelligence. Its contents
have not been fully reviewed by a human. **No warranty is given for the accuracy
or completeness of the information provided.**
