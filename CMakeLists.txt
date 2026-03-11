cmake_minimum_required(VERSION 3.10)
set(CMAKE_OSX_DEPLOYMENT_TARGET "13.7" CACHE STRING "Minimum macOS version")

# Set the project name
project(ObjectDetection)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenCV
find_package(OpenCV REQUIRED)
set(OpenVINO_DIR "/opt/intel/openvino_2025.4.0/runtime/cmake")
set(CMAKE_BUILD_RPATH "/opt/intel/openvino_2025.4.0/runtime/3rdparty/tbb/lib")
find_package(OpenVINO REQUIRED)

set(ORT_ROOT "/Users/georg/Downloads/onnxruntime")

# Add your source files here
set(SOURCE_FILES    main.cpp
                    src/Detection.cpp
                    src/VideoHandler.cpp
                    src/Tracker.cpp
        src/Detection_OpenCV.cpp
        src/Detection_OpenVINO.cpp
        src/Detection_ORT.cpp
        src/Timer.cpp
)

# Add the executable, assuming your source file is named main.cpp
add_executable(ObjectDetection ${SOURCE_FILES}
)

# Includes
target_include_directories(ObjectDetection PRIVATE
                                    include
                                    ${OpenCV_DIRS}
                                    ${ORT_ROOT}/include/onnxruntime/core/session
                                    ${ORT_ROOT}/include/onnxruntime/core/providers/coreml
                                    ${ORT_ROOT}/include/onnxruntime/core/providers/cpu
)

# Link to OpenCV libraries
target_link_libraries(ObjectDetection
                        ${OpenCV_LIBS}
                        ${ORT_ROOT}/build/MacOS/Release/libonnxruntime.dylib
                        openvino::runtime
)

# Optional: Konsolen-Output für Debugging-Zwecke
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
