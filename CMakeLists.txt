cmake_minimum_required(VERSION 3.10)
set(CMAKE_OSX_DEPLOYMENT_TARGET "13.7" CACHE STRING "Minimum macOS version")

# Set the project name
project(ObjectDetection)

message(STATUS "System: ${CMAKE_SYSTEM_NAME}")

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

# ONNX Runtime path (local installation)
set(ORT_DIR "${CMAKE_SOURCE_DIR}/onnxruntime" CACHE PATH "ONNX Runtime directory")

# Gemeinsame Quelldateien
set(COMMON_SOURCES
    src/main.cpp
    src/VideoHandler.cpp
    src/Tracker.cpp
    src/Timer.cpp
    src/Detection.cpp
    src/Detection_OpenCV.cpp
    src/Detection_ORT.cpp
)

# Plattformspezifische Quelldateien
if(APPLE)
    message(STATUS "Building for macOS")
    set(OpenVINO_DIR "/opt/intel/openvino_2025.4.0/runtime/cmake")
    set(CMAKE_BUILD_RPATH "/opt/intel/openvino_2025.4.0/runtime/3rdparty/tbb/lib")
    find_package(OpenVINO REQUIRED)

    set(PLATFORM_SOURCES
        src/macos/Detection_OpenVINO.cpp
    )
    set(PLATFORM_LIBS
        ${OpenCV_LIBS}
        openvino::runtime
        ${ORT_DIR}/build/MacOS/Release/libonnxruntime.dylib
    )
    set(PLATFORM_INCLUDE_DIRS
        ${ORT_DIR}/include/onnxruntime/core/session
        ${ORT_DIR}/include/onnxruntime/core/providers/coreml
        ${ORT_DIR}/include/onnxruntime/core/providers/cpu
    )
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64" AND NOT APPLE)
    message(STATUS "ARM-Architektur erkannt: ${CMAKE_SYSTEM_PROCESSOR}")
    message(STATUS "Building for ARM on Linux")

    set(PLATFORM_SOURCES)
    
    set(PLATFORM_LIBS
        ${OpenCV_LIBS}
        ${ORT_DIR}/lib/libonnxruntime.so
        Threads::Threads
        ${CMAKE_DL_LIBS}
    )
    set(PLATFORM_INCLUDE_DIRS
        ${ORT_DIR}/include/onnxruntime/core/session
        ${ORT_DIR}/include/onnxruntime/core/providers/cpu
    )
endif()


# Add the executable, assuming your source file is named main.cpp
add_executable(${PROJECT_NAME}
    ${COMMON_SOURCES}
    ${PLATFORM_SOURCES}
)

# Includes
target_include_directories(${PROJECT_NAME} PRIVATE
                                    include
                                    ${PLATFORM_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${PLATFORM_LIBS}
)

# Optional: Konsolen-Output für Debugging-Zwecke
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
