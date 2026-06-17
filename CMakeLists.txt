cmake_minimum_required(VERSION 3.10)

if(APPLE AND (NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET OR CMAKE_OSX_DEPLOYMENT_TARGET STREQUAL "" OR CMAKE_OSX_DEPLOYMENT_TARGET STREQUAL "13.7"))
    execute_process(
        COMMAND sw_vers -productVersion
        OUTPUT_VARIABLE MACOS_PRODUCT_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(MACOS_PRODUCT_VERSION MATCHES "^([0-9]+)\\.([0-9]+)")
        set(MACOS_MAJOR_VERSION "${CMAKE_MATCH_1}")
        set(MACOS_MINOR_VERSION "${CMAKE_MATCH_2}")
        if(MACOS_MAJOR_VERSION GREATER_EQUAL 11)
            set(PROJECT_DEFAULT_MACOS_DEPLOYMENT_TARGET "${MACOS_MAJOR_VERSION}.0")
        else()
            set(PROJECT_DEFAULT_MACOS_DEPLOYMENT_TARGET "${MACOS_MAJOR_VERSION}.${MACOS_MINOR_VERSION}")
        endif()
        set(CMAKE_OSX_DEPLOYMENT_TARGET "${PROJECT_DEFAULT_MACOS_DEPLOYMENT_TARGET}" CACHE STRING "Minimum macOS version" FORCE)
    endif()
endif()

# Set the project name
project(ObjectDetection)

message(STATUS "System: ${CMAKE_SYSTEM_NAME}")
message(STATUS "Processor: ${CMAKE_SYSTEM_PROCESSOR}")

set(DETECTED_CPU_VENDOR "")
if(APPLE)
    execute_process(
        COMMAND sysctl -n machdep.cpu.vendor
        OUTPUT_VARIABLE DETECTED_CPU_VENDOR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
elseif(UNIX)
    execute_process(
        COMMAND sh -c "awk -F': ' '/vendor_id/{print $2; exit}' /proc/cpuinfo"
        OUTPUT_VARIABLE DETECTED_CPU_VENDOR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
endif()
if(NOT DETECTED_CPU_VENDOR AND APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(DETECTED_CPU_VENDOR "Apple")
endif()

set(HAS_INTEL_CPU OFF)
if(DETECTED_CPU_VENDOR STREQUAL "GenuineIntel")
    set(HAS_INTEL_CPU ON)
endif()

if(DETECTED_CPU_VENDOR)
    message(STATUS "CPU vendor: ${DETECTED_CPU_VENDOR}")
else()
    message(STATUS "CPU vendor: unknown")
endif()

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

option(ENABLE_ONNX_RUNTIME "Build ONNX Runtime backend when local runtime is available" ON)
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(DEFAULT_ENABLE_ORT_COREML ON)
else()
    set(DEFAULT_ENABLE_ORT_COREML OFF)
endif()
option(ENABLE_ORT_COREML "Enable ONNX Runtime CoreML execution provider on macOS" ${DEFAULT_ENABLE_ORT_COREML})
if(HAS_INTEL_CPU)
    set(DEFAULT_ENABLE_OPENVINO ON)
else()
    set(DEFAULT_ENABLE_OPENVINO OFF)
endif()
option(ENABLE_OPENVINO "Build OpenVINO backend when an Intel CPU is detected and OpenVINO is available" ${DEFAULT_ENABLE_OPENVINO})
if(ENABLE_OPENVINO AND NOT HAS_INTEL_CPU)
    message(STATUS "OpenVINO disabled: no Intel CPU detected.")
    set(ENABLE_OPENVINO OFF CACHE BOOL "Build OpenVINO backend when an Intel CPU is detected and OpenVINO is available" FORCE)
elseif(NOT ENABLE_OPENVINO AND NOT HAS_INTEL_CPU)
    message(STATUS "OpenVINO disabled by default: no Intel CPU detected.")
endif()
option(ENABLE_YAML_CONFIG "Load runtime config from config.yaml when yaml-cpp is available" ON)
option(FETCH_GTEST "Download GoogleTest into the build directory when it is not installed" OFF)
option(ENABLE_INFERENCE_TESTS "Build real ONNX inference tests when local model assets are available" ON)

list(APPEND CMAKE_PREFIX_PATH "/opt/local" "/opt/local/libexec/opencv4" "/opt/homebrew" "/usr/local")

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

# ONNX Runtime path (local installation)
set(ORT_DIR "${CMAKE_SOURCE_DIR}/onnxruntime" CACHE PATH "ONNX Runtime directory")
set(ONNX_RUNTIME_BACKEND_ENABLED OFF)

set(PLATFORM_SOURCES)
set(PLATFORM_LIBS ${OpenCV_LIBS})
set(PLATFORM_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})
set(PROJECT_COMPILE_DEFINITIONS)

# Common source files
set(COMMON_SOURCES
    src/main.cpp
    src/VideoHandler.cpp
    src/Tracker.cpp
    src/Timer.cpp
    src/Detection.cpp
    src/Detection_OpenCV.cpp
)

if(ENABLE_YAML_CONFIG)
    find_package(yaml-cpp QUIET)
    if(yaml-cpp_FOUND)
        list(APPEND PROJECT_COMPILE_DEFINITIONS USE_YAML_CONFIG)
        if(TARGET yaml-cpp::yaml-cpp)
            list(APPEND PLATFORM_LIBS yaml-cpp::yaml-cpp)
        else()
            list(APPEND PLATFORM_LIBS ${YAML_CPP_LIBRARIES})
            list(APPEND PLATFORM_INCLUDE_DIRS ${YAML_CPP_INCLUDE_DIR})
        endif()
    else()
        message(WARNING "yaml-cpp not found: building with built-in default config.")
    endif()
endif()

if(ENABLE_ONNX_RUNTIME)
    if(APPLE)
        find_library(ORT_LIBRARY
            NAMES onnxruntime
            PATHS
                "${ORT_DIR}/build/MacOS/Release"
                "${ORT_DIR}/lib"
            NO_DEFAULT_PATH
        )
        set(ORT_PROVIDER_INCLUDE_DIRS
            "${ORT_DIR}/include"
            "${ORT_DIR}/include/onnxruntime/core/providers/coreml"
            "${ORT_DIR}/include/onnxruntime/core/providers/cpu"
        )
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64" AND NOT APPLE)
        find_library(ORT_LIBRARY
            NAMES onnxruntime
            PATHS "${ORT_DIR}/lib"
            NO_DEFAULT_PATH
        )
        set(ORT_PROVIDER_INCLUDE_DIRS
            "${ORT_DIR}/include"
            "${ORT_DIR}/include/onnxruntime/core/providers/cpu"
        )
    endif()

    find_path(ORT_SESSION_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        PATHS
            "${ORT_DIR}/include"
            "${ORT_DIR}/include/onnxruntime/core/session"
        NO_DEFAULT_PATH
    )
    if(ORT_LIBRARY
       AND ORT_SESSION_INCLUDE_DIR)
        list(APPEND COMMON_SOURCES src/Detection_ORT.cpp)
        list(APPEND PLATFORM_LIBS "${ORT_LIBRARY}" Threads::Threads ${CMAKE_DL_LIBS})
        list(APPEND PLATFORM_INCLUDE_DIRS "${ORT_SESSION_INCLUDE_DIR}" ${ORT_PROVIDER_INCLUDE_DIRS})
        list(APPEND PROJECT_COMPILE_DEFINITIONS HAVE_ONNX_RUNTIME)
        if(APPLE AND ENABLE_ORT_COREML)
            list(APPEND PROJECT_COMPILE_DEFINITIONS ENABLE_ORT_COREML)
        endif()
        set(ONNX_RUNTIME_BACKEND_ENABLED ON)
        message(STATUS "ONNX Runtime backend enabled: ${ORT_LIBRARY}")
    else()
        message(WARNING "ONNX Runtime not found at ${ORT_DIR}: backend disabled.")
    endif()
endif()

if(ENABLE_OPENVINO)
    # Discover any system OpenVINO install instead of pinning a single version.
    # A user-supplied OpenVINO_DIR or CMAKE_PREFIX_PATH still takes precedence.
    if(NOT DEFINED OpenVINO_DIR)
        file(GLOB OPENVINO_CMAKE_DIRS "/opt/intel/openvino_*/runtime/cmake")
        if(OPENVINO_CMAKE_DIRS)
            list(SORT OPENVINO_CMAKE_DIRS)
            list(REVERSE OPENVINO_CMAKE_DIRS)            # prefer the newest version
            list(GET OPENVINO_CMAKE_DIRS 0 OPENVINO_NEWEST_CMAKE_DIR)
            list(APPEND CMAKE_PREFIX_PATH "${OPENVINO_NEWEST_CMAKE_DIR}")
        endif()
    endif()

    find_package(OpenVINO QUIET)

    if(OpenVINO_FOUND)
        list(APPEND PLATFORM_SOURCES src/macos/Detection_OpenVINO.cpp)
        list(APPEND PLATFORM_LIBS openvino::runtime)
        list(APPEND PROJECT_COMPILE_DEFINITIONS HAVE_OPENVINO)
        # Point the run-time path at the TBB that ships next to the discovered
        # runtime, regardless of the installed OpenVINO version.
        get_filename_component(OPENVINO_RUNTIME_DIR "${OpenVINO_DIR}/.." ABSOLUTE)
        if(EXISTS "${OPENVINO_RUNTIME_DIR}/3rdparty/tbb/lib")
            set(CMAKE_BUILD_RPATH "${OPENVINO_RUNTIME_DIR}/3rdparty/tbb/lib")
        endif()
        message(STATUS "OpenVINO backend enabled (OpenVINO_DIR=${OpenVINO_DIR})")
    else()
        message(WARNING "OpenVINO requested on Intel CPU, but package was not found: backend disabled.")
    endif()
endif()

# Platform-specific source files
if(APPLE)
    message(STATUS "Building for macOS")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
    message(STATUS "ARM architecture detected: ${CMAKE_SYSTEM_PROCESSOR}")
    message(STATUS "Building for ARM on Linux")
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

target_compile_definitions(${PROJECT_NAME} PRIVATE
    ${PROJECT_COMPILE_DEFINITIONS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${PLATFORM_LIBS}
)

# Optional: Konsolen-Output für Debugging-Zwecke
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "yaml-cpp: ${YAML_CPP_LIBRARIES}")
message(STATUS "yaml-cpp libraries: ${YAML_CPP_LIBRARIES}")
message(STATUS "yaml-cpp includes:  ${YAML_CPP_INCLUDE_DIR}")

# Enable testing and add GoogleTest-based unit tests
include(CTest)

set(DEFAULT_INFERENCE_TEST_MODEL "")
set(DEFAULT_INFERENCE_TEST_CLASSES "")
if(EXISTS "${CMAKE_SOURCE_DIR}/yoloe-26m-seg-pf.onnx")
    set(DEFAULT_INFERENCE_TEST_MODEL "${CMAKE_SOURCE_DIR}/yoloe-26m-seg-pf.onnx")
    set(DEFAULT_INFERENCE_TEST_CLASSES "${CMAKE_SOURCE_DIR}/yoloe_pf_classes.txt")
elseif(EXISTS "${CMAKE_SOURCE_DIR}/yoloe-11l-seg.onnx")
    set(DEFAULT_INFERENCE_TEST_MODEL "${CMAKE_SOURCE_DIR}/yoloe-11l-seg.onnx")
    set(DEFAULT_INFERENCE_TEST_CLASSES "${CMAKE_SOURCE_DIR}/yoloe_classes.txt")
elseif(EXISTS "${CMAKE_SOURCE_DIR}/yolo26m.onnx")
    set(DEFAULT_INFERENCE_TEST_MODEL "${CMAKE_SOURCE_DIR}/yolo26m.onnx")
    set(DEFAULT_INFERENCE_TEST_CLASSES "${CMAKE_SOURCE_DIR}/coco.txt")
elseif(EXISTS "${CMAKE_SOURCE_DIR}/yolo26x.onnx")
    set(DEFAULT_INFERENCE_TEST_MODEL "${CMAKE_SOURCE_DIR}/yolo26x.onnx")
    set(DEFAULT_INFERENCE_TEST_CLASSES "${CMAKE_SOURCE_DIR}/coco.txt")
endif()
set(INFERENCE_TEST_MODEL "${DEFAULT_INFERENCE_TEST_MODEL}" CACHE FILEPATH "ONNX model used by the real inference test")
set(INFERENCE_TEST_CLASSES "${DEFAULT_INFERENCE_TEST_CLASSES}" CACHE FILEPATH "Class names file used by the real inference test")
set(INFERENCE_TEST_VIDEO "${CMAKE_SOURCE_DIR}/test.mov" CACHE FILEPATH "Video file used by the real inference test")

if(BUILD_TESTING)
    find_package(GTest QUIET)
    if(NOT GTest_FOUND AND FETCH_GTEST)
        include(FetchContent)
        FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )
        FetchContent_MakeAvailable(googletest)
    endif()

    if(TARGET GTest::gtest_main)
        add_executable(tests_runner tests/test_framework.cpp)
        target_include_directories(tests_runner PRIVATE include)
        target_link_libraries(tests_runner PRIVATE GTest::gtest_main ${OpenCV_LIBS})
        add_test(NAME framework_tests COMMAND tests_runner)

        if(ENABLE_INFERENCE_TESTS)
            if(NOT ONNX_RUNTIME_BACKEND_ENABLED)
                message(STATUS "Real inference tests disabled: ONNX Runtime backend is unavailable.")
            elseif(NOT EXISTS "${INFERENCE_TEST_MODEL}")
                message(STATUS "Real inference tests disabled: model not found (${INFERENCE_TEST_MODEL}).")
            elseif(NOT EXISTS "${INFERENCE_TEST_CLASSES}")
                message(STATUS "Real inference tests disabled: class names file not found (${INFERENCE_TEST_CLASSES}).")
            elseif(NOT EXISTS "${INFERENCE_TEST_VIDEO}")
                message(STATUS "Real inference tests disabled: video file not found (${INFERENCE_TEST_VIDEO}).")
            else()
                add_executable(inference_tests_runner
                    tests/test_inference.cpp
                    src/Detection.cpp
                    src/Detection_ORT.cpp
                )
                target_include_directories(inference_tests_runner PRIVATE
                    include
                    ${PLATFORM_INCLUDE_DIRS}
                )
                target_compile_definitions(inference_tests_runner PRIVATE
                    ${PROJECT_COMPILE_DEFINITIONS}
                    INFERENCE_TEST_MODEL_FILE="${INFERENCE_TEST_MODEL}"
                    INFERENCE_TEST_CLASSES_FILE="${INFERENCE_TEST_CLASSES}"
                    INFERENCE_TEST_VIDEO_FILE="${INFERENCE_TEST_VIDEO}"
                )
                target_link_libraries(inference_tests_runner PRIVATE
                    GTest::gtest_main
                    ${OpenCV_LIBS}
                    "${ORT_LIBRARY}"
                    Threads::Threads
                    ${CMAKE_DL_LIBS}
                )
                add_test(NAME onnx_inference_tests COMMAND inference_tests_runner)
                set_tests_properties(onnx_inference_tests PROPERTIES TIMEOUT 180)
                message(STATUS "Real inference tests enabled with model: ${INFERENCE_TEST_MODEL}")
            endif()
        endif()
    else()
        message(WARNING "GTest not found: tests will not be built. Install GTest or configure CMake with -DFETCH_GTEST=ON.")
    endif()
endif()
