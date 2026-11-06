#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <csignal>
#include <exception>
#include <iostream>
#include <memory>
#include <string>

#include "config_loader.hpp"
#include "VideoHandler.hpp"
#include "Source.hpp"
#include "Timer.hpp"
#include "Detection.hpp"
#include "DetectorFactory.hpp"

volatile sig_atomic_t keep_running = 1;

void signalHandler(int /*sig*/) {
    keep_running = 0;
}

namespace {
void printUsage(const char* program) {
    std::cout << "Usage: " << program << " [--config <path>]\n"
              << "  -c, --config <path>  Path to config.yaml (default: search ./ then ../).\n"
              << "  -h, --help           Show this help.\n";
}
}

int main(int argc, char** argv) {
    // ── Parse command-line arguments ────────────────────────────────────────────
    std::string config_path;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--config" || arg == "-c") {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << arg << " requires a path argument." << std::endl;
                return 1;
            }
            config_path = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }

    // ── Load config ───────────────────────────────────────────────────────────
#if defined(USE_YAML_CONFIG)
    Config cfg;
    try {
        cfg = loadConfigYAML(resolveConfigPath(config_path));
    } catch (const std::exception& e) {
        // Missing or malformed keys fall back to defaults inside loadConfigYAML;
        // what reaches here is a value that must not be guessed (e.g. an unknown
        // backend name), so stop instead of running something else.
        std::cerr << "Error in config: " << e.what() << std::endl;
        return 1;
    }
#else
    if (!config_path.empty()) {
        std::cerr << "Note: --config is ignored; this build has no YAML support." << std::endl;
    }
    const Config cfg;  // use default values from struct
#endif

    Timer timer;
    timer.start();

    VideoHandler video(cfg.video_output_file);

    std::unique_ptr<Detection> detection;
    try {
        detection = createDetector(cfg);
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialise the detection backend: " << e.what() << std::endl;
        return 1;
    }

    bool source_opened = false;
    if (cfg.source == SOURCE::VIDEO) {
        source_opened = video.open_file(cfg.video_input_file);
    } else if (cfg.source == SOURCE::WEBCAM) {
        source_opened = video.open_webcam(cfg.webcam_index);
    } else {
        std::cerr << "Please select a valid source: VIDEO or WEBCAM?" << std::endl;
        return 1;
    }
    if (!source_opened) {
        return 1;
    }

    if (!detection->load_class_list(cfg.class_names_file)) {
        std::cerr << "Aborting: no usable class names loaded." << std::endl;
        return 1;
    }

    cv::Mat frame;
    signal(SIGINT, signalHandler);
    while (keep_running && video.read(frame)) {
        video.crop_frame(frame);
        try {
            detection->detect(frame);
        } catch (const std::exception& e) {
            // Backends throw when inference itself fails (e.g. Core AI rejecting
            // the input, ONNX Runtime shape errors). Such failures repeat on every
            // frame, so abort instead of writing a video without detections.
            std::cerr << "Aborting: detection failed: " << e.what() << std::endl;
            return 1;
        }
        if (cfg.show_frames) {
            video.showFrame("Object Detector", frame);
        }
        // The writer opens on the first frame, sized to the cropped frames.
        if (!video.write(frame)) {
            std::cerr << "Aborting: could not open or write the output video." << std::endl;
            return 1;
        }
    }

    std::cout << "No more frames to read or error occurred." << std::endl;
    timer.stop();
    std::cout << "Total execution time: " << timer.getElapsedTime() << " seconds" << std::endl;
    return 0;
}
