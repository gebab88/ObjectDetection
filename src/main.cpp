#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <csignal>
#include <iostream>
#include <memory>

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

int main() {
    // std::cout << cv::getBuildInformation() << std::endl;

    // ── Load config ───────────────────────────────────────────────────────────
#if defined(USE_YAML_CONFIG)
    const Config cfg = loadConfigYAML("config.yaml");
#else
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

    if (!video.set_video_writer()) {
        return 1;
    }
    detection->load_class_list(cfg.class_names_file);

    cv::Mat frame;
    signal(SIGINT, signalHandler);
    while (keep_running && video.read(frame)) {
        video.crop_frame(frame);
        detection->detect(frame);
        if (cfg.show_frames) {
            video.showFrame("Object Detector", frame);
        }
        video.write(frame);
    }

    std::cout << "No more frames to read or error occurred." << std::endl;
    timer.stop();
    std::cout << "Total execution time: " << timer.getElapsedTime() << " seconds" << std::endl;
    return 0;
}
