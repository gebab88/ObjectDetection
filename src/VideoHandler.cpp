#include <filesystem>
#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

#include "VideoHandler.hpp"

// int VideoHandler::fourcc = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));

namespace {
struct WriterCandidate {
    std::string path;
    int backend;
    int fourcc;
    const char* label;
};

std::string lowercaseExtension(const std::string& path) {
    std::string ext = std::filesystem::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return ext;
}

std::string availableFallbackPath(const std::string& requested_path, const std::string& extension) {
    auto fallback = std::filesystem::path(requested_path);
    fallback.replace_extension(extension);
    if (!std::filesystem::exists(fallback)) {
        return fallback.string();
    }

    const auto parent = fallback.parent_path();
    const auto stem = fallback.stem().string();
    auto candidate = parent / (stem + "_fallback" + extension);
    for (int suffix = 1; std::filesystem::exists(candidate); ++suffix) {
        candidate = parent / (stem + "_fallback_" + std::to_string(suffix) + extension);
    }
    return candidate.string();
}

std::vector<WriterCandidate> writerCandidates(const std::string& requested_path) {
    const std::string ext = lowercaseExtension(requested_path);
    std::vector<WriterCandidate> candidates;

    auto add = [&](const std::string& path, int backend, int fourcc, const char* label) {
        candidates.push_back({path, backend, fourcc, label});
    };

    if (ext == ".mp4" || ext == ".m4v" || ext == ".mov") {
        add(requested_path, cv::CAP_ANY, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), "mp4v");
        add(requested_path, cv::CAP_ANY, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), "avc1");
        #ifdef __APPLE__
        add(requested_path, cv::CAP_AVFOUNDATION, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), "AVFoundation/mp4v");
        add(requested_path, cv::CAP_AVFOUNDATION, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), "AVFoundation/avc1");
        #endif
    } else if (ext == ".avi") {
        add(requested_path, cv::CAP_ANY, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), "MJPG");
        add(requested_path, cv::CAP_ANY, cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), "XVID");
    } else {
        add(requested_path, cv::CAP_ANY, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), "MJPG");
    }

    const std::string avi_path = availableFallbackPath(requested_path, ".avi");
    if (avi_path != requested_path) {
        add(avi_path, cv::CAP_ANY, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), "fallback AVI/MJPG");
        add(avi_path, cv::CAP_ANY, cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), "fallback AVI/XVID");
    }

    return candidates;
}
}

VideoHandler::VideoHandler(const std::string output_path) : output_path_(output_path){
}
VideoHandler::~VideoHandler(){
        cap_.release();
        output_.release();
        std::cout << "Released video resources." << std::endl;
        cv::destroyAllWindows();
}

bool VideoHandler::open_file(const std::string &input_path) {
    // Implementation to open video file
    cap_.open(input_path);
    if (!cap_.isOpened()) {
        std::cerr << "Error: Could not open video file: " << input_path << std::endl;
        return false;
    }
    std::cout << "Successfully opened video file: " << input_path << std::endl;
    fps = static_cast<float>(cap_.get(cv::CAP_PROP_FPS));
    frame_width = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
    frame_height = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
    printVideoProperties();
    return true;
}

bool VideoHandler::open_webcam(const int &camera_index) {
    // Implementation to open webcam
    cap_.open(camera_index);
    if (!cap_.isOpened()) {
        std::cerr << "Error: Could not open webcam with index: " << camera_index << std::endl;
        return false;
    }
    std::cout << "Successfully opened webcam with index: " << camera_index << std::endl;
    // Use the camera's reported rate when available; fall back to a conservative
    // 5 fps for devices that do not report a meaningful CAP_PROP_FPS.
    const double reported_fps = cap_.get(cv::CAP_PROP_FPS);
    fps = (reported_fps > 1.0) ? static_cast<float>(reported_fps) : 5.0f;
    frame_width = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
    frame_height = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
    return true;
}

bool VideoHandler::set_video_writer() {
    int min_side_len = std::min(frame_height, frame_width);
    const cv::Size output_size(min_side_len, min_side_len);

    for (const auto& candidate : writerCandidates(output_path_)) {
        if (candidate.path == output_path_ && std::filesystem::exists(candidate.path)) {
            std::filesystem::remove(candidate.path);
            std::cout << "Removed existing file: " << candidate.path << std::endl;
        }

        if (output_.open(candidate.path, candidate.backend, candidate.fourcc, fps, output_size, true)) {
            output_path_ = candidate.path;
            std::cout << "Successfully opened video writer with path: " << output_path_
                      << " (" << candidate.label << ")" << std::endl;
            return true;
        }

        std::cerr << "Warning: Could not open video writer with path: " << candidate.path
                  << " (" << candidate.label << ")" << std::endl;
    }

    std::cerr << "Error: Could not open any video writer for requested path: " << output_path_ << std::endl;
    return false;
}

void VideoHandler::crop_frame( cv::Mat &frame ) {
        // Center-crop the frame to a square so it matches the square model input.
        // NOTE: this discards the left/right (or top/bottom) borders, so objects
        // near the frame edges are lost. Letterboxing (padding instead of cropping)
        // would preserve the full field of view at the cost of some resolution.
        // Derive the crop from the actual frame size rather than the cached capture
        // properties, which can disagree with the decoded frame for some codecs.
        const int min_side_len = std::min(frame.cols, frame.rows);
        const int x = (frame.cols - min_side_len) / 2;
        const int y = (frame.rows - min_side_len) / 2;
        frame = frame(cv::Rect(x, y, min_side_len, min_side_len));
    }

void VideoHandler::printVideoProperties(){
    std::cout << "Video Properties:" << std::endl;
    std::cout << "Frame Width: " << cap_.get(cv::CAP_PROP_FRAME_WIDTH) << std::endl;
    std::cout << "Frame Height: " << cap_.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;
    std::cout << "FPS: " << cap_.get(cv::CAP_PROP_FPS) << std::endl;
    std::cout << "Frame Count: " << cap_.get(cv::CAP_PROP_FRAME_COUNT) << std::endl;
    std::cout << "Format: " << cap_.get(cv::CAP_PROP_FORMAT) << std::endl;
    std::cout << "Mode: " << cap_.get(cv::CAP_PROP_MODE) << std::endl;
}

void VideoHandler::showFrame( const std::string &windowName, const cv::Mat &frame ) {
        cv::imshow(windowName, frame);
        cv::waitKey(1);
        // cv::destroyAllWindows();  // Close the window after displaying each frame
}

void VideoHandler::write( cv::Mat &frame ) {
    output_.write(frame);
}

bool VideoHandler::read( cv::Mat &frame ) {
        return cap_.read(frame);
}
