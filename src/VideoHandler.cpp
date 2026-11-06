#include <filesystem>
#include <algorithm>
#include <cctype>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "VideoHandler.hpp"
#include "FrameCrop.hpp"

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

VideoHandler::VideoHandler(std::string output_path) : output_path_(std::move(output_path)){
}

float VideoHandler::sanitizedFps(double reported_fps) {
    // Use the source's reported rate when meaningful; fall back to a
    // conservative 5 fps for sources that do not report a usable
    // CAP_PROP_FPS (some containers and webcams report 0). A zero rate
    // would make every writer candidate fail to open.
    return (reported_fps > 1.0) ? static_cast<float>(reported_fps) : 5.0f;
}
VideoHandler::~VideoHandler(){
        cap_.release();
        output_.release();
        std::cout << "Released video resources." << std::endl;
        cv::destroyAllWindows();
}

bool VideoHandler::open_file(const std::string &input_path) {
    // Implementation to open video file
    cap_.open(input_path, VideoHandler::preferredCaptureBackend());
    if (!cap_.isOpened()) {
        std::cerr << "Error: Could not open video file: " << input_path << std::endl;
        return false;
    }
    std::cout << "Successfully opened video file: " << input_path << std::endl;
    fps = sanitizedFps(cap_.get(cv::CAP_PROP_FPS));
    printVideoProperties();
    return true;
}

bool VideoHandler::open_webcam(const int &camera_index) {
    // Implementation to open webcam
    cap_.open(camera_index, VideoHandler::preferredCaptureBackend());
    if (!cap_.isOpened()) {
        std::cerr << "Error: Could not open webcam with index: " << camera_index << std::endl;
        return false;
    }
    std::cout << "Successfully opened webcam with index: " << camera_index << std::endl;
    fps = sanitizedFps(cap_.get(cv::CAP_PROP_FPS));
    return true;
}

bool VideoHandler::openWriter(const cv::Size &output_size) {
    for (const auto& candidate : writerCandidates(output_path_)) {
        // Use the error_code overloads: the throwing ones would escape as an
        // uncaught std::filesystem::filesystem_error (e.g. permission denied on a
        // read-only directory) and terminate the program instead of failing cleanly.
        std::error_code ec;
        if (candidate.path == output_path_ && std::filesystem::exists(candidate.path, ec)) {
            if (std::filesystem::is_symlink(candidate.path, ec)) {
                std::cerr << "Error: output path is a symlink, refusing to overwrite: "
                          << candidate.path << std::endl;
                return false;
            }
            if (!std::filesystem::remove(candidate.path, ec) || ec) {
                std::cerr << "Error: Could not remove existing output file " << candidate.path
                          << ": " << (ec ? ec.message() : std::string("not removed")) << std::endl;
                return false;
            }
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
        // Center-crop to a square matching the model input. The crop is derived
        // from the actual frame size (not the cached capture properties, which can
        // disagree with the decoded frame for some codecs). See FrameCrop.hpp for
        // the field-of-view trade-off this implies.
        frame = frame(frame_crop::centeredSquare(frame.cols, frame.rows));
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

bool VideoHandler::write( cv::Mat &frame ) {
    // The writer is opened lazily on the first frame so its size matches the
    // frames actually written; the cached capture properties can disagree with
    // the decoded (and cropped) frame size for some codecs, and cv::VideoWriter
    // silently drops frames whose size differs from the writer's.
    if (!output_.isOpened() && !openWriter(frame.size())) {
        return false;
    }
    output_.write(frame);
    return true;
}

bool VideoHandler::read( cv::Mat &frame ) {
        return cap_.read(frame);
}
