#ifndef AUDIO_MUX_HPP
#define AUDIO_MUX_HPP

#include <string>

// Copies the audio track of the input video into the video written by OpenCV.
//
// cv::VideoWriter only writes images, so the processed video has no sound.
// copyAudioTrack() remuxes the finished file: the video stream and the source's
// audio stream are copied packet by packet (no re-encoding) into a temporary
// file next to the output, which then replaces the output. The audio keeps its
// offset to the source's first video frame and is cut at the end of the
// processed video (e.g. after Ctrl+C).
//
// Only available when the project is built with the FFmpeg libraries
// (HAVE_AUDIO_PASSTHROUGH). On any failure the video-only output is left as is.
namespace audio_mux {

enum class Status {
    Added,        // output now contains the source's audio track
    NoAudio,      // source has no audio track; output unchanged
    Unsupported,  // source audio cannot be stored in the output container; output unchanged
    Failed,       // I/O or FFmpeg error; output unchanged
};

struct Result {
    Status status = Status::Failed;
    std::string message;  // human-readable detail for logs
};

Result copyAudioTrack(const std::string& source_path, const std::string& video_path);

}  // namespace audio_mux

#endif  // AUDIO_MUX_HPP
