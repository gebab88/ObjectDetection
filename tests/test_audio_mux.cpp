// Tests for audio_mux::copyAudioTrack (audio passthrough into the OpenCV output).
// Built only when the FFmpeg libraries are available (HAVE_AUDIO_PASSTHROUGH).

#include <gtest/gtest.h>

#include "AudioMux.hpp"

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/opt.h>
}

namespace fs = std::filesystem;

namespace {

// ── Fixture helpers ───────────────────────────────────────────────────────────

// Video-only file as written by the application (cv::VideoWriter).
void writeVideoOnly(const fs::path& path, int frames, double fps, int fourcc) {
    cv::VideoWriter writer(path.string(), fourcc, fps, cv::Size(64, 64), true);
    if (!writer.isOpened()) {
        throw std::runtime_error("cannot open VideoWriter for " + path.string());
    }
    for (int i = 0; i < frames; ++i) {
        cv::Mat frame(64, 64, CV_8UC3, cv::Scalar(i % 256, 64, 128));
        writer.write(frame);
    }
}

void writeMp4VideoOnly(const fs::path& path, int frames, double fps) {
    writeVideoOnly(path, frames, fps, cv::VideoWriter::fourcc('m', 'p', '4', 'v'));
}

// Audio file with an encoded sine tone (e.g. AAC in .m4a, like an iPhone track).
// pts_shift (in samples) moves all timestamps; see kIphoneLikeShift.
void writeAudioFile(const fs::path& path, AVCodecID codec_id, double seconds, int64_t pts_shift = 0) {
    constexpr int kSampleRate = 48000;

    const AVCodec* codec = avcodec_find_encoder(codec_id);
    if (!codec) throw std::runtime_error("encoder not available");

    AVFormatContext* fmt = nullptr;
    if (avformat_alloc_output_context2(&fmt, nullptr, nullptr, path.string().c_str()) < 0) {
        throw std::runtime_error("cannot create muxer for " + path.string());
    }
    std::unique_ptr<AVFormatContext, void (*)(AVFormatContext*)> fmt_guard(
        fmt, [](AVFormatContext* f) {
            if (f->pb) avio_closep(&f->pb);
            avformat_free_context(f);
        });

    AVCodecContext* enc = avcodec_alloc_context3(codec);
    std::unique_ptr<AVCodecContext, void (*)(AVCodecContext*)> enc_guard(
        enc, [](AVCodecContext* c) { avcodec_free_context(&c); });

    const void* formats = nullptr;
    int num_formats = 0;
    avcodec_get_supported_config(enc, codec, AV_CODEC_CONFIG_SAMPLE_FORMAT, 0, &formats, &num_formats);
    enc->sample_fmt = (formats && num_formats > 0)
        ? static_cast<const AVSampleFormat*>(formats)[0] : AV_SAMPLE_FMT_FLTP;
    enc->sample_rate = kSampleRate;
    av_channel_layout_default(&enc->ch_layout, 2);
    enc->bit_rate = 128000;
    enc->time_base = AVRational{1, kSampleRate};
    if (fmt->oformat->flags & AVFMT_GLOBALHEADER) enc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    if (avcodec_open2(enc, codec, nullptr) < 0) throw std::runtime_error("cannot open encoder");

    AVStream* st = avformat_new_stream(fmt, nullptr);
    avcodec_parameters_from_context(st->codecpar, enc);
    st->time_base = enc->time_base;
    if (avio_open(&fmt->pb, path.string().c_str(), AVIO_FLAG_WRITE) < 0) throw std::runtime_error("cannot open file");
    if (avformat_write_header(fmt, nullptr) < 0) throw std::runtime_error("cannot write header");

    AVFrame* frame = av_frame_alloc();
    AVPacket* pkt = av_packet_alloc();
    const int frame_size = enc->frame_size > 0 ? enc->frame_size : 1024;
    const int64_t total = static_cast<int64_t>(seconds * kSampleRate);

    auto drain = [&]() {
        while (avcodec_receive_packet(enc, pkt) == 0) {
            av_packet_rescale_ts(pkt, enc->time_base, st->time_base);
            pkt->stream_index = st->index;
            av_interleaved_write_frame(fmt, pkt);
        }
    };

    for (int64_t pos = 0; pos < total; pos += frame_size) {
        frame->nb_samples = static_cast<int>(std::min<int64_t>(frame_size, total - pos));
        if (enc->frame_size > 0 && !(codec->capabilities & AV_CODEC_CAP_VARIABLE_FRAME_SIZE)) {
            frame->nb_samples = frame_size;  // pad the last frame
        }
        frame->format = enc->sample_fmt;
        frame->sample_rate = kSampleRate;
        av_channel_layout_copy(&frame->ch_layout, &enc->ch_layout);
        av_frame_get_buffer(frame, 0);
        av_frame_make_writable(frame);
        const bool planar = av_sample_fmt_is_planar(enc->sample_fmt);
        for (int i = 0; i < frame->nb_samples; ++i) {
            const double v = 0.3 * std::sin(2.0 * M_PI * 440.0 * double(pos + i) / kSampleRate);
            for (int ch = 0; ch < 2; ++ch) {
                uint8_t* plane = frame->data[planar ? ch : 0];
                const int idx = planar ? i : i * 2 + ch;
                switch (enc->sample_fmt) {
                    case AV_SAMPLE_FMT_FLTP: case AV_SAMPLE_FMT_FLT:
                        reinterpret_cast<float*>(plane)[idx] = static_cast<float>(v); break;
                    case AV_SAMPLE_FMT_S16P: case AV_SAMPLE_FMT_S16:
                        reinterpret_cast<int16_t*>(plane)[idx] = static_cast<int16_t>(v * 32767); break;
                    case AV_SAMPLE_FMT_S32P: case AV_SAMPLE_FMT_S32:
                        reinterpret_cast<int32_t*>(plane)[idx] = static_cast<int32_t>(v * 2147483647.0); break;
                    default: throw std::runtime_error("unsupported test sample format");
                }
            }
        }
        frame->pts = pos + pts_shift;
        avcodec_send_frame(enc, frame);
        av_frame_unref(frame);
        drain();
    }
    avcodec_send_frame(enc, nullptr);
    drain();
    av_write_trailer(fmt);
    av_packet_free(&pkt);
    av_frame_free(&frame);
}

// ── Inspection helpers ────────────────────────────────────────────────────────

struct MediaInfo {
    int video_streams = 0;
    int audio_streams = 0;
    int video_packets = 0;
    double audio_end_seconds = 0.0;  // end of the last audio packet
    double audio_first_pts_seconds = 0.0;
    int audio_packets = 0;
    int audio_packets_ending_after_zero = 0;  // packets with pts + duration > 0
    AVCodecID audio_codec = AV_CODEC_ID_NONE;
};

MediaInfo probe(const fs::path& path) {
    AVFormatContext* fmt = nullptr;
    if (avformat_open_input(&fmt, path.string().c_str(), nullptr, nullptr) < 0) {
        throw std::runtime_error("cannot open " + path.string());
    }
    avformat_find_stream_info(fmt, nullptr);
    MediaInfo info;
    for (unsigned i = 0; i < fmt->nb_streams; ++i) {
        const auto type = fmt->streams[i]->codecpar->codec_type;
        if (type == AVMEDIA_TYPE_VIDEO) ++info.video_streams;
        if (type == AVMEDIA_TYPE_AUDIO) {
            ++info.audio_streams;
            info.audio_codec = fmt->streams[i]->codecpar->codec_id;
        }
    }
    AVPacket* pkt = av_packet_alloc();
    while (av_read_frame(fmt, pkt) >= 0) {
        const AVStream* st = fmt->streams[pkt->stream_index];
        if (st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) ++info.video_packets;
        if (st->codecpar->codec_type == AVMEDIA_TYPE_AUDIO && pkt->pts != AV_NOPTS_VALUE) {
            const double start = double(pkt->pts) * av_q2d(st->time_base);
            const double end = double(pkt->pts + pkt->duration) * av_q2d(st->time_base);
            if (info.audio_packets == 0) info.audio_first_pts_seconds = start;
            ++info.audio_packets;
            if (end > 0.0) ++info.audio_packets_ending_after_zero;
            info.audio_end_seconds = std::max(info.audio_end_seconds, end);
        }
        av_packet_unref(pkt);
    }
    av_packet_free(&pkt);
    avformat_close_input(&fmt);
    return info;
}

std::string readBytes(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

class AudioMuxTest : public ::testing::Test {
protected:
    void SetUp() override {
        av_log_set_level(AV_LOG_ERROR);  // silence encoder statistics in the test output
        dir_ =fs::temp_directory_path() /
               ("audio_mux_test_" + std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
        fs::remove_all(dir_);
        fs::create_directories(dir_);
    }
    void TearDown() override { fs::remove_all(dir_); }

    std::vector<fs::path> filesInDir() const {
        std::vector<fs::path> files;
        for (const auto& e : fs::directory_iterator(dir_)) files.push_back(e.path().filename());
        return files;
    }

    fs::path dir_;
};

// FFmpeg's AAC encoder primes with 1024 samples, so packets start at -1024, 0,
// 1024, ... Shifting by -1088 gives the iPhone layout (2112 priming samples):
// -2112, -1088, -64, 960, ... where the packet at -64 straddles t = 0.
constexpr int64_t kIphoneLikeShift = -1088;

// 1024-sample AAC frames at 48 kHz: the cut can only happen on a frame border.
constexpr double kAacFrameSeconds = 1024.0 / 48000.0;

}  // namespace

// ── Tests ─────────────────────────────────────────────────────────────────────

TEST_F(AudioMuxTest, AddsSourceAudioToMp4Output) {
    const auto source = dir_ / "source.m4a";
    const auto output = dir_ / "output.mp4";
    writeAudioFile(source, AV_CODEC_ID_AAC, 3.0);
    writeMp4VideoOnly(output, 50, 25.0);  // 2.0 s

    const auto result = audio_mux::copyAudioTrack(source.string(), output.string());

    ASSERT_EQ(result.status, audio_mux::Status::Added) << result.message;
    const auto info = probe(output);
    EXPECT_EQ(info.video_streams, 1);
    EXPECT_EQ(info.audio_streams, 1);
    EXPECT_EQ(info.audio_codec, AV_CODEC_ID_AAC);
    EXPECT_EQ(info.video_packets, 50) << "video must be copied, not re-encoded or truncated";
}

TEST_F(AudioMuxTest, CutsLongerAudioAtEndOfVideo) {
    const auto source = dir_ / "source.m4a";
    const auto output = dir_ / "output.mp4";
    writeAudioFile(source, AV_CODEC_ID_AAC, 3.0);
    writeMp4VideoOnly(output, 50, 25.0);  // 2.0 s, e.g. run interrupted with Ctrl+C

    ASSERT_EQ(audio_mux::copyAudioTrack(source.string(), output.string()).status, audio_mux::Status::Added);

    EXPECT_NEAR(probe(output).audio_end_seconds, 2.0, kAacFrameSeconds);
}

TEST_F(AudioMuxTest, KeepsShorterAudioComplete) {
    const auto source = dir_ / "source.m4a";
    const auto output = dir_ / "output.mp4";
    writeAudioFile(source, AV_CODEC_ID_AAC, 1.0);
    writeMp4VideoOnly(output, 50, 25.0);

    ASSERT_EQ(audio_mux::copyAudioTrack(source.string(), output.string()).status, audio_mux::Status::Added);

    EXPECT_NEAR(probe(output).audio_end_seconds, 1.0, 2 * kAacFrameSeconds);
}

TEST_F(AudioMuxTest, AddsSourceAudioToAviOutput) {
    const auto source = dir_ / "source.m4a";
    const auto output = dir_ / "output.avi";
    writeAudioFile(source, AV_CODEC_ID_AAC, 2.0);
    writeVideoOnly(output, 20, 10.0, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));

    const auto result = audio_mux::copyAudioTrack(source.string(), output.string());

    ASSERT_EQ(result.status, audio_mux::Status::Added) << result.message;
    const auto info = probe(output);
    EXPECT_EQ(info.video_streams, 1);
    EXPECT_EQ(info.audio_streams, 1);
    EXPECT_EQ(info.video_packets, 20);
}

// AAC starts with encoder priming ("pre-roll"): the first packets have negative
// timestamps and are needed to decode the first audible samples. iPhone clips
// carry 2112 priming samples. MP4/MOV can store them (edit list), so they must
// be kept exactly as in the source.
TEST_F(AudioMuxTest, KeepsAacPrimingInMp4Output) {
    const auto source = dir_ / "source.m4a";
    const auto output = dir_ / "output.mp4";
    writeAudioFile(source, AV_CODEC_ID_AAC, 3.0, kIphoneLikeShift);
    writeMp4VideoOnly(output, 50, 25.0);
    const auto src = probe(source);
    ASSERT_LT(src.audio_first_pts_seconds, 0.0) << "fixture must have AAC priming";

    ASSERT_EQ(audio_mux::copyAudioTrack(source.string(), output.string()).status, audio_mux::Status::Added);

    EXPECT_DOUBLE_EQ(probe(output).audio_first_pts_seconds, src.audio_first_pts_seconds);
}

// AVI has no per-packet audio timestamps: audio plays back-to-back from t = 0.
// To stay in sync, the output must start with the packet covering t = 0 and
// drop the priming packets before it.
TEST_F(AudioMuxTest, AviOutputStartsWithAudioPacketCoveringTimeZero) {
    const auto source = dir_ / "source.m4a";
    const auto output = dir_ / "output.avi";
    writeAudioFile(source, AV_CODEC_ID_AAC, 1.0, kIphoneLikeShift);  // shorter than the video
    writeVideoOnly(output, 20, 10.0, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    const auto src = probe(source);
    ASSERT_LT(src.audio_first_pts_seconds, 0.0) << "fixture must have AAC priming";

    ASSERT_EQ(audio_mux::copyAudioTrack(source.string(), output.string()).status, audio_mux::Status::Added);

    EXPECT_EQ(probe(output).audio_packets, src.audio_packets_ending_after_zero);
}

TEST_F(AudioMuxTest, SourceWithoutAudioLeavesOutputUnchanged) {
    const auto source = dir_ / "silent_source.mp4";
    const auto output = dir_ / "output.mp4";
    writeMp4VideoOnly(source, 25, 25.0);
    writeMp4VideoOnly(output, 50, 25.0);
    const auto before = readBytes(output);

    const auto result = audio_mux::copyAudioTrack(source.string(), output.string());

    EXPECT_EQ(result.status, audio_mux::Status::NoAudio) << result.message;
    EXPECT_EQ(readBytes(output), before);
}

TEST_F(AudioMuxTest, MissingSourceFailsAndLeavesOutputUnchanged) {
    const auto output = dir_ / "output.mp4";
    writeMp4VideoOnly(output, 50, 25.0);
    const auto before = readBytes(output);

    const auto result = audio_mux::copyAudioTrack((dir_ / "missing.mov").string(), output.string());

    EXPECT_EQ(result.status, audio_mux::Status::Failed);
    EXPECT_FALSE(result.message.empty());
    EXPECT_EQ(readBytes(output), before);
}

TEST_F(AudioMuxTest, MissingVideoFails) {
    const auto source = dir_ / "source.m4a";
    writeAudioFile(source, AV_CODEC_ID_AAC, 1.0);

    const auto result = audio_mux::copyAudioTrack(source.string(), (dir_ / "missing.mp4").string());

    EXPECT_EQ(result.status, audio_mux::Status::Failed);
    EXPECT_FALSE(fs::exists(dir_ / "missing.mp4"));
}

TEST_F(AudioMuxTest, LeavesNoTemporaryFilesBehind) {
    const auto source = dir_ / "source.m4a";
    const auto output = dir_ / "output.mp4";
    writeAudioFile(source, AV_CODEC_ID_AAC, 1.0);
    writeMp4VideoOnly(output, 25, 25.0);

    ASSERT_EQ(audio_mux::copyAudioTrack(source.string(), output.string()).status, audio_mux::Status::Added);

    const auto files = filesInDir();
    EXPECT_EQ(files.size(), 2u) << "only source and output may remain";
}

// ── VideoHandler integration ──────────────────────────────────────────────────
// finish_output() closes the writer and copies the audio of the file opened
// with open_file() into the processed video.

#include "VideoHandler.hpp"

namespace {
// Source clip like an iPhone recording: video plus AAC audio in one file.
fs::path writeSourceWithAudio(const fs::path& dir, int frames, double fps, double audio_seconds) {
    const auto video = dir / "src_video.mp4";
    const auto audio = dir / "src_audio.m4a";
    writeMp4VideoOnly(video, frames, fps);
    writeAudioFile(audio, AV_CODEC_ID_AAC, audio_seconds);
    if (audio_mux::copyAudioTrack(audio.string(), video.string()).status != audio_mux::Status::Added) {
        throw std::runtime_error("cannot build source clip");
    }
    fs::remove(audio);
    const auto source = dir / "source.mp4";
    fs::rename(video, source);
    return source;
}

// Reads every frame of the source and writes it to the output, like main().
int processAllFrames(VideoHandler& handler) {
    cv::Mat frame;
    int frames = 0;
    while (handler.read(frame)) {
        if (!handler.write(frame)) throw std::runtime_error("write failed");
        ++frames;
    }
    return frames;
}
}  // namespace

TEST_F(AudioMuxTest, FinishOutputCopiesAudioOfInputFile) {
    const auto source = writeSourceWithAudio(dir_, 50, 25.0, 2.0);
    const auto output = dir_ / "output.mp4";

    VideoHandler handler(output.string());
    ASSERT_TRUE(handler.open_file(source.string()));
    ASSERT_EQ(processAllFrames(handler), 50);
    handler.finish_output();

    const auto info = probe(output);
    EXPECT_EQ(info.video_streams, 1);
    EXPECT_EQ(info.audio_streams, 1);
    EXPECT_EQ(info.video_packets, 50);
    EXPECT_NEAR(info.audio_end_seconds, 2.0, 2 * kAacFrameSeconds);
}

TEST_F(AudioMuxTest, FinishOutputWithoutWrittenFramesCreatesNoFile) {
    const auto source = writeSourceWithAudio(dir_, 10, 25.0, 0.5);
    const auto output = dir_ / "output.mp4";

    VideoHandler handler(output.string());
    ASSERT_TRUE(handler.open_file(source.string()));
    EXPECT_NO_THROW(handler.finish_output());

    EXPECT_FALSE(fs::exists(output));
}

TEST_F(AudioMuxTest, FinishOutputKeepsVideoWhenSourceHasNoAudio) {
    const auto source = dir_ / "silent.mp4";
    writeMp4VideoOnly(source, 25, 25.0);
    const auto output = dir_ / "output.mp4";

    VideoHandler handler(output.string());
    ASSERT_TRUE(handler.open_file(source.string()));
    processAllFrames(handler);
    handler.finish_output();

    const auto info = probe(output);
    EXPECT_EQ(info.video_streams, 1);
    EXPECT_EQ(info.audio_streams, 0);
    EXPECT_EQ(info.video_packets, 25);
}
