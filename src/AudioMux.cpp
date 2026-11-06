#include "AudioMux.hpp"

#include <algorithm>
#include <filesystem>
#include <system_error>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/dict.h>
#include <libavutil/error.h>
#include <libavutil/mathematics.h>
}

namespace fs = std::filesystem;

namespace audio_mux {
namespace {

// ── RAII wrappers for the FFmpeg C API ───────────────────────────────────────

struct Input {
    AVFormatContext* ctx = nullptr;
    ~Input() { avformat_close_input(&ctx); }
};

struct Output {
    AVFormatContext* ctx = nullptr;
    ~Output() {
        if (!ctx) return;
        if (ctx->pb && !(ctx->oformat->flags & AVFMT_NOFILE)) avio_closep(&ctx->pb);
        avformat_free_context(ctx);
    }
};

struct Packet {
    AVPacket* pkt = av_packet_alloc();
    ~Packet() { av_packet_free(&pkt); }
};

// Removes the temporary file unless it was moved into place.
struct TempFile {
    fs::path path;
    bool keep = false;
    ~TempFile() {
        if (!keep) {
            std::error_code ec;
            fs::remove(path, ec);
        }
    }
};

std::string avError(int err) {
    char buf[AV_ERROR_MAX_STRING_SIZE] = {};
    av_strerror(err, buf, sizeof(buf));
    return buf;
}

Result fail(const std::string& what, int err) {
    return {Status::Failed, what + ": " + avError(err)};
}

int openInput(const std::string& path, Input& in) {
    int ret = avformat_open_input(&in.ctx, path.c_str(), nullptr, nullptr);
    if (ret < 0) return ret;
    return avformat_find_stream_info(in.ctx, nullptr);
}

// Picks the source audio stream to copy: FFmpeg's preferred one if the output
// container can store its codec, otherwise the first one that fits. iPhones
// recording spatial audio store extra tracks (e.g. APAC) next to the AAC track.
// Returns -1 when the source has no audio, -2 when no audio stream fits.
int pickAudioStream(AVFormatContext* src, const AVOutputFormat* ofmt) {
    auto fits = [&](int index) {
        // 1 = supported, 0 = not supported, < 0 = unknown (let the muxer decide).
        return avformat_query_codec(ofmt, src->streams[index]->codecpar->codec_id,
                                    FF_COMPLIANCE_NORMAL) != 0;
    };

    const int best = av_find_best_stream(src, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
    if (best < 0) return -1;
    if (fits(best)) return best;
    for (unsigned i = 0; i < src->nb_streams; ++i) {
        if (src->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO && fits(static_cast<int>(i))) {
            return static_cast<int>(i);
        }
    }
    return -2;
}

// Start of the source's video in the given time base. OpenCV's first frame is
// the first video frame, so audio is placed relative to it, not to the file.
int64_t sourceVideoStart(AVFormatContext* src, AVRational tb) {
    const int v = av_find_best_stream(src, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (v >= 0 && src->streams[v]->start_time != AV_NOPTS_VALUE) {
        return av_rescale_q(src->streams[v]->start_time, src->streams[v]->time_base, tb);
    }
    if (src->start_time != AV_NOPTS_VALUE) {
        return av_rescale_q(src->start_time, AV_TIME_BASE_Q, tb);
    }
    return 0;
}

// Reads the next packet of the given stream; false at end of file or on error.
bool readStreamPacket(AVFormatContext* ctx, int stream_index, AVPacket* pkt, int& error) {
    while (true) {
        const int ret = av_read_frame(ctx, pkt);
        if (ret < 0) {
            error = (ret == AVERROR_EOF) ? 0 : ret;
            return false;
        }
        if (pkt->stream_index == stream_index) return true;
        av_packet_unref(pkt);
    }
}

}  // namespace

Result copyAudioTrack(const std::string& source_path, const std::string& video_path) {
    // ── Open the processed video and pick the output format from its name ─────
    Input video;
    int ret = openInput(video_path, video);
    if (ret < 0) return fail("cannot open output video " + video_path, ret);
    const int v_index = av_find_best_stream(video.ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (v_index < 0) return {Status::Failed, "no video stream in " + video_path};
    AVStream* v_in = video.ctx->streams[v_index];

    const fs::path final_path(video_path);
    TempFile temp{final_path.parent_path() /
                  (final_path.stem().string() + ".audio-tmp" + final_path.extension().string())};

    Output out;
    ret = avformat_alloc_output_context2(&out.ctx, nullptr, nullptr, temp.path.string().c_str());
    if (ret < 0) return fail("cannot create muxer for " + temp.path.string(), ret);

    // ── Pick the source audio stream ───────────────────────────────────────────
    Input source;
    ret = openInput(source_path, source);
    if (ret < 0) return fail("cannot open source video " + source_path, ret);
    const int a_index = pickAudioStream(source.ctx, out.ctx->oformat);
    if (a_index == -1) return {Status::NoAudio, "source has no audio track"};
    if (a_index == -2) {
        return {Status::Unsupported, std::string("source audio cannot be stored in ")
                                         + out.ctx->oformat->name + " output"};
    }
    AVStream* a_in = source.ctx->streams[a_index];

    // ── Output streams: plain stream copies, no re-encoding ────────────────────
    AVStream* v_out = avformat_new_stream(out.ctx, nullptr);
    AVStream* a_out = avformat_new_stream(out.ctx, nullptr);
    if (!v_out || !a_out) return {Status::Failed, "cannot allocate output streams"};
    for (auto [in, os] : {std::pair{v_in, v_out}, std::pair{a_in, a_out}}) {
        ret = avcodec_parameters_copy(os->codecpar, in->codecpar);
        if (ret < 0) return fail("cannot copy stream parameters", ret);
        os->codecpar->codec_tag = 0;  // let the muxer choose the tag for its container
        os->time_base = in->time_base;
        os->avg_frame_rate = in->avg_frame_rate;
        os->r_frame_rate = in->r_frame_rate;
        av_dict_copy(&os->metadata, in->metadata, 0);  // e.g. audio language
    }

    {
        std::error_code ec;
        fs::remove(temp.path, ec);  // stale leftover of an earlier, aborted run
    }
    ret = avio_open(&out.ctx->pb, temp.path.string().c_str(), AVIO_FLAG_WRITE);
    if (ret < 0) return fail("cannot create " + temp.path.string(), ret);
    ret = avformat_write_header(out.ctx, nullptr);
    if (ret < 0) return fail("cannot write header of " + temp.path.string(), ret);

    // ── Interleave video and audio packets by timestamp ────────────────────────
    // Audio timestamps are shifted so that the source's first video frame maps to
    // the start of the processed video. Audio past the end of the processed video
    // is dropped, so a run stopped early gets a matching sound track.
    const int64_t audio_offset = sourceVideoStart(source.ctx, a_in->time_base)
        - av_rescale_q(v_in->start_time != AV_NOPTS_VALUE ? v_in->start_time : 0,
                       v_in->time_base, a_in->time_base);
    // Fallback length of a video packet without duration (e.g. some AVI files).
    const int64_t frame_duration = (v_in->avg_frame_rate.num > 0)
        ? av_rescale_q(1, av_inv_q(v_in->avg_frame_rate), v_in->time_base) : 0;

    Packet v_pkt, a_pkt;
    int v_err = 0, a_err = 0;
    bool have_v = readStreamPacket(video.ctx, v_index, v_pkt.pkt, v_err);
    bool have_a = false;
    int64_t video_end = AV_NOPTS_VALUE;  // in video time base, known once video is done

    // AAC starts with encoder priming: packets with negative timestamps that the
    // source's edit list hides but the decoder needs for the first audible
    // samples (iPhones: 2112 samples, so the third packet straddles t = 0).
    // MP4/MOV can store negative timestamps and keep the priming as in the
    // source. AVI plays audio back-to-back from t = 0, so there the first packet
    // must be the one covering t = 0. Priming is only meaningful when audio and
    // video start together (offset 0); otherwise everything before the first
    // video frame is dropped.
    const bool keep_priming = (out.ctx->oformat->flags & AVFMT_TS_NEGATIVE) && audio_offset == 0;

    auto nextAudio = [&]() {
        while (readStreamPacket(source.ctx, a_index, a_pkt.pkt, a_err)) {
            AVPacket* pkt = a_pkt.pkt;
            if (pkt->pts == AV_NOPTS_VALUE) {
                av_packet_unref(pkt);
                continue;
            }
            const bool priming = pkt->pts < 0;
            pkt->pts -= audio_offset;
            if (pkt->dts != AV_NOPTS_VALUE) pkt->dts -= audio_offset;
            const bool ends_after_start = pkt->pts + std::max<int64_t>(pkt->duration, 0) > 0;
            if (!ends_after_start && !(keep_priming && priming)) {
                av_packet_unref(pkt);  // audio before the first video frame
                continue;
            }
            return true;
        }
        return false;
    };
    have_a = nextAudio();

    auto write = [&](AVPacket* pkt, AVStream* in, AVStream* os) {
        av_packet_rescale_ts(pkt, in->time_base, os->time_base);
        pkt->stream_index = os->index;
        pkt->pos = -1;
        return av_interleaved_write_frame(out.ctx, pkt);  // takes ownership of the data
    };

    int64_t v_last_end = 0;
    while (have_v || have_a) {
        const bool take_video = have_v &&
            (!have_a || av_compare_ts(v_pkt.pkt->dts != AV_NOPTS_VALUE ? v_pkt.pkt->dts : v_pkt.pkt->pts,
                                      v_in->time_base, a_pkt.pkt->pts, a_in->time_base) <= 0);
        if (take_video) {
            if (v_pkt.pkt->pts != AV_NOPTS_VALUE) {
                const int64_t duration = v_pkt.pkt->duration > 0 ? v_pkt.pkt->duration : frame_duration;
                v_last_end = std::max(v_last_end, v_pkt.pkt->pts + duration);
            }
            ret = write(v_pkt.pkt, v_in, v_out);
            if (ret < 0) return fail("cannot write video packet", ret);
            have_v = readStreamPacket(video.ctx, v_index, v_pkt.pkt, v_err);
            if (!have_v) {
                if (v_err < 0) return fail("cannot read " + video_path, v_err);
                video_end = v_last_end;
            }
        } else {
            if (video_end != AV_NOPTS_VALUE &&
                av_compare_ts(a_pkt.pkt->pts, a_in->time_base, video_end, v_in->time_base) >= 0) {
                break;  // rest of the audio lies beyond the processed video
            }
            ret = write(a_pkt.pkt, a_in, a_out);
            if (ret < 0) return fail("cannot write audio packet", ret);
            have_a = nextAudio();
            if (!have_a && a_err < 0) return fail("cannot read " + source_path, a_err);
        }
    }

    ret = av_write_trailer(out.ctx);
    if (ret < 0) return fail("cannot finish " + temp.path.string(), ret);
    ret = avio_closep(&out.ctx->pb);
    if (ret < 0) return fail("cannot close " + temp.path.string(), ret);

    // ── Replace the video-only output ──────────────────────────────────────────
    std::error_code ec;
    fs::rename(temp.path, final_path, ec);
    if (ec) return {Status::Failed, "cannot replace " + video_path + ": " + ec.message()};
    temp.keep = true;

    return {Status::Added, std::string("copied ") + avcodec_get_name(a_in->codecpar->codec_id)
                               + " audio track from " + source_path};
}

}  // namespace audio_mux
