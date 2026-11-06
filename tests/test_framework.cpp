#include <gtest/gtest.h>

#include "Framework.hpp"
#include "YoloPostprocess.hpp"
#include "FrameCrop.hpp"
#include "config_loader.hpp"
#include "DetectorFactory.hpp"
#include "VideoHandler.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

// Tests for detectModelFormat helper

namespace {
// Scoped capture of a std::ostream (std::cout / std::cerr) via rdbuf
// redirection. Replaces testing::internal::CaptureStdout/CaptureStderr, which
// are not part of GTest's public API.
class StreamCapture {
public:
    explicit StreamCapture(std::ostream& stream)
        : stream_(stream), old_buf_(stream.rdbuf(captured_.rdbuf())) {}
    ~StreamCapture() { stream_.rdbuf(old_buf_); }

    StreamCapture(const StreamCapture&) = delete;
    StreamCapture& operator=(const StreamCapture&) = delete;

    std::string str() const { return captured_.str(); }

private:
    std::ostream& stream_;
    std::ostringstream captured_;
    std::streambuf* old_buf_;
};

// These decode tests assert on the labels printed to stdout, so opt into the
// detection logging that is disabled by default on the real-time path.
const bool kDetectionLoggingEnabled = [] {
    yolo_postprocess::logDetectionsToStdout() = true;
    return true;
}();

int changedPixelCount(const cv::Mat& before, const cv::Mat& after) {
    cv::Mat diff;
    cv::absdiff(before, after, diff);

    cv::Mat gray;
    cv::cvtColor(diff, gray, cv::COLOR_BGR2GRAY);

    cv::Mat mask;
    cv::threshold(gray, mask, 0, 255, cv::THRESH_BINARY);
    return cv::countNonZero(mask);
}

void expectYellowPixelNear(const cv::Mat& frame, int x, int y) {
    bool found = false;
    for (int yy = std::max(0, y - 1); yy <= std::min(frame.rows - 1, y + 1); ++yy) {
        for (int xx = std::max(0, x - 1); xx <= std::min(frame.cols - 1, x + 1); ++xx) {
            const cv::Vec3b pixel = frame.at<cv::Vec3b>(yy, xx);
            if (pixel[0] == 0 && pixel[1] == 255 && pixel[2] == 255) {
                found = true;
            }
        }
    }
    EXPECT_TRUE(found) << "Expected yellow bounding-box pixel near (" << x << ", " << y << ")";
}

size_t countOccurrences(const std::string& text, const std::string& needle) {
    size_t count = 0;
    size_t pos = 0;
    while ((pos = text.find(needle, pos)) != std::string::npos) {
        ++count;
        pos += needle.size();
    }
    return count;
}
}

TEST(DetectModelFormat, HefRequiresCompiledHailoBackend) {
    for (const auto* filename : {"nanodet.hef", "yolo26m.HEF", "yoloe.hef", "yolo12m.hef"}) {
#ifdef HAVE_HAILO
        EXPECT_EQ(detectModelFormat(filename), MODEL_FORMAT::HEF);
        EXPECT_EQ(supportedBackendsFor(detectModelFormat(filename)),
                  std::vector<FRAMEWORK>{FRAMEWORK::Hailo});
#else
        EXPECT_EQ(detectModelFormat(filename), MODEL_FORMAT::Unknown);
        EXPECT_TRUE(supportedBackendsFor(detectModelFormat(filename)).empty());
#endif
    }
}

TEST(DetectModelFormat, ReturnsYOLO12ForYolo12m) {
    EXPECT_EQ(detectModelFormat("/path/to/yolo12m.onnx"), MODEL_FORMAT::YOLO12);
}

TEST(DetectModelFormat, ReturnsYOLO12ForYolo12x) {
    EXPECT_EQ(detectModelFormat("yolo12x.onnx"), MODEL_FORMAT::YOLO12);
}

TEST(DetectModelFormat, ReturnsYOLO26ForYolo26m) {
    EXPECT_EQ(detectModelFormat("models/yolo26m.onnx"), MODEL_FORMAT::YOLO26);
}

TEST(DetectModelFormat, ReturnsYOLO26ForYolo26x) {
    EXPECT_EQ(detectModelFormat("yolo26x.onnx"), MODEL_FORMAT::YOLO26);
}

TEST(DetectModelFormat, ReturnsYOLOEForYoloE11Seg) {
    EXPECT_EQ(detectModelFormat("models/yoloe-11l-seg.onnx"), MODEL_FORMAT::YOLOE);
}

TEST(DetectModelFormat, ReturnsYOLOEForYoloE26Seg) {
    EXPECT_EQ(detectModelFormat("yoloe-26l-seg.onnx"), MODEL_FORMAT::YOLOE);
}

TEST(DetectModelFormat, ReturnsYOLOEForYoloE26PromptFree) {
    EXPECT_EQ(detectModelFormat("yoloe-26m-seg-pf.onnx"), MODEL_FORMAT::YOLOE);
}

TEST(DetectModelFormat, ReturnsUnknownForOtherFiles) {
    EXPECT_EQ(detectModelFormat("some_model.onnx"), MODEL_FORMAT::Unknown);
}

TEST(DetectModelFormat, ReturnsYOLO26ForYolo26Aimodel) {
    EXPECT_EQ(detectModelFormat("models/yolo26m.aimodel"), MODEL_FORMAT::YOLO26);
}

TEST(YoloPostprocess, DecodeRawDrawsBoundingBoxAndLabel) {
    constexpr int fields = 7; // x, y, w, h, 3 class scores
    constexpr int detections = 8;
    std::vector<float> output(fields * detections, 0.01f);

    auto set_value = [&](int field, int det, float value) {
        output[field * detections + det] = value;
    };

    set_value(0, 3, 50.0f);
    set_value(1, 3, 50.0f);
    set_value(2, 3, 20.0f);
    set_value(3, 3, 20.0f);
    set_value(4, 3, 0.05f);
    set_value(5, 3, 0.92f);
    set_value(6, 3, 0.10f);

    cv::Mat frame(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
    const cv::Mat before = frame.clone();
    const std::vector<std::string> classes = {"person", "helmet", "forklift"};

    StreamCapture capture(std::cout);
    const bool decoded = yolo_postprocess::decodeRaw(output.data(), fields, detections, frame,
                                                     cv::Size2f(100, 100), classes,
                                                     0.5f, 0.5f);
    const std::string stdout_text = capture.str();

    EXPECT_TRUE(decoded);
    EXPECT_GT(changedPixelCount(before, frame), 0);
    expectYellowPixelNear(frame, 40, 40);
    EXPECT_NE(stdout_text.find("helmet"), std::string::npos);
}

TEST(YoloPostprocess, DecodeEndToEndDrawsBoundingBoxAndLabel) {
    constexpr int detections = 2;
    constexpr int fields = 6; // x1, y1, x2, y2, score, class id
    std::vector<float> output(detections * fields, 0.0f);

    output[0 * fields + 0] = 20.0f;
    output[0 * fields + 1] = 20.0f;
    output[0 * fields + 2] = 70.0f;
    output[0 * fields + 3] = 80.0f;
    output[0 * fields + 4] = 0.95f;
    output[0 * fields + 5] = 1.0f;

    cv::Mat frame(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
    const cv::Mat before = frame.clone();
    const std::vector<std::string> classes = {"person", "truck"};

    StreamCapture capture(std::cout);
    const bool decoded = yolo_postprocess::decodeEndToEnd(output.data(), detections, fields, frame,
                                                          cv::Size2f(100, 100), classes, 0.5f, 0.5f);
    const std::string stdout_text = capture.str();

    EXPECT_TRUE(decoded);
    EXPECT_GT(changedPixelCount(before, frame), 0);
    expectYellowPixelNear(frame, 20, 20);
    EXPECT_NE(stdout_text.find("truck"), std::string::npos);
}

namespace {
// Two strongly overlapping end-to-end detections (IoU ~0.9), e.g. a rider on a horse.
std::string decodeOverlappingPair(int first_class, int second_class) {
    constexpr int detections = 2;
    constexpr int fields = 6; // x1, y1, x2, y2, score, class id
    const std::vector<float> output = {
        20.0f, 20.0f, 80.0f, 80.0f, 0.90f, static_cast<float>(first_class),
        22.0f, 22.0f, 80.0f, 82.0f, 0.80f, static_cast<float>(second_class),
    };
    cv::Mat frame(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
    const std::vector<std::string> classes = {"person", "horse"};

    StreamCapture capture(std::cout);
    EXPECT_TRUE(yolo_postprocess::decodeEndToEnd(output.data(), detections, fields, frame,
                                                 cv::Size2f(100, 100), classes, 0.5f, 0.5f));
    return capture.str();
}
} // namespace

TEST(YoloPostprocess, NmsKeepsOverlappingDetectionsOfDifferentClasses) {
    // NMS must run per class (as in Ultralytics): an overlapping box of another
    // class is a different object, not a duplicate.
    const std::string stdout_text = decodeOverlappingPair(0, 1);
    EXPECT_EQ(countOccurrences(stdout_text, "person"), 1u);
    EXPECT_EQ(countOccurrences(stdout_text, "horse"), 1u);
}

TEST(YoloPostprocess, NmsSuppressesOverlappingDetectionsOfSameClass) {
    const std::string stdout_text = decodeOverlappingPair(0, 0);
    EXPECT_EQ(countOccurrences(stdout_text, "person  90%"), 1u);
    EXPECT_EQ(countOccurrences(stdout_text, "person"), 1u);
}

TEST(YoloPostprocess, ClassAwareNmsKeepsBestPerClassOrderedByScore) {
    const std::vector<cv::Rect> boxes = {
        {10, 10, 50, 50},  // 0: class 1, 0.60 -> suppressed by box 2
        {12, 12, 50, 50},  // 1: class 0, 0.70 -> kept (different class)
        {11, 11, 50, 50},  // 2: class 1, 0.95 -> kept
        {70, 70, 20, 20},  // 3: class 1, 0.80 -> kept (no overlap)
    };
    const std::vector<float> scores = {0.60f, 0.70f, 0.95f, 0.80f};
    const std::vector<int> class_ids = {1, 0, 1, 1};

    const std::vector<int> kept = yolo_postprocess::classAwareNms(boxes, scores, class_ids, 0.5f, 0.5f);

    EXPECT_EQ(kept, (std::vector<int>{2, 3, 1}));
}

TEST(YoloPostprocess, ClassAwareNmsRejectsMismatchedInputs) {
    const std::vector<cv::Rect> boxes = {{10, 10, 50, 50}, {70, 70, 20, 20}};
    const std::vector<float> scores = {0.9f, 0.8f};
    const std::vector<int> class_ids = {0};

    EXPECT_TRUE(yolo_postprocess::classAwareNms(boxes, scores, class_ids, 0.5f, 0.5f).empty());
}

TEST(YoloPostprocess, DecodeEndToEndIgnoresSegmentationFields) {
    constexpr int detections = 300;
    constexpr int fields = 38; // x1, y1, x2, y2, score, class id, mask coefficients
    std::vector<float> output(detections * fields, 0.0f);

    constexpr int det = 17;
    output[det * fields + 0] = 20.0f;
    output[det * fields + 1] = 20.0f;
    output[det * fields + 2] = 70.0f;
    output[det * fields + 3] = 80.0f;
    output[det * fields + 4] = 0.95f;
    output[det * fields + 5] = 1.0f;
    output[det * fields + 6] = 0.25f;

    cv::Mat frame(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
    const cv::Mat before = frame.clone();
    const std::vector<std::string> classes = {"person", "truck"};

    StreamCapture capture(std::cout);
    const bool decoded = yolo_postprocess::decodeEndToEnd(output.data(), detections, fields, frame,
                                                          cv::Size2f(100, 100), classes, 0.5f, 0.5f);
    const std::string stdout_text = capture.str();

    EXPECT_TRUE(decoded);
    EXPECT_GT(changedPixelCount(before, frame), 0);
    expectYellowPixelNear(frame, 20, 20);
    EXPECT_NE(stdout_text.find("truck"), std::string::npos);
}

TEST(YoloPostprocess, DecodeEndToEndAppliesNmsToOverlappingBoxes) {
    constexpr int detections = 2;
    constexpr int fields = 6; // x1, y1, x2, y2, score, class id
    std::vector<float> output(detections * fields, 0.0f);

    output[0 * fields + 0] = 20.0f;
    output[0 * fields + 1] = 20.0f;
    output[0 * fields + 2] = 70.0f;
    output[0 * fields + 3] = 80.0f;
    output[0 * fields + 4] = 0.95f;
    output[0 * fields + 5] = 1.0f;

    output[1 * fields + 0] = 21.0f;
    output[1 * fields + 1] = 21.0f;
    output[1 * fields + 2] = 71.0f;
    output[1 * fields + 3] = 81.0f;
    output[1 * fields + 4] = 0.90f;
    output[1 * fields + 5] = 1.0f;

    cv::Mat frame(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
    const std::vector<std::string> classes = {"person", "truck"};

    StreamCapture capture(std::cout);
    const bool decoded = yolo_postprocess::decodeEndToEnd(output.data(), detections, fields, frame,
                                                          cv::Size2f(100, 100), classes, 0.5f, 0.5f);
    const std::string stdout_text = capture.str();

    EXPECT_TRUE(decoded);
    EXPECT_EQ(countOccurrences(stdout_text, "truck"), 1u);
}

TEST(YoloPostprocess, DecodeRawRejectsShapeThatDoesNotFitClassList) {
    constexpr int fields = 5;
    constexpr int detections = 8;
    std::vector<float> output(fields * detections, 0.9f);

    cv::Mat frame(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
    const cv::Mat before = frame.clone();
    const std::vector<std::string> classes = {"person", "helmet", "forklift"};

    const bool decoded = yolo_postprocess::decodeRaw(output.data(), fields, detections, frame,
                                                     cv::Size2f(100, 100), classes,
                                                     0.5f, 0.5f);

    EXPECT_FALSE(decoded);
    EXPECT_EQ(changedPixelCount(before, frame), 0);
}

// confidence(): probabilities pass through, raw logits get squashed via sigmoid.

TEST(Confidence, PassesProbabilitiesThrough) {
    EXPECT_FLOAT_EQ(yolo_postprocess::confidence(0.0f), 0.0f);
    EXPECT_FLOAT_EQ(yolo_postprocess::confidence(1.0f), 1.0f);
    EXPECT_FLOAT_EQ(yolo_postprocess::confidence(0.42f), 0.42f);
}

TEST(Confidence, AppliesSigmoidToOutOfRangeLogits) {
    EXPECT_NEAR(yolo_postprocess::confidence(4.0f), 1.0f / (1.0f + std::exp(-4.0f)), 1e-6f);
    EXPECT_NEAR(yolo_postprocess::confidence(-4.0f), 1.0f / (1.0f + std::exp(4.0f)), 1e-6f);
    EXPECT_GT(yolo_postprocess::confidence(10.0f), 0.99f);
    EXPECT_LT(yolo_postprocess::confidence(-10.0f), 0.01f);
}

// frame_crop::centeredSquare(): largest centered square ROI inside the frame.

TEST(FrameCrop, CentersWiderThanTallFrame) {
    EXPECT_EQ(frame_crop::centeredSquare(200, 100), cv::Rect(50, 0, 100, 100));
}

TEST(FrameCrop, CentersTallerThanWideFrame) {
    EXPECT_EQ(frame_crop::centeredSquare(100, 200), cv::Rect(0, 50, 100, 100));
}

TEST(FrameCrop, ReturnsFullFrameWhenAlreadySquare) {
    EXPECT_EQ(frame_crop::centeredSquare(128, 128), cv::Rect(0, 0, 128, 128));
}

TEST(FrameCrop, StaysWithinFrameForOddDimensions) {
    const cv::Rect r = frame_crop::centeredSquare(201, 100);
    EXPECT_EQ(r.width, 100);
    EXPECT_EQ(r.height, 100);
    EXPECT_GE(r.x, 0);
    EXPECT_LE(r.x + r.width, 201);
    EXPECT_EQ(r.y, 0);
}

// config_loader string -> enum helpers, including their fallback behaviour.

TEST(ConfigParsing, FrameworkFromStringKnownValues) {
    EXPECT_EQ(frameworkFromString("OpenCV"), FRAMEWORK::OpenCV);
    EXPECT_EQ(frameworkFromString("OpenVINO"), FRAMEWORK::OpenVINO);
    EXPECT_EQ(frameworkFromString("ONNXRuntime"), FRAMEWORK::ONNXRuntime);
    EXPECT_EQ(frameworkFromString("CoreAI"), FRAMEWORK::CoreAI);
}

TEST(ConfigParsing, FrameworkFromStringFallsBackToOpenCV) {
    StreamCapture capture(std::cerr);
    const FRAMEWORK framework = frameworkFromString("does-not-exist");
    EXPECT_EQ(framework, FRAMEWORK::OpenCV);
}

TEST(ConfigParsing, SourceFromStringKnownValues) {
    EXPECT_EQ(sourceFromString("VIDEO"), SOURCE::VIDEO);
    EXPECT_EQ(sourceFromString("WEBCAM"), SOURCE::WEBCAM);
}

TEST(ConfigParsing, SourceFromStringFallsBackToVideo) {
    StreamCapture capture(std::cerr);
    const SOURCE source = sourceFromString("does-not-exist");
    EXPECT_EQ(source, SOURCE::VIDEO);
}

// resolveAgainst: relative asset paths are resolved against the config directory.

TEST(ConfigPaths, ResolvesRelativePathAgainstBase) {
    EXPECT_EQ(resolveAgainst("..", "yoloe.onnx"), std::string("../yoloe.onnx"));
    EXPECT_EQ(resolveAgainst("/opt/app", "tests/clip.mov"), std::string("/opt/app/tests/clip.mov"));
}

TEST(ConfigPaths, LeavesAbsolutePathUnchanged) {
    EXPECT_EQ(resolveAgainst("/base", "/abs/model.onnx"), std::string("/abs/model.onnx"));
}

TEST(ConfigPaths, LeavesPathUnchangedWhenBaseIsEmpty) {
    EXPECT_EQ(resolveAgainst("", "model.onnx"), std::string("model.onnx"));
}

TEST(ConfigPaths, LeavesEmptyPathUnchanged) {
    EXPECT_EQ(resolveAgainst("/base", ""), std::string(""));
}

// supportedBackendsFor: format -> compatible backends (build-independent).

namespace {
bool backendAllowed(MODEL_FORMAT format, FRAMEWORK framework) {
    const std::vector<FRAMEWORK> backends = supportedBackendsFor(format);
    return std::find(backends.begin(), backends.end(), framework) != backends.end();
}
}

TEST(DetectorFactory, YoloEAllowsAllBackends) {
    EXPECT_TRUE(backendAllowed(MODEL_FORMAT::YOLOE, FRAMEWORK::OpenCV));
    EXPECT_TRUE(backendAllowed(MODEL_FORMAT::YOLOE, FRAMEWORK::ONNXRuntime));
    EXPECT_TRUE(backendAllowed(MODEL_FORMAT::YOLOE, FRAMEWORK::OpenVINO));
}

TEST(DetectorFactory, Yolo12AllowsOpenCV) {
    EXPECT_TRUE(backendAllowed(MODEL_FORMAT::YOLO12, FRAMEWORK::OpenCV));
}

TEST(DetectorFactory, Yolo26ExcludesOpenCVButKeepsOthers) {
    EXPECT_FALSE(backendAllowed(MODEL_FORMAT::YOLO26, FRAMEWORK::OpenCV));
    EXPECT_TRUE(backendAllowed(MODEL_FORMAT::YOLO26, FRAMEWORK::ONNXRuntime));
    EXPECT_TRUE(backendAllowed(MODEL_FORMAT::YOLO26, FRAMEWORK::OpenVINO));
}

TEST(DetectorFactory, UnknownFormatHasNoBackends) {
    EXPECT_TRUE(supportedBackendsFor(MODEL_FORMAT::Unknown).empty());
}

TEST(DetectorFactory, Yolo26SupportsCoreAIBackend) {
    EXPECT_TRUE(backendAllowed(MODEL_FORMAT::YOLO26, FRAMEWORK::CoreAI));
}

// Fallback mapping: a Core AI (.aimodel) selection must map to the equivalent
// ONNX asset when the factory falls back to a backend that consumes ONNX models
// (ONNX Runtime / OpenVINO / OpenCV DNN), instead of handing the .aimodel package
// to an ONNX-format backend.

TEST(DetectorFactory, AimodelFallbackMapsToOnnxCounterpart) {
    EXPECT_EQ(onnxFallbackModelPath("yolo26x.aimodel"), "yolo26x.onnx");
    EXPECT_EQ(onnxFallbackModelPath("models/yolo26m.aimodel"), "models/yolo26m.onnx");
}

TEST(DetectorFactory, OnnxModelFilesPassThroughUnchanged) {
    EXPECT_EQ(onnxFallbackModelPath("yolo26x.onnx"), "yolo26x.onnx");
    EXPECT_EQ(onnxFallbackModelPath("yoloe-26m-seg-pf.onnx"), "yoloe-26m-seg-pf.onnx");
}

// decodeByFormat: dispatches to the right decoder(s) for each model format.

namespace {
// End-to-end layout [num_dets=2, fields=6]: x1, y1, x2, y2, score, class id.
std::vector<float> makeEndToEndOutput() {
    constexpr int detections = 2;
    constexpr int fields = 6;
    std::vector<float> out(detections * fields, 0.0f);
    out[0 * fields + 0] = 20.0f;  // x1
    out[0 * fields + 1] = 20.0f;  // y1
    out[0 * fields + 2] = 70.0f;  // x2
    out[0 * fields + 3] = 80.0f;  // y2
    out[0 * fields + 4] = 0.95f;  // score
    out[0 * fields + 5] = 1.0f;   // class id
    return out;
}

// Raw layout, fields-first [fields=7, num_dets=8]: x, y, w, h + 3 class scores.
std::vector<float> makeRawOutput() {
    constexpr int fields = 7;
    constexpr int detections = 8;
    std::vector<float> out(fields * detections, 0.01f);
    auto set_value = [&](int field, int det, float value) {
        out[field * detections + det] = value;
    };
    set_value(0, 3, 50.0f);
    set_value(1, 3, 50.0f);
    set_value(2, 3, 20.0f);
    set_value(3, 3, 20.0f);
    set_value(4, 3, 0.05f);
    set_value(5, 3, 0.92f);  // class 1 wins
    set_value(6, 3, 0.10f);
    return out;
}
}  // namespace

TEST(DecodeByFormat, Yolo12UsesRawDecoder) {
    const std::vector<float> out = makeRawOutput();
    cv::Mat frame(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
    const cv::Mat before = frame.clone();
    const std::vector<std::string> classes = {"person", "helmet", "forklift"};

    const bool decoded = yolo_postprocess::decodeByFormat(
        MODEL_FORMAT::YOLO12, out.data(), 7, 8, frame,
        cv::Size2f(100, 100), classes, 0.5f, 0.5f);

    EXPECT_TRUE(decoded);
    EXPECT_GT(changedPixelCount(before, frame), 0);
}

TEST(DecodeByFormat, Yolo26UsesEndToEndDecoder) {
    const std::vector<float> out = makeEndToEndOutput();
    cv::Mat frame(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
    const cv::Mat before = frame.clone();
    const std::vector<std::string> classes = {"person", "truck"};

    const bool decoded = yolo_postprocess::decodeByFormat(
        MODEL_FORMAT::YOLO26, out.data(), 2, 6, frame,
        cv::Size2f(100, 100), classes, 0.5f, 0.5f);

    EXPECT_TRUE(decoded);
    EXPECT_GT(changedPixelCount(before, frame), 0);
}

TEST(DecodeByFormat, Yolo26UsesRawDecoderForRawShapedOutput) {
    // A raw-head YOLO26 export has the shape [1, 84, 8400] (batch, 4 + numClasses
    // fields, anchor rows). It must go through the raw decoder, not the end-to-end
    // one, and still produce a detection.
    constexpr int fields = 7;  // x, y, w, h + 3 class scores
    constexpr int detections = 2000;
    std::vector<float> out(fields * detections, 0.01f);
    auto set_value = [&](int field, int det, float value) {
        out[field * detections + det] = value;
    };
    set_value(0, 3, 50.0f);
    set_value(1, 3, 50.0f);
    set_value(2, 3, 20.0f);
    set_value(3, 3, 20.0f);
    set_value(4, 3, 0.05f);
    set_value(5, 3, 0.92f);  // class 1 wins
    set_value(6, 3, 0.10f);

    cv::Mat frame(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
    const cv::Mat before = frame.clone();
    const std::vector<std::string> classes = {"person", "helmet", "forklift"};

    const bool decoded = yolo_postprocess::decodeByFormat(
        MODEL_FORMAT::YOLO26, out.data(), fields, detections, frame,
        cv::Size2f(100, 100), classes, 0.5f, 0.5f);

    EXPECT_TRUE(decoded);
    EXPECT_GT(changedPixelCount(before, frame), 0);
    expectYellowPixelNear(frame, 40, 40);
}

TEST(DecodeByFormat, Yolo12RejectsEndToEndLayoutWithNoFallback) {
    // A [2, 6] end-to-end output has too few fields for a raw decode, and YOLO12
    // has no end-to-end fallback, so nothing is decoded or drawn.
    const std::vector<float> out = makeEndToEndOutput();
    cv::Mat frame(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
    const cv::Mat before = frame.clone();
    const std::vector<std::string> classes = {"person", "truck"};

    const bool decoded = yolo_postprocess::decodeByFormat(
        MODEL_FORMAT::YOLO12, out.data(), 2, 6, frame,
        cv::Size2f(100, 100), classes, 0.5f, 0.5f);

    EXPECT_FALSE(decoded);
    EXPECT_EQ(changedPixelCount(before, frame), 0);
}

TEST(DecodeByFormat, YoloEFallsBackToEndToEnd) {
    // YOLOE tries raw first; for an end-to-end layout raw fails and the
    // end-to-end fallback must handle it.
    const std::vector<float> out = makeEndToEndOutput();
    cv::Mat frame(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
    const cv::Mat before = frame.clone();
    const std::vector<std::string> classes = {"person", "truck"};

    const bool decoded = yolo_postprocess::decodeByFormat(
        MODEL_FORMAT::YOLOE, out.data(), 2, 6, frame,
        cv::Size2f(100, 100), classes, 0.5f, 0.5f);

    EXPECT_TRUE(decoded);
    EXPECT_GT(changedPixelCount(before, frame), 0);
}

TEST(DecodeByFormat, YoloEPrefersEndToEndForEndToEndShapedOutput) {
    // Regression: an end-to-end [300, 6] output together with a 2-entry class
    // list also satisfies decodeRaw's field check (6 >= 4 + 2), so a raw-first
    // dispatch misdecodes x1,y1,x2,y2 as cx,cy,w,h. End-to-end-shaped outputs
    // (few detection rows, 6..128 fields) must go to the end-to-end decoder.
    constexpr int detections = 300;
    constexpr int fields = 6;
    std::vector<float> out(detections * fields, 0.0f);
    out[0] = 20.0f;  // x1
    out[1] = 20.0f;  // y1
    out[2] = 70.0f;  // x2
    out[3] = 80.0f;  // y2
    out[4] = 0.95f;  // score
    out[5] = 1.0f;   // class id

    cv::Mat frame(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
    const std::vector<std::string> classes = {"person", "truck"};

    const bool decoded = yolo_postprocess::decodeByFormat(
        MODEL_FORMAT::YOLOE, out.data(), detections, fields, frame,
        cv::Size2f(100, 100), classes, 0.5f, 0.5f);

    EXPECT_TRUE(decoded);
    // Only a correct end-to-end decode draws the box corner at (20, 20).
    expectYellowPixelNear(frame, 20, 20);
    expectYellowPixelNear(frame, 70, 80);
}

TEST(DecodeByFormat, YoloEUsesRawDecoderForRawShapedOutput) {
    // A realistic raw export has thousands of anchor rows, far above any
    // end-to-end detection count, and must keep going to the raw decoder.
    constexpr int fields = 7;  // x, y, w, h + 3 class scores
    constexpr int detections = 2000;
    std::vector<float> out(fields * detections, 0.01f);
    auto set_value = [&](int field, int det, float value) {
        out[field * detections + det] = value;
    };
    set_value(0, 3, 50.0f);
    set_value(1, 3, 50.0f);
    set_value(2, 3, 20.0f);
    set_value(3, 3, 20.0f);
    set_value(4, 3, 0.05f);
    set_value(5, 3, 0.92f);  // class 1 wins
    set_value(6, 3, 0.10f);

    cv::Mat frame(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
    const std::vector<std::string> classes = {"person", "helmet", "forklift"};

    const bool decoded = yolo_postprocess::decodeByFormat(
        MODEL_FORMAT::YOLOE, out.data(), fields, detections, frame,
        cv::Size2f(100, 100), classes, 0.5f, 0.5f);

    EXPECT_TRUE(decoded);
    expectYellowPixelNear(frame, 40, 40);
}

TEST(DecodeByFormat, UnknownFormatDecodesNothing) {
    const std::vector<float> out = makeEndToEndOutput();
    cv::Mat frame(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
    const cv::Mat before = frame.clone();
    const std::vector<std::string> classes = {"person", "truck"};

    const bool decoded = yolo_postprocess::decodeByFormat(
        MODEL_FORMAT::Unknown, out.data(), 2, 6, frame,
        cv::Size2f(100, 100), classes, 0.5f, 0.5f);

    EXPECT_FALSE(decoded);
    EXPECT_EQ(changedPixelCount(before, frame), 0);
}

// warnUnsupportedShape: warn once per distinct (backend, shape), not once ever.
// Uses backend names unique to this test so it is independent of test order.

TEST(WarnUnsupportedShape, WarnsOncePerDistinctMessage) {
    // Repeated identical (backend, shape) warns only the first time.
    std::string first;
    {
        StreamCapture capture(std::cerr);
        yolo_postprocess::warnUnsupportedShape("WarnTestBackendA", {1, 2, 3});
        first = capture.str();
    }
    EXPECT_NE(first.find("WarnTestBackendA"), std::string::npos);

    StreamCapture capture(std::cerr);
    yolo_postprocess::warnUnsupportedShape("WarnTestBackendA", {1, 2, 3});
    EXPECT_TRUE(capture.str().empty());
}

// VideoHandler: the writer is opened lazily with the size of the frames that
// are actually written, not the cached capture properties (which can disagree
// with the decoded frame size for some codecs).

namespace {
std::string tempOutputPath(const std::string& name) {
    return (std::filesystem::temp_directory_path() / name).string();
}
}  // namespace

TEST(VideoHandlerWriter, OpensWriterWithActualFrameSize) {
    const std::string output_path = tempOutputPath("objdet_writer_size.avi");
    std::filesystem::remove(output_path);

    {
        VideoHandler handler(output_path);
        handler.fps = 30.0f;
        cv::Mat frame(64, 64, CV_8UC3, cv::Scalar(0, 128, 255));
        for (int i = 0; i < 5; ++i) {
            ASSERT_TRUE(handler.write(frame));
        }
    }  // destructor releases the writer

    cv::VideoCapture reader(output_path);
    ASSERT_TRUE(reader.isOpened()) << "Could not read back " << output_path;
    cv::Mat read_back;
    ASSERT_TRUE(reader.read(read_back));
    EXPECT_EQ(read_back.cols, 64);
    EXPECT_EQ(read_back.rows, 64);
    std::filesystem::remove(output_path);
}

// sanitizedFps: sources that do not report a usable CAP_PROP_FPS (0, negative,
// or a nonsense 1.0) get a conservative 5 fps fallback so the writer can open.

TEST(VideoHandlerFps, PassesReportedRateThrough) {
    EXPECT_FLOAT_EQ(VideoHandler::sanitizedFps(29.97), 29.97f);
    EXPECT_FLOAT_EQ(VideoHandler::sanitizedFps(240.0), 240.0f);
}

TEST(VideoHandlerFps, FallsBackForUnusableRates) {
    EXPECT_FLOAT_EQ(VideoHandler::sanitizedFps(0.0), 5.0f);
    EXPECT_FLOAT_EQ(VideoHandler::sanitizedFps(-1.0), 5.0f);
    EXPECT_FLOAT_EQ(VideoHandler::sanitizedFps(1.0), 5.0f);
}

TEST(VideoHandlerWriter, WriteFailsWhenWriterCannotBeOpened) {
    VideoHandler handler("/nonexistent_dir_objdet/out.avi");
    handler.fps = 30.0f;
    cv::Mat frame(64, 64, CV_8UC3, cv::Scalar(0, 0, 0));

    StreamCapture capture(std::cerr);
    EXPECT_FALSE(handler.write(frame));
}

// clippedRect: boxes are clamped to the frame but keep their full extent when
// they touch or overhang the border (cv::Rect's br() is exclusive).

TEST(ClippedRect, KeepsFullExtentAtFrameBorder) {
    EXPECT_EQ(yolo_postprocess::clippedRect(0.0f, 0.0f, 100.0f, 100.0f, cv::Size(100, 100)),
              cv::Rect(0, 0, 100, 100));
}

TEST(ClippedRect, ClampsBoxOverhangingTheBorder) {
    EXPECT_EQ(yolo_postprocess::clippedRect(-10.0f, -10.0f, 110.0f, 110.0f, cv::Size(100, 100)),
              cv::Rect(0, 0, 100, 100));
}

TEST(ClippedRect, ReturnsEmptyForBoxOutsideFrame) {
    EXPECT_TRUE(yolo_postprocess::clippedRect(120.0f, 10.0f, 150.0f, 30.0f, cv::Size(100, 100)).empty());
}

// sanitizeConfig: values that would corrupt downstream math (zero model size,
// thresholds outside [0, 1]) are reset to the built-in defaults.

TEST(ConfigSanitize, ResetsNonPositiveModelShapeToDefault) {
    Config cfg;
    cfg.model_shape = cv::Size2f(0.0f, -640.0f);

    StreamCapture capture(std::cerr);
    sanitizeConfig(cfg);

    const Config defaults;
    EXPECT_EQ(cfg.model_shape, defaults.model_shape);
    EXPECT_FALSE(capture.str().empty());
}

TEST(ConfigSanitize, ResetsOutOfRangeThresholdsToDefaults) {
    Config cfg;
    cfg.score_threshold = -0.1f;
    cfg.nms_threshold = 1.5f;

    StreamCapture capture(std::cerr);
    sanitizeConfig(cfg);

    const Config defaults;
    EXPECT_FLOAT_EQ(cfg.score_threshold, defaults.score_threshold);
    EXPECT_FLOAT_EQ(cfg.nms_threshold, defaults.nms_threshold);
}

TEST(ConfigSanitize, LeavesValidConfigUntouched) {
    Config cfg;
    cfg.score_threshold = 0.7f;
    cfg.nms_threshold = 0.3f;
    cfg.model_shape = cv::Size2f(320.0f, 320.0f);

    StreamCapture capture(std::cerr);
    sanitizeConfig(cfg);

    EXPECT_FLOAT_EQ(cfg.score_threshold, 0.7f);
    EXPECT_FLOAT_EQ(cfg.nms_threshold, 0.3f);
    EXPECT_EQ(cfg.model_shape, cv::Size2f(320.0f, 320.0f));
    EXPECT_TRUE(capture.str().empty());
}

#ifdef USE_YAML_CONFIG
// loadConfigYAML: video_output_file lives in the output: section; the legacy
// input: location keeps working so existing configs do not silently regress.

namespace {
std::string writeTempConfig(const std::string& name, const std::string& contents) {
    const auto path = std::filesystem::temp_directory_path() / name;
    std::ofstream out(path);
    out << contents;
    return path.string();
}

std::string loadedOutputFileName(const std::string& config_path) {
    const Config cfg = loadConfigYAML(config_path);
    std::filesystem::remove(config_path);
    // Relative paths are resolved against the config directory; compare only
    // the file name.
    return std::filesystem::path(cfg.video_output_file).filename().string();
}
}  // namespace

TEST(ConfigYaml, ReadsVideoOutputFileFromOutputSection) {
    const std::string path = writeTempConfig(
        "objdet_test_output_section.yaml",
        "output:\n  video_output_file: \"from_output.mp4\"\n");
    EXPECT_EQ(loadedOutputFileName(path), "from_output.mp4");
}

TEST(ConfigYaml, StillReadsVideoOutputFileFromLegacyInputSection) {
    const std::string path = writeTempConfig(
        "objdet_test_legacy_input.yaml",
        "input:\n  video_output_file: \"legacy.mp4\"\n");
    EXPECT_EQ(loadedOutputFileName(path), "legacy.mp4");
}

TEST(ConfigYaml, PrefersOutputSectionOverLegacyInputKey) {
    const std::string path = writeTempConfig(
        "objdet_test_output_wins.yaml",
        "input:\n  video_output_file: \"legacy.mp4\"\n"
        "output:\n  video_output_file: \"from_output.mp4\"\n");
    EXPECT_EQ(loadedOutputFileName(path), "from_output.mp4");
}
#endif  // USE_YAML_CONFIG

TEST(WarnUnsupportedShape, WarnsAgainForDifferentBackendOrShape) {
    {
        StreamCapture capture(std::cerr);
        yolo_postprocess::warnUnsupportedShape("WarnTestBackendB", {1, 2, 3});
        EXPECT_FALSE(capture.str().empty());
    }

    // Same backend, different shape -> warns again.
    {
        StreamCapture capture(std::cerr);
        yolo_postprocess::warnUnsupportedShape("WarnTestBackendB", {1, 4, 5});
        EXPECT_FALSE(capture.str().empty());
    }

    // Different backend, same shape -> warns again.
    {
        StreamCapture capture(std::cerr);
        yolo_postprocess::warnUnsupportedShape("WarnTestBackendC", {1, 2, 3});
        EXPECT_FALSE(capture.str().empty());
    }
}
