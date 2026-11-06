#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "Yolo26Decoder.hpp"

namespace {

// Quantization used by the fixtures: value = (q - 1000) * 0.01.
constexpr yolo26::Quant kQuant{1000.0f, 0.01f};

uint16_t quantize(float value, yolo26::Quant quant = kQuant) {
    return static_cast<uint16_t>(std::lround(value / quant.scale + quant.zero_point));
}

float logit(float probability) {
    return std::log(probability / (1.0f - probability));
}

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// One head level in the HEF's NHWC layout, initialised to "nothing here":
// zero box distances and very negative class logits.
struct LevelData {
    int height, width, classes;
    float stride;
    yolo26::Quant box_quant = kQuant;
    yolo26::Quant score_quant = kQuant;
    std::vector<uint16_t> boxes;
    std::vector<uint16_t> scores;

    LevelData(int h, int w, int c, float s)
        : height(h), width(w), classes(c), stride(s),
          boxes(static_cast<size_t>(h) * w * 4, quantize(0.0f)),
          scores(static_cast<size_t>(h) * w * c, quantize(-9.0f)) {}

    void setBox(int y, int x, float left, float top, float right, float bottom) {
        uint16_t* cell = &boxes[(static_cast<size_t>(y) * width + x) * 4];
        cell[0] = quantize(left, box_quant);
        cell[1] = quantize(top, box_quant);
        cell[2] = quantize(right, box_quant);
        cell[3] = quantize(bottom, box_quant);
    }

    void setLogit(int y, int x, int class_id, float value) {
        scores[(static_cast<size_t>(y) * width + x) * classes + class_id] = quantize(value, score_quant);
    }

    yolo26::HeadLevel view() const {
        return {boxes.data(), box_quant, scores.data(), score_quant, height, width, stride};
    }
};

}  // namespace

TEST(Yolo26Decoder, DecodesBoxOfCellAboveThreshold) {
    LevelData level(2, 2, 3, 8.0f);
    level.setBox(0, 1, 0.5f, 0.25f, 1.0f, 2.0f);
    level.setLogit(0, 1, 2, 2.0f);

    const auto results = yolo26::decode({level.view()}, 3, 0.5f);

    ASSERT_EQ(results.size(), 1u);
    // Cell (x=1, y=0) has its anchor at (1.5, 0.5) grid units.
    EXPECT_FLOAT_EQ(results[0].x1, (1.5f - 0.5f) * 8.0f);
    EXPECT_FLOAT_EQ(results[0].y1, (0.5f - 0.25f) * 8.0f);
    EXPECT_FLOAT_EQ(results[0].x2, (1.5f + 1.0f) * 8.0f);
    EXPECT_FLOAT_EQ(results[0].y2, (0.5f + 2.0f) * 8.0f);
    EXPECT_EQ(results[0].class_id, 2);
    EXPECT_NEAR(results[0].score, sigmoid(2.0f), 1e-6f);
}

TEST(Yolo26Decoder, IgnoresScoresBelowThreshold) {
    LevelData level(2, 2, 3, 8.0f);
    level.setLogit(1, 1, 0, logit(0.5f) - 0.02f);

    EXPECT_TRUE(yolo26::decode({level.view()}, 3, 0.5f).empty());
}

TEST(Yolo26Decoder, KeepsScoreExactlyAtThreshold) {
    LevelData level(1, 1, 1, 8.0f);
    level.setLogit(0, 0, 0, 0.0f);  // sigmoid(0) = 0.5 exactly

    EXPECT_EQ(yolo26::decode({level.view()}, 1, 0.5f).size(), 1u);
}

TEST(Yolo26Decoder, ReportsEveryClassAboveThresholdOfOneCell) {
    LevelData level(1, 1, 4, 8.0f);
    level.setLogit(0, 0, 1, 1.0f);
    level.setLogit(0, 0, 3, 2.0f);

    const auto results = yolo26::decode({level.view()}, 4, 0.5f);

    ASSERT_EQ(results.size(), 2u);
    EXPECT_EQ(results[0].class_id, 3);
    EXPECT_EQ(results[1].class_id, 1);
}

TEST(Yolo26Decoder, SortsByScoreAndKeepsAtMostMaxDetections) {
    LevelData level(1, 4, 1, 8.0f);
    level.setLogit(0, 0, 0, 1.0f);
    level.setLogit(0, 1, 0, 3.0f);
    level.setLogit(0, 2, 0, 2.0f);
    level.setLogit(0, 3, 0, 0.5f);

    const auto results = yolo26::decode({level.view()}, 1, 0.5f, 2);

    ASSERT_EQ(results.size(), 2u);
    EXPECT_NEAR(results[0].score, sigmoid(3.0f), 1e-6f);
    EXPECT_NEAR(results[1].score, sigmoid(2.0f), 1e-6f);
}

TEST(Yolo26Decoder, UsesStrideAndQuantizationOfEachLevel) {
    LevelData fine(2, 2, 1, 8.0f);
    LevelData coarse(1, 1, 1, 16.0f);
    coarse.box_quant = {10.0f, 0.5f};
    coarse.score_quant = {0.0f, 0.25f};
    coarse.setBox(0, 0, 1.0f, 1.0f, 1.0f, 1.0f);
    coarse.setLogit(0, 0, 0, 1.0f);

    const auto results = yolo26::decode({fine.view(), coarse.view()}, 1, 0.5f);

    ASSERT_EQ(results.size(), 1u);
    // Anchor (0.5, 0.5) grid units, distances of 1 cell, 16 px per cell.
    EXPECT_FLOAT_EQ(results[0].x1, -8.0f);
    EXPECT_FLOAT_EQ(results[0].y1, -8.0f);
    EXPECT_FLOAT_EQ(results[0].x2, 24.0f);
    EXPECT_FLOAT_EQ(results[0].y2, 24.0f);
    EXPECT_NEAR(results[0].score, sigmoid(1.0f), 1e-6f);
}
