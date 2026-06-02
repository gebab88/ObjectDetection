#include <gtest/gtest.h>
#include "Framework.hpp"

// Tests for detectModelFormat helper

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

TEST(DetectModelFormat, ReturnsUnknownForOtherFiles) {
    EXPECT_EQ(detectModelFormat("some_model.onnx"), MODEL_FORMAT::Unknown);
}
