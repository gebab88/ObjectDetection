#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "HailoMessages.hpp"

namespace {

// gmock (HasSubstr) is not linked into tests_runner.
::testing::AssertionResult Contains(const std::string& text, const std::string& part) {
    if (text.find(part) != std::string::npos) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << "\"" << text << "\" does not contain \"" << part << "\"";
}

}  // namespace

TEST(HailoMessages, IncompatibleHefNamesBothArchitectures) {
    const std::string message = hailo_messages::incompatibleHef(
        "/models/yolov8n.hef", {"HAILO8"}, "HAILO10H");

    EXPECT_TRUE(Contains(message, "yolov8n.hef is compiled for HAILO8"));
    EXPECT_TRUE(Contains(message, "the connected device is a HAILO10H"));
    EXPECT_FALSE(Contains(message, "/models/"));
}

TEST(HailoMessages, IncompatibleHefPointsToModelZooFolderOfTheDevice) {
    const std::string message = hailo_messages::incompatibleHef(
        "yolov8n.hef", {"HAILO8"}, "HAILO10H");

    EXPECT_TRUE(Contains(message, "compiled for HAILO10H"));
    EXPECT_TRUE(Contains(message, "ModelZoo/Compiled/<version>/hailo10h/"));
}

TEST(HailoMessages, IncompatibleHefListsEveryArchitectureOfTheHef) {
    const std::string message = hailo_messages::incompatibleHef(
        "yolo26m.hef", {"HAILO15H", "HAILO10H"}, "HAILO8");

    EXPECT_TRUE(Contains(message, "compiled for HAILO15H, HAILO10H"));
}

TEST(HailoMessages, IncompatibleHefWithUnknownArchitectures) {
    const std::string message = hailo_messages::incompatibleHef("x.hef", {}, "");

    EXPECT_TRUE(Contains(message, "x.hef is not compiled for the connected Hailo device"));
}
