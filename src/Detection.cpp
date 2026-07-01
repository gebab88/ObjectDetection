#include "Detection.hpp"

#include <iostream>

Detection::Detection(
            float score_threshold,
            cv::Size2f model_shape,
            const std::string &model_file ) :

            score_threshold_(score_threshold),
            model_shape_(model_shape),
            model_file_(model_file) {

            model_format_ = detectModelFormat(model_file);
}

bool Detection::load_class_list(const std::string &class_file) {
    class_list.clear();

    std::ifstream ifs(class_file);
    if (!ifs.is_open()) {
        std::cerr << "Could not open class names file: " << class_file << std::endl;
        return false;
    }

    std::string line;
    while ( getline(ifs, line) ) {
        if (!line.empty()) {
            class_list.push_back(line);
        }
    }

    if (class_list.empty()) {
        std::cerr << "Class names file is empty: " << class_file << std::endl;
        return false;
    }
    return true;
}
