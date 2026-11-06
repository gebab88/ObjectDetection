#include "Detection.hpp"
#include "ClassNames.hpp"

#include <iostream>

Detection::Detection(
            float score_threshold,
            cv::Size2f model_shape,
            const std::string &model_file ) :

            // Initialised in declaration order (see Detection.hpp).
            model_file_(model_file),
            model_format_(detectModelFormat(model_file)),
            score_threshold_(score_threshold),
            model_shape_(model_shape) {
}

bool Detection::load_class_list(const std::string &class_file) {
    class_list_.clear();

    std::ifstream ifs(class_file);
    if (!ifs.is_open()) {
        std::cerr << "Could not open class names file: " << class_file << std::endl;
        return false;
    }

    class_list_ = parseClassNames(ifs);

    if (class_list_.empty()) {
        std::cerr << "Class names file is empty: " << class_file << std::endl;
        return false;
    }
    return true;
}
