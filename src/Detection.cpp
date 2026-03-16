#include "Detection.hpp"

Detection::Detection(
            float score_threshold,
            cv::Size2f model_shape,
            const std::string &model_file ) :

            score_threshold_(score_threshold),
            model_shape_(model_shape),
            model_file_(model_file) {

            model_format_ = detectModelFormat(model_file);
}

void Detection::load_class_list(const std::string &class_file) {
    // change this txt file  to your txt file that contains labels
    std::ifstream ifs(class_file);
    std::string line;
    while ( getline(ifs, line) ) {
        class_list.push_back(line);
    }
}