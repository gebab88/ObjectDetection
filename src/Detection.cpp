#include "Detection.hpp"

Detection::Detection(
            float score_threshold,
            Size2f model_shape,
            const string &model_file ) :

            score_threshold_(score_threshold),
            model_shape_(model_shape),
            model_file_(model_file) {
}

void Detection::load_class_list(const string &class_file) {
    // change this txt file  to your txt file that contains labels
    ifstream ifs(class_file);
    string line;
    while ( getline(ifs, line) ) {
        class_list.push_back(line);
    }
}

void Detection::change_modell_file() {
    // Change Modell-File string --> shorten it
    size_t first_non_dot = this->model_file_.find_first_not_of('.');
    if (first_non_dot != std::string::npos) {
        // 2. Erase everything from the start up to that index
        this->model_file_.erase(0, first_non_dot +1 );
    } else {
        // 3. If the string was ONLY dots, clear it entirely
        this->model_file_.clear();
    }
}




