#include "Detection_OpenCV.hpp"

Detection_OpenCV::Detection_OpenCV(const float score_threshold,
                                   const cv::Size2f model_shape,
                                   const std::string &model_file,
                                   const float nms,
                                   const bool use_cuda) :
                                    Detection(score_threshold, model_shape, model_file),
                                    nms_threshold_(nms),
                                    use_cuda_(use_cuda) {
        net = cv::dnn::readNet(model_file);

        if (use_cuda_) {
            std::cout << "Running on CUDA" << std::endl << std::endl;
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        } else {
            std::cout << "Running on CPU" << std::endl << std::endl;
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        }
    }

    void Detection_OpenCV::detect( cv::Mat &frame) {
        const float x_scale = (float)frame.cols / model_shape_.width;
        const float y_scale = (float)frame.rows / model_shape_.height;

        // Convert the image into a blob and set it as input to the network
        //cv::resize(frame, resized_, {640, 640});
        cv::dnn::blobFromImage(frame, blob_, 1.0/255.0, model_shape_, cv::Scalar(), true, false);
        net.setInput(blob_);
        net.forward(outs, net.getUnconnectedOutLayersNames());

    if (model_file_ == "yolo12m.onnx") {
        std::vector<float> max_scores;
        std::vector<int> classIds;
        std::vector<cv::Rect> boxes;

        const int dimensions = outs[0].size[1];
        const int rows = outs[0].size[2];

        outs[0] = outs[0].reshape(1, dimensions);
        transpose(outs[0], outs[0]);

        data_ = (float *)outs[0].data;

        // Loop through all the rows to process predictions
        for (int i = 0; i < rows; ++i, data_ += dimensions) {
            // Get class scores and find the class with the highest score
            cv::Mat scores(1, class_list.size(), CV_32FC1, data_ + 4);
            cv::Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

            // If the class score is above the threshold, store the detection
            if (max_class_score > score_threshold_) {
                max_scores.push_back(max_class_score);
                classIds.push_back(class_id.x);

                // Calculate the bounding box coordinates
                const float x = data_[0] * x_scale;
                const float y = data_[1] * y_scale;
                const float w = data_[2] * x_scale;
                const float h = data_[3] * y_scale;
                const int left = int(x - 0.5 * w);
                const int top = int(y - 0.5 * h);
                const int width = int(w);
                const int height = int(h);
                boxes.push_back( cv::Rect(left, top, width, height) );
            }
        }

        // Apply Non-Maximum Suppression
        cv::dnn::NMSBoxes(boxes, max_scores, score_threshold_, nms_threshold_, nms_result);

        for ( auto idx : nms_result ) {
            cv::rectangle(frame, boxes[idx], cv::Scalar(0,255,255), 1);
            std::string label = class_list[ classIds[idx] ] + "  " + std::to_string(int(100 * max_scores[ idx ])) + "%" ;
            std::cout << label << '\n';
            putText(frame, label, cv::Point( boxes[idx].x, boxes[idx].y ), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
        }
    } else if (model_file_=="yolo26m.onnx"){
        // YOLO26 Output Format (End-to-End): [Batch, Anzahl_Boxen, 6]
        // Die 6 Werte sind: [x1, y1, x2, y2, Score, Klasse]

        // Pointer auf die Daten holen für schnellen Zugriff
        data_ = outs[0].ptr<float>();

        // Dimensionen abrufen: z.B. 300 Erkennungen
        const int numDets = outs[0].size[1];
        const int entries = outs[0].size[2];

        for (int i = 0; i < numDets; ++i,  data_ += entries) {
            // Frühzeitiger Abbruch – häufigster Pfad zuerst
            const float  score = data_[4];
            if (score < score_threshold_) continue;

            // Calculate the bounding box coordinates
            const float x1 = data_[0] * x_scale;
            const float y1 = data_[1] * y_scale;
            const float x2 = data_[2] * x_scale;
            const float y2 = data_[3] * y_scale;
            int class_id = (int)data_[5];
            if (class_id < 0 || class_id >= (int)class_list.size()) continue;

            cv::Rect box(int(x1), int(y1), int(x2-x1), int(y2-y1));

            cv::rectangle(frame, box, cv::Scalar(0,255,255), 1);
            std::string label = class_list[ class_id ] + "  " + std::to_string(int(100 * score)) + "%" ;
            std::cout << label << std::endl;
            putText(frame, label, cv::Point( box.x, box.y ), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
        }
    } else {
        // in progress
    }
}