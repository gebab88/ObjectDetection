#include "Detection_OpenVINO.hpp"


Detection_OpenVINO::Detection_OpenVINO( float score_threshold,
                                        Size2f model_shape,
                                        const string &model_file ) :
                                        Detection(score_threshold, model_shape, model_file){
    auto model = core_.read_model(model_file_);
    // auto compiled = core_.compile_model(model, "AUTO");
    auto compiled = core_.compile_model(model, "CPU", ov::hint::enable_hyper_threading(true));
    infer_request_       = compiled.create_infer_request();
    auto input_port = compiled.input();
    input_ = ov::Tensor(  input_port.get_element_type(),
                                input_port.get_shape()
                                );
    plane_ = 640 * 640;
    chans_ = std::vector<Mat>(3);
};

void Detection_OpenVINO::detect( cv::Mat &frame) {
    // Skalierungsfaktoren berechnen, um Boxen auf Originalbildgröße zu ziehen
    const float x_scale = (float)frame.cols / model_shape_.width;
    const float y_scale = (float)frame.rows / model_shape_.height;

    cv::resize(frame, resized_, {640, 640});

    // BGR → RGB, uint8 → float32, normalisieren [0,1]
    resized_.convertTo(blob_, CV_32F, 1.0 / 255.0);
    cvtColor(blob_, blob_, cv::COLOR_BGR2RGB);

    // HWC → CHW in Eingabetensor kopieren
    split(blob_, chans_);

    data_  = input_.data<float>();
    for (int c = 0; c < 3; ++c)
        std::memcpy(data_ + c * plane_, chans[c].ptr<float>(), plane_ * sizeof(float));

    infer_request_.set_input_tensor(input_);

    // ── Inferenz ──────────────────────────────────────────────────────
    infer_request_.infer();

    output_ = infer_request_.get_output_tensor(0);
    if (output_.get_size() == 0) { std::cerr << "Leerer Output!\n"; }

    // Dimensionen abrufen: z.B. 300 Erkennungen
    ov::Shape shape = output_.get_shape();
    const int numDets = shape[1];
    const int numFields = shape[2];

    float x1,x2,y1,y2,score;
    int class_id;
    cv::Rect box;

    data_ = output_.data<float>();
    const float* end = data_ + numDets * numFields;

    for ( ; data_ != end; data_ += numFields) {
        score = data_[4];

        // Frühzeitiger Abbruch – häufigster Pfad zuerst
        if (score < score_threshold_) continue;

        // Koordinaten extrahieren und auf Originalgröße skalieren
        x1 = data_[0] * x_scale;
        y1 = data_[1] * y_scale;
        x2 = data_[2] * x_scale;
        y2 = data_[3] * y_scale;
        class_id = (int)data_[5];
        if (class_id < 0 || class_id >= (int)class_list.size()) continue;

        // Ungültige Boxen (Breite/Höhe <= 0) direkt verwerfen
        if (x2 <= x1 || y2 <= y1) continue;

        box = cv::Rect(int(x1), int(y1), int(x2-x1), int(y2-y1));

        rectangle(frame, box, Scalar(0,255,255), 1);
        tsd::string label = class_list[ class_id ] + "  " + to_string(int(100 * score)) + "%" ;
        std::cout << label << std::endl;
        putText(frame, label, Point( box.x, box.y ), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
    }
}