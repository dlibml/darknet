#ifndef yolo_h_INCLUDED
#define yolo_h_INCLUDED

#include "darknet.h"
#include "yolo_utils.h"

template <typename net_type> class yolo_detector
{
    public:
    yolo_detector() = default;
    yolo_detector(const std::string& dnn_path, const std::string& labels_path, bool new_coords = false): new_coords(new_coords)
    {
        load_weights(dnn_path);
        load_labels(labels_path);
    }

    void detect(
        const dlib::image_view<dlib::matrix<dlib::rgb_pixel>> image,
        std::vector<detection>& detections,
        const long image_size = 512,
        const float conf_thresh = 0.25,
        const float nms_thresh = 0.45)
    {
        dlib::matrix<dlib::rgb_pixel> scaled(image_size, image_size);
        dlib::resize_image(image, scaled);
        net(scaled);
        const auto& out8 = dlib::layer<darknet::ytag8>(net).get_output();
        const auto& out16 = dlib::layer<darknet::ytag16>(net).get_output();
        const auto& out32 = dlib::layer<darknet::ytag32>(net).get_output();
        add_detections(out8, anchors8, labels, 8, conf_thresh, detections, new_coords);
        add_detections(out16, anchors16, labels, 16, conf_thresh, detections, new_coords);
        add_detections(out32, anchors32, labels, 32, conf_thresh, detections, new_coords);
        nms(conf_thresh, nms_thresh, detections);
    }

    std::vector<std::string> get_labels() { return labels; };

    void print() const { std::cout << net << std::endl; };

    protected:
    bool new_coords = false;
    void load_weights(const std::string& dnn_path) { dlib::deserialize(dnn_path) >> net; };

    void load_labels(const std::string& labels_path)
    {
        std::ifstream fin(labels_path);
        if (not fin.good())
            throw std::runtime_error("error while opening " + labels_path);
        for (std::string line; std::getline(fin, line);)
            labels.push_back(line);
    }
    net_type net;
    std::vector<std::string> labels;
    std::vector<std::pair<float, float>> anchors8, anchors16, anchors32;
};

#endif  // yolo_h_INCLUDED
