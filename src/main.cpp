#include "darknet.h"
#include "ui_utils.h"
#include "weights_visitor.h"
#include "yolo.h"

#include <dlib/cmd_line_parser.h>
#include <dlib/dir_nav.h>

template <typename net_type> auto detect(
    net_type& net,
    const dlib::matrix<dlib::rgb_pixel>& image,
    const std::vector<std::string>& labels,
    const float conf_thresh = 0.25,
    const float nms_thresh = 0.45,
    const long img_size = 416) -> std::vector<detection>
{
    dlib::matrix<dlib::rgb_pixel> scaled(img_size, img_size);
    dlib::resize_image(image, scaled);
    net(scaled);
    const auto& out8 = dlib::layer<darknet::ytag8>(net).get_output();
    const auto& out16 = dlib::layer<darknet::ytag16>(net).get_output();
    const auto& out32 = dlib::layer<darknet::ytag32>(net).get_output();
    std::vector<detection> detections;
    // add_detections(out8, {{10, 13}, {16, 30}, {33, 23}}, labels, 8, conf_thresh, detections);
    // add_detections(out16, {{30, 61}, {62, 45}, {59, 119}}, labels, 16, conf_thresh, detections);
    // add_detections(out32, {{116, 90}, {156, 198}, {373, 326}}, labels, 32, conf_thresh, detections);
    add_detections(out8, {{12, 16}, {19, 36}, {40, 28}}, labels, 8, conf_thresh, detections);
    add_detections(out16, {{36, 75}, {76, 55}, {72, 146}}, labels, 16, conf_thresh, detections);
    add_detections(out32, {{142, 110}, {192, 243}, {459, 401}}, labels, 32, conf_thresh, detections);
    nms(conf_thresh, nms_thresh, detections);
    return detections;
}

int main(const int argc, const char** argv)
try
{
    dlib::command_line_parser parser;
    parser.add_option("input", "path to video file to process (defaults to webcam)", 1);
    parser.add_option("output", "path to output video file (.mkv extension)", 1);
    parser.add_option("webcam", "index of webcam to use (default: 0)", 1);
    parser.add_option("names", "path to file with label names (one per line)", 1);
    parser.add_option("weights", "path to the darknet trained weights", 1);
    parser.add_option("img-size", "image size to process (default: 416)", 1);
    parser.add_option("conf-thresh", "confidence threshold (default: 0.25)", 1);
    parser.add_option("nms-thresh", "non-max suppression threshold (default: 0.45)", 1);
    parser.add_option("fps", "force frames per second (default: 30)", 1);
    parser.add_option("print", "print out the network architecture");
    parser.add_option("save", "save network weights in dlib format", 1);
    parser.add_option("dnn", "path to dlib saved model", 1);
    parser.set_group_name("Help Options");
    parser.add_option("h", "alias for --help");
    parser.add_option("help", "display this message and exit");
    parser.parse(argc, argv);

    if (parser.option("h") or parser.option("help"))
    {
        parser.print_options();
        webcam_window::print_keyboard_shortcuts();
        return EXIT_SUCCESS;
    }

    parser.check_incompatible_options("input", "webcam");
    parser.check_incompatible_options("weights", "dnn");
    parser.check_sub_option("weights", "save");

    const std::string weights_path = dlib::get_option(parser, "weights", "");
    const std::string names_path = dlib::get_option(parser, "names", "");
    const int webcam_idx = dlib::get_option(parser, "webcam", 0);
    float fps = dlib::get_option(parser, "fps", 30);
    const long img_size = dlib::get_option(parser, "img-size", 416);
    const float conf_thresh = dlib::get_option(parser, "conf-thresh", 0.25);
    const float nms_thresh = dlib::get_option(parser, "nms-thresh", 0.45);
    const std::string dnn_path = dlib::get_option(parser, "dnn", "");
    if (weights_path.empty() and dnn_path.empty())
    {
        std::cout << "Please provide a path to the trained weights\n";
        return EXIT_FAILURE;
    }
    std::vector<std::string> labels;
    if (names_path.empty())
    {
        std::cout << "Please provide a path to the label names file\n";
        return EXIT_FAILURE;
    }
    else
    {
        std::ifstream fin(names_path);
        for(std::string line; std::getline(fin, line); )
        {
            labels.push_back(line);
        }
    }
    std::cout << "found " << labels.size() << " classes\n";

    darknet::yolov4_sam_mish_infer yolo;
    if (dlib::file_exists(weights_path))
    {
        darknet::yolov4_sam_mish_train net;
        darknet::setup(net, labels.size(), img_size);
        std::cout << "#params: " << dlib::count_parameters(net) << '\n';
        dlib::visit_layers_backwards(net, darknet::weights_visitor(weights_path));
        yolo = net;
        if (parser.option("save"))
        {
            yolo.clean();
            dlib::serialize(parser.option("save").argument()) << yolo;
        }
    }
    else if (dlib::file_exists(dnn_path))
    {
        dlib::deserialize(dnn_path) >> yolo;
    }
    else
    {
        std::cout << "ERROR: could not find the network file: "
                  << (dnn_path.empty() ? weights_path : dnn_path) << '\n';
        return EXIT_FAILURE;
    }
    if (parser.option("print"))
        std::cout << yolo << '\n';

    webcam_window win;
    win.conf_thresh = conf_thresh;
    const std::string out_path = dlib::get_option(parser, "output", "");
    cv::VideoCapture vid_src;
    cv::VideoWriter vid_snk;
    if (parser.option("input"))
    {
        const std::string video_path = parser.option("input").argument();
        cv::VideoCapture file(video_path);
        if (not parser.option("fps"))
            fps = file.get(cv::CAP_PROP_FPS);
        vid_src = file;
        win.mirror = false;
    }
    else
    {
        cv::VideoCapture cap(webcam_idx);
        cap.set(cv::CAP_PROP_FPS, fps);
        vid_src = cap;
        win.mirror = true;
    }
    int width, height;
    {
        cv::Mat cv_tmp;
        vid_src.read(cv_tmp);
        width = cv_tmp.cols;
        height = cv_tmp.rows;
    }
    if (not out_path.empty())
    {
        vid_snk = cv::VideoWriter(out_path, cv::VideoWriter::fourcc('X', '2', '6', '4'), fps, cv::Size(width, height));
    }

    const auto label_to_color = get_color_map(labels);
    dlib::running_stats_decayed<float> rs(100);
    std::cout << std::fixed << std::setprecision(2);
    while (not win.is_closed())
    {
        dlib::matrix<dlib::rgb_pixel> image;
        cv::Mat cv_cap;
        if (!vid_src.read(cv_cap))
        {
            break;
        }
        // convert the BRG opencv image to RGB dlib image
        const dlib::cv_image<dlib::bgr_pixel> tmp(cv_cap);
        if (win.mirror)
            dlib::flip_image_left_right(tmp, image);
        else
            dlib::assign_image(image, tmp);
        win.clear_overlay();
        const auto t0 = std::chrono::steady_clock::now();
        const auto detections = detect(yolo, image, labels, win.conf_thresh, nms_thresh, img_size);
        const auto t1 = std::chrono::steady_clock::now();
        rs.add(std::chrono::duration_cast<std::chrono::duration<float>>(t1 - t0).count());
        std::cout << "avg fps: " << 1.0f / rs.mean() << '\r' << std::flush;
        render_bounding_boxes(image, detections, label_to_color);
        win.set_image(image);
        if (not out_path.empty())
        {
            dlib::matrix<dlib::bgr_pixel> bgr_img(height, width);
            dlib::assign_image(bgr_img, image);
            vid_snk.write(dlib::toMat(bgr_img));
        }
    }
    if (not out_path.empty())
        vid_snk.release();

    return EXIT_SUCCESS;
}
catch (const std::exception& e)
{
    std::cout << e.what() << '\n';
    return EXIT_FAILURE;
}
