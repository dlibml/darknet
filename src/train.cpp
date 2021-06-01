#include "darknet.h"
#include "loss_yolo.h"
#include "ui_utils.h"

#include <dlib/cmd_line_parser.h>
#include <dlib/data_io.h>
#include <dlib/dir_nav.h>
#include <dlib/image_io.h>

using namespace std;
using namespace dlib;

using darknet::ytag8, darknet::ytag16, darknet::ytag32;
using net_type = dlib::loss_yolo<ytag8, ytag16, ytag32, darknet::yolov3_train>;

int main(const int argc, const char** argv)
try
{
    command_line_parser parser;
    parser.add_option("learning-rate", "initial learning rate (default: 0.1)", 1);
    parser.add_option("batch-size", "mini batch size (default: 8)", 1);
    parser.set_group_name("Help Options");
    parser.add_option("h", "alias of --help");
    parser.add_option("help", "display this message and exit");
    parser.parse(argc, argv);
    if (parser.number_of_arguments() == 0 || parser.option("h") || parser.option("help"))
    {
        parser.print_options();
        cout << "Give the path to a folder containing the training.xml file." << endl;
        return 0;
    }
    const double learning_rate = get_option(parser, "learning-rate", 0.1);
    const size_t batch_size = get_option(parser, "batch-size", 8);
    const std::string data_directory = parser[0];
    image_dataset_metadata::dataset dataset;
    image_dataset_metadata::load_image_dataset_metadata(dataset, data_directory + "/training.xml");
    std::cout << "# images: " << dataset.images.size() << std::endl;
    std::map<std::string, size_t> labels;
    size_t num_objects = 0;
    for (const auto& im : dataset.images)
    {
        for (const auto& b : im.boxes)
        {
            labels[b.label]++;
            ++num_objects;
        }
    }
    std::cout << "# labels: " << labels.size() << std::endl;

    yolo_options options;
    for (const auto& [label, count] : labels)
    {
        std::cout <<  " - " << label << ": " << count << " (" << (100.0*count)/num_objects << "%)" << std::endl;
        options.labels.push_back(label);
    }
    options.confidence_threshold = 0.25;
    options.add_anchors<ytag8>({{10, 13}, {16, 30}, {33, 23}});
    options.add_anchors<ytag16>({{30, 61}, {62, 45}, {59, 119}});
    options.add_anchors<ytag32>({{116, 90}, {156, 198}, {373, 326}});
    options.overlaps_nms = dlib::test_box_overlap(0.45);
    net_type net(options);
    darknet::setup_detector(net, options.labels.size());

    dnn_trainer<net_type> trainer(net, sgd(0.0005, 0.9));
    trainer.be_verbose();
    trainer.set_iterations_without_progress_threshold(5000);
    trainer.set_learning_rate(learning_rate);
    trainer.set_mini_batch_size(batch_size);
    trainer.set_min_learning_rate(1e-5);
    trainer.set_synchronization_file("yolov3_voc2012_sync", std::chrono::minutes(15));
    std::cout << trainer << std::endl;

    dlib::pipe<std::pair<matrix<rgb_pixel>, std::vector<yolo_rect>>> train_data(1000);
    auto loader = [&dataset, &data_directory, &train_data](time_t seed) {
        dlib::rand rnd(time(nullptr) + seed);
        matrix<rgb_pixel> image, letterbox;
        std::pair<matrix<rgb_pixel>, std::vector<yolo_rect>> temp;
        while (train_data.is_enabled())
        {
            std::vector<yolo_rect> boxes;
            auto idx = rnd.get_random_32bit_number() % dataset.images.size();
            load_image(image, data_directory + "/" + dataset.images[idx].filename);

            auto tform = rectangle_transform(letterbox_image(image, letterbox, 416));
            for (const auto& box : dataset.images[idx].boxes)
            {
                boxes.push_back(yolo_rect(tform(box.rect), 1, box.label));
            }
            disturb_colors(image, rnd);
            temp.first = letterbox;
            temp.second = boxes;
            train_data.enqueue(temp);
        }
    };

    std::vector<std::thread> data_loaders;
    for (int i = 0; i < 4; ++i)
        data_loaders.emplace_back([loader, i]() { loader(i + 1); });

    // image_window win;
    // while (true)
    // {
    //     std::pair<matrix<rgb_pixel>, std::vector<yolo_rect>> temp;
    //     train_data.dequeue(temp);
    //     win.set_image(temp.first);
    //     for (const auto& r : temp.second)
    //         win.add_overlay(r.rect, rgb_pixel(255, 0, 0), r.label);
    //     cin.get();
    //     win.clear_overlay();
    // }

    std::vector<matrix<rgb_pixel>> images;
    std::vector<std::vector<yolo_rect>> bboxes;
    while (trainer.get_learning_rate() > trainer.get_min_learning_rate())
    {
        images.clear();
        bboxes.clear();
        std::pair<matrix<rgb_pixel>, std::vector<yolo_rect>> temp;
        while (images.size() < trainer.get_mini_batch_size())
        {
            train_data.dequeue(temp);
            images.push_back(std::move(temp.first));
            bboxes.push_back(std::move(temp.second));
        }
        trainer.train_one_step(images, bboxes);
    }

    for (auto&& l : data_loaders)
        l.join();

    trainer.get_net();
    serialize("yolov3_voc2012.dnn") << net;
}
catch (const std::exception& e)
{
    std::cout << e.what() << '\n';
    return EXIT_FAILURE;
}
