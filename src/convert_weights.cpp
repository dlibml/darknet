#include "darknet.h"
#include "weights_visitor.h"

#include <dlib/cmd_line_parser.h>

using net_train_type = darknet::yolov3_train;
using net_infer_type = darknet::yolov3_infer;
// layer offset is 2 for yolov4x_mish yolov4_csp and scaled_yolov4, and 1 for the previous models
const unsigned int layer_offset = 1;

int main(const int argc, const char** argv)
try
{
    dlib::command_line_parser parser;
    parser.add_option("weights", "path to the darknet trained weights", 1);
    parser.add_option("num-classes", "number of classes to detect", 1);
    parser.add_option("img-size", "image size to process (default: 416)", 1);
    parser.add_option("print", "print out the network architecture");
    parser.add_option("save", "save network weights in dlib format", 1);
    parser.set_group_name("Help Options");
    parser.add_option("h", "alias for --help");
    parser.add_option("help", "display this message and exit");
    parser.parse(argc, argv);

    if (parser.option("h") or parser.option("help"))
    {
        parser.print_options();
        return EXIT_SUCCESS;
    }

    parser.check_sub_option("weights", "save");

    const long img_size = dlib::get_option(parser, "img-size", 416);
    const long num_classes = dlib::get_option(parser, "num-classes", 0);
    if (num_classes <= 0)
    {
        std::cout << "Specify the number of output classes with --num-classes\n";
        return EXIT_FAILURE;
    }
    const std::string weights_path = dlib::get_option(parser, "weights", "");
    if (weights_path.empty())
    {
        std::cout << "Specify the darknet weights with --weights\n";
        return EXIT_FAILURE;
    }

    net_train_type net_train;
    darknet::setup_detector<net_train_type, layer_offset>(net_train, num_classes, img_size);
    std::cout << "#params: " << dlib::count_parameters(net_train) << '\n';
    dlib::visit_layers_backwards(net_train, darknet::weights_visitor(weights_path));
    net_train.clean();
    net_infer_type net_infer = net_train;

    if (parser.option("save"))
    {
        dlib::serialize(parser.option("save").argument()) << net_infer;
    }

    if (parser.option("print"))
        std::cout << net_infer << '\n';

    return EXIT_SUCCESS;
}
catch (const std::exception& e)
{
    std::cout << e.what() << '\n';
    return EXIT_FAILURE;
}
