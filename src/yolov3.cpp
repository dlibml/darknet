#include "yolov3.h"

yolov3::yolov3(const std::string& dnn_path, const std::string& labels_path)
{
    load_weights(dnn_path);
    load_labels(labels_path);
    anchors8 = {{10, 13}, {16, 30}, {33, 23}};
    anchors16 = {{30, 61}, {62, 45}, {59, 119}};
    anchors32 = {{116, 90}, {156, 198}, {373, 326}};
}
