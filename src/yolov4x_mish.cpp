#include "yolov4x_mish.h"

yolov4x_mish::yolov4x_mish(const std::string& dnn_path, const std::string& labels_path)
{
    new_coords = true;
    load_weights(dnn_path);
    load_labels(labels_path);
    anchors8 = {{12, 16}, {19, 36}, {40, 28}};
    anchors16 = {{36, 75}, {76, 55}, {72, 146}};
    anchors32 = {{142, 110}, {192, 243}, {459, 401}};
}
