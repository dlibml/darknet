#ifndef yolov4x_mish_h_INCLUDED
#define yolov4x_mish_h_INCLUDED

#include "yolo.h"

class yolov4x_mish : public yolo_detector<darknet::yolov4x_mish_infer>
{
    public:
    yolov4x_mish(const std::string& dnn_path, const std::string& labels_path);
};

#endif // yolov4x_mish_h_INCLUDED
