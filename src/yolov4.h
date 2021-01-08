#ifndef yolov4_h_INCLUDED
#define yolov4_h_INCLUDED

#include "yolo.h"

class yolov4 : public yolo_detector<darknet::yolov4_infer>
{
    public:
    yolov4(const std::string& dnn_path, const std::string& labels_path);
};

#endif // yolov4_h_INCLUDED
