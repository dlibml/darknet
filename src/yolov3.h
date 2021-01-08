#ifndef yolov3_h_INCLUDED
#define yolov3_h_INCLUDED

#include "yolo.h"

class yolov3 : public yolo_detector<darknet::yolov3_infer>
{
    public:
    yolov3(const std::string& dnn_path, const std::string& labels_path);
};

#endif // yolov3_h_INCLUDED
