#ifndef yolov4_sam_mish_h_INCLUDED
#define yolov4_sam_mish_h_INCLUDED

#include "yolo.h"

class yolov4_sam_mish : public yolo_detector<darknet::yolov4_sam_mish_infer>
{
    public:
    yolov4_sam_mish(const std::string& dnn_path, const std::string& labels_path);
};

#endif // yolov4_sam_mish_h_INCLUDED
