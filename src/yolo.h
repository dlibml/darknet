#ifndef yolo_h_INCLUDED
#define yolo_h_INCLUDED

#include <dlib/dnn.h>

struct detection
{
    float x = 0;
    float y = 0;
    float w = 0;
    float h = 0;
    float obj = 0;
    float score = 0;
    int id = -1;
    std::string label = "";

    float xstart() const { return x - 0.5 * w; }
    float xstop() const { return x + 0.5 * w; }
    float ystart() const { return y - 0.5 * h; }
    float ystop() const { return y + 0.5 * h; }
    bool is_empty() const
    {
        detection empty;
        return *this == empty;
    }
    void set_empty()
    {
        detection empty;
        std::swap(*this, empty);
    }
    bool operator==(const detection& o) const
    {
        return std::tie(x, y, w, h, obj, score, id, label) ==
               std::tie(o.x, o.y, o.w, o.h, o.obj, o.score, o.id, o.label);
    }
    friend std::ostream& operator<<(std::ostream& os, const detection& d);
};

std::ostream& operator<<(std::ostream& os, const detection& d)
{
    os << d.id << " " << d.label << " " << d.score << " : x " << d.x << " y " << d.y << " w "
       << d.w << " h " << d.h;
    return os;
}

typedef enum
{
    IOU = 0,
    GIOU,
    DIOU,
    CIOU
} iout_t;

float iou(const detection& a, const detection& b, iout_t type)
{
    float inter_x_start = std::max(a.xstart(), b.xstart());
    float inter_x_stop = std::min(a.xstop(), b.xstop());
    float inter_y_start = std::max(a.ystart(), b.ystart());
    float inter_y_stop = std::min(a.ystop(), b.ystop());
    float inter_area = std::max(0.0f, (inter_x_stop - inter_x_start)) *
                       std::max(0.0f, (inter_y_stop - inter_y_start));
    float union_area = a.w * a.h + b.w * b.h - inter_area;
    float iou = inter_area / union_area;

    float ret = iou;

    if (type > IOU)
    {
        float cw = std::max(a.xstop(), b.xstop()) - std::min(a.xstart(), b.xstart());
        float ch = std::max(a.ystop(), b.ystop()) - std::min(a.ystart(), b.ystart());

        if (type == GIOU)
        {
            float c_area = cw * ch + 1e-16;
            ret = iou - (c_area - union_area) / c_area;
        }

        if (type > GIOU)
        {
            float c2 = cw * cw + ch * ch + 1e-16;
            float rho2 = std::pow(((b.xstart() + b.xstop()) - (a.xstart() + a.xstop())), 2) / 4 +
                         std::pow((b.ystart() + b.ystop()) - (a.ystart() + a.ystop()), 2) / 4;

            if (type == DIOU)
            {
                ret = iou - rho2 / c2;
            }
            else
            {
                float v = (4 / (M_PI * M_PI)) * pow(atan(b.w / b.h) - atan(a.w / a.h), 2);
                float alpha = v / (1 - iou + v);
                ret = iou - (rho2 / c2 + v * alpha);
            }
        }
    }

    return ret;
}

float sigmoid(const float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

void add_detections(
    const dlib::tensor& t,
    const std::vector<std::pair<float, float>>& anchors,
    const std::vector<std::string>& labels,
    const int stride,
    const float conf_thresh,
    std::vector<detection>& detections)
{
    const size_t nattr = t.k() / anchors.size();
    const size_t nclasses = nattr - 5;
    const float* const out = t.host();
    for (size_t a = 0; a < anchors.size(); ++a)
    {
        for (long y = 0; y < t.nr(); ++y)
        {
            for (long x = 0; x < t.nc(); ++x)
            {
                const float obj = sigmoid(out[dlib::tensor_index(t, 0, a * nattr + 4, y, x)]); 

                if (obj > conf_thresh)
                {
                    detection d;
                    d.x = (sigmoid(out[dlib::tensor_index(t, 0, a * nattr + 0, y, x)]) + x) / t.nc();
                    d.y = (sigmoid(out[dlib::tensor_index(t, 0, a * nattr + 1, y, x)]) + y) / t.nr();
                    d.w = std::exp(out[dlib::tensor_index(t, 0, a * nattr + 2, y, x)]) * anchors[a].first / (t.nc() * stride);
                    d.h = std::exp(out[dlib::tensor_index(t, 0, a * nattr + 3, y, x)]) * anchors[a].second / (t.nr() * stride);
                    d.obj = obj;
                        
                    for (size_t p = 0; p < nclasses; ++p)
                    {
                        const float temp = sigmoid(out[dlib::tensor_index(t, 0, a * nattr + 5 + p, y, x)]);
                        if (temp > d.score)
                        {
                            d.score = temp;
                            d.id = p;
                            d.label = labels[d.id];
                        }
                    }
                    d.score *= d.obj;
                    detections.push_back(std::move(d));
                }
            }
        }
    }
}

void nms(float conf_thresh, float nms_thresh, std::vector<detection>& detections)
{
    // conf thresh on score
    detections.erase(
        std::remove_if(
            detections.begin(),
            detections.end(),
            [conf_thresh](const detection& d) { return d.score < conf_thresh; }),
        detections.end());

    // sort conf
    std::sort(
        detections.begin(),
        detections.end(),
        [](const detection& a, const detection& b) -> bool { return a.score > b.score; });

    // nms
    for (size_t d = 0; d < detections.size(); d++)
    {
        if (detections[d].is_empty())
            continue;

        for (size_t dj = 0; dj < detections.size(); dj++)
        {
            if (d == dj || detections[dj].is_empty())
                continue;

            float _iou = iou(detections[d], detections[dj], IOU);

            if (_iou > nms_thresh)
                detections[dj].set_empty();
        }
    }

    detections.erase(
        std::remove_if(
            detections.begin(),
            detections.end(),
            [](const detection& d) { return d.is_empty(); }),
        detections.end());
}

#endif  // yolo_h_INCLUDED
