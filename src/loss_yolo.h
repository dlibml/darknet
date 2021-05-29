#ifndef loss_yolo_h_INCLUDED
#define loss_yolo_h_INCLUDED

#include <dlib/dnn.h>

namespace dlib
{
    using yolo_rect = mmod_rect;
    inline bool operator<(const yolo_rect& lhs, const yolo_rect& rhs)
    {
        return lhs.detection_confidence < rhs.detection_confidence;
    }

    // struct yolo_rect
    // {
    //     yolo_rect(const drectangle& r) : rect(r) {}
    //     yolo_rect(const drectangle& r, double score) : rect(r), detection_confidence(score) {}
    //     yolo_rect(const drectangle& r, double score, const std::string& label)
    //         : rect(r),
    //           detection_confidence(score),
    //           label(label)
    //     {
    //     }
    //     drectangle rect;
    //     double detection_confidence = 0;
    //     bool ignore = false;
    //     std::string label;

    //     operator rectangle() const { return rect; }

    //     bool operator==(const mmod_rect& rhs) const
    //     {
    //         return rect == rhs.rect && detection_confidence == rhs.detection_confidence &&
    //                ignore == rhs.ignore && label == rhs.label;
    //     }

    //     bool operator<(const yolo_rect& item) const
    //     {
    //         return detection_confidence < item.detection_confidence;
    //     }
    // };

    // clang-format off
    struct yolo_options
    {
    public:
        struct anchor_box_details
        {
            anchor_box_details() = default;
            anchor_box_details(unsigned long w, unsigned long h) : width(w), height(h) {}

            unsigned long width = 0;
            unsigned long height = 0;

            friend inline void serialize(const anchor_box_details& item, std::ostream& out)
            {
                int version = 0;
                serialize(version, out);
                serialize(item.width, out);
                serialize(item.height, out);
            }

            friend inline void deserialize(anchor_box_details& item, std::istream& in)
            {
                int version = 0;
                deserialize(version, in);
                deserialize(item.width, in);
                deserialize(item.height, in);
            }
        };

        yolo_options() = default;

        template <template <typename> class TAG_TYPE>
        void add_anchors(const std::vector<anchor_box_details>& boxes)
        {
            anchors[tag_id<TAG_TYPE>::id] = boxes;
        }

        // map between the stride and the anchor boxes
        std::unordered_map<int, std::vector<anchor_box_details>> anchors;
        std::vector<std::string> labels;
        double confidence_threshold = 0.25;
        float truth_match_iou_threshold = 0.4;
        test_box_overlap overlaps_nms = test_box_overlap(0.45, 1.0);
        test_box_overlap overlaps_ignore = test_box_overlap(0.5, 1.0);
        float lambda_obj = 1.0f;
        float lambda_noobj = 0.5f;
        float lambda_bbr = 5.0f;
        float lambda_cls = 1.0f;

    };

    inline void serialize(const yolo_options& item, std::ostream& out)
    {
        int version = 1;
        serialize(version, out);
        serialize(item.anchors, out);
        serialize(item.confidence_threshold, out);
        serialize(item.truth_match_iou_threshold, out);
        serialize(item.overlaps_nms, out);
        serialize(item.overlaps_ignore, out);
        serialize(item.lambda_obj, out);
        serialize(item.lambda_bbr, out);
        serialize(item.lambda_cls, out);
    }

    inline void deserialize(yolo_options& item, std::istream& in)
    {
        int version = 0;
        deserialize(version, in);
        if (version != 1)
            throw serialization_error("Unexpected version found while deserializing dlib::yolo_options.");
        deserialize(item.anchors, in);
        deserialize(item.confidence_threshold, in);
        deserialize(item.truth_match_iou_threshold, in);
        deserialize(item.overlaps_nms, in);
        deserialize(item.overlaps_ignore, in);
        deserialize(item.lambda_obj, in);
        deserialize(item.lambda_bbr, in);
        deserialize(item.lambda_cls, in);
    }

    inline std::ostream& operator<<(std::ostream& out, const std::unordered_map<int, std::vector<yolo_options::anchor_box_details>>& anchors)
    {
        out << "anchors: " << anchors.size();
        return out;
    }

    namespace impl
    {
        template <template <typename> class TAG_TYPE, template <typename> class... TAG_TYPES>
        struct yolo_helper_impl
        {
            constexpr static size_t tag_count()
            {
                return 1 + yolo_helper_impl<TAG_TYPES...>::tag_count();
            }

            static void list_tags(std::ostream& out)
            {
                out << tag_id<TAG_TYPE>::id << (tag_count() > 1 ? "," : "");
                yolo_helper_impl<TAG_TYPES...>::list_tags(out);
            }

            template <typename SUBNET>
            static void tensor_to_dets(
                const tensor& input,
                const SUBNET& sub,
                const long n,
                const yolo_options& options,
                std::vector<yolo_rect>& dets
            )
            {
                yolo_helper_impl<TAG_TYPE>::tensor_to_dets(input, sub, n, options, dets);
                yolo_helper_impl<TAG_TYPES...>::tensor_to_dets(input, sub, n, options, dets);
            }

            template <
                typename const_label_iterator,
                typename SUBNET
            >
            static void tensor_to_grad(
                const tensor& input_tensor,
                const_label_iterator truth,
                SUBNET& sub,
                const long n,
                const yolo_options& options,
                double& loss
            )
            {
                yolo_helper_impl<TAG_TYPE>::tensor_to_grad(input_tensor, truth, sub, n, options, loss);
                yolo_helper_impl<TAG_TYPES...>::tensor_to_grad(input_tensor, truth, sub, n, options, loss);
            }
        };

        template <template <typename> class TAG_TYPE>
        struct yolo_helper_impl<TAG_TYPE>
        {
            constexpr static size_t tag_count() { return 1; }

            static void list_tags(std::ostream& out) { out << tag_id<TAG_TYPE>::id; }

            template <typename net_type>
            static void tensor_to_dets(
                const tensor& input_tensor,
                const net_type& net,
                const long n,
                const yolo_options& options,
                std::vector<yolo_rect>& dets
            )
            {
                DLIB_CASSERT(net.sample_expansion_factor() == 1, net.sample_expansion_factor());
                const tensor& output_tensor = layer<TAG_TYPE>(net).get_output();
                const float stride_x = static_cast<float>(input_tensor.nc()) / output_tensor.nc();
                const float stride_y = static_cast<float>(input_tensor.nr()) / output_tensor.nr();
                const auto& anchors = options.anchors.at(tag_id<TAG_TYPE>::id);
                const size_t num_attribs = output_tensor.k() / anchors.size();
                const size_t num_classes = num_attribs - 5;
                const float* const out = output_tensor.host();
                for (size_t a = 0; a < anchors.size(); ++a)
                {
                    for (long r = 0; r < output_tensor.nr(); ++r)
                    {
                        for (long c = 0; c < output_tensor.nc(); ++c)
                        {
                            const float obj = sigmoid(out[tensor_index(output_tensor, n, a * num_attribs + 4, r, c)]);
                            if (obj > options.confidence_threshold)
                            {
                                const float x = (sigmoid(out[tensor_index(output_tensor, n, a * num_attribs + 0, r, c)]) + c) * stride_x;
                                const float y = (sigmoid(out[tensor_index(output_tensor, n, a * num_attribs + 1, r, c)]) + r) * stride_y;
                                const float w = std::exp(out[tensor_index(output_tensor, n, a * num_attribs + 2, r, c)]) * anchors[a].width;
                                const float h = std::exp(out[tensor_index(output_tensor, n, a * num_attribs + 3, r, c)]) * anchors[a].height;
                                yolo_rect d(centered_drect(dpoint(x, y), w, h), 0);
                                for (size_t k = 0; k < num_classes; ++k)
                                {
                                    const float p = sigmoid(out[tensor_index(output_tensor, n, a * num_attribs + 5 + k, r, c)]);
                                    if (p > d.detection_confidence)
                                    {
                                        d.detection_confidence = p;
                                        d.label = options.labels[k];
                                    }
                                }
                                d.detection_confidence *= obj;
                                if (d.detection_confidence > options.confidence_threshold)
                                    dets.push_back(std::move(d));
                            }
                        }
                    }
                }
            }

            // loss and gradient for a positve sample
            static void binary_loss_log_and_gradient_pos(
                const float input,
                const double scale,
                double& loss,
                float& grad
            )
            {
                loss += scale * log1pexp(-input);
                grad = scale * (sigmoid(input) - 1);
            }

            // loss and gradient for a negative sample
            static void binary_loss_log_and_gradient_neg(
                const float input,
                const double scale,
                double& loss,
                float& grad
            )
            {
                loss += scale * (input + log1pexp(-input));
                grad = scale * sigmoid(input);
            }

            template <
                typename const_label_iterator,
                typename SUBNET
            >
            static void tensor_to_grad(
                const tensor& input_tensor,
                const_label_iterator truth,
                SUBNET& sub,
                const long n,
                const yolo_options& options,
                double& loss
            )
            {
                const tensor& output_tensor = layer<TAG_TYPE>(sub).get_output();
                tensor& grad = layer<TAG_TYPE>(sub).get_gradient_input();
                const float* const out_data = output_tensor.host();
                float* g = grad.host_write_only();
                for (size_t i = 0; i < grad.size(); ++i)
                    g[i] = 0;
                const auto stride_x = static_cast<double>(input_tensor.nc()) / output_tensor.nc();
                const auto stride_y = static_cast<double>(input_tensor.nr()) / output_tensor.nr();

                const auto& anchors = options.anchors.at(tag_id<TAG_TYPE>::id);
                const size_t num_attribs = output_tensor.k() / anchors.size();
                const size_t num_classes = num_attribs - 5;
                // std::cout << "stride: " << stride_x << 'x' << stride_y << std::endl;

                const double scale = 1.0 / (output_tensor.nr() * output_tensor.nc());
                if (truth->empty())
                {
                    for (size_t a = 0; a < anchors.size(); ++a)
                    {
                        for (long r = 0; r < output_tensor.nr(); ++r)
                        {
                            for (long c = 0; c < output_tensor.nc(); ++c)
                            {
                                const auto obj_idx = tensor_index(output_tensor, n, a * num_attribs + 4, r, c);
                                binary_loss_log_and_gradient_neg(out_data[obj_idx], scale * options.lambda_noobj, loss, g[obj_idx]);
                            }
                        }
                    }
                    return;
                }

                const double eps = 1e-9;
                for (const yolo_rect& truth_box : *truth)
                {
                    const auto truth_center = dcenter(truth_box.rect);
                    // const point tc = input_tensor_to_output_tensor(layer<TAG_TYPE>(sub), truth_center);
                    const point tc(truth_center.x() / stride_x, truth_center.y() / stride_y);;
                    // std::cout << "truth: " << truth_center << " (" << truth_box.rect.width() << 'x' << truth_box.rect.height()  << "), label: " << truth_box.label << std::endl;
                    // std::cout << "center: " << tc << std::endl;
                    for (size_t a = 0; a < anchors.size(); ++a)
                    {
                        const drectangle anchor = centered_drect(truth_box.rect, anchors[a].width, anchors[a].height);
                        const double inner = truth_box.rect.intersect(anchor).area();
                        const double outer = (truth_box.rect + anchor).area();
                        const double iou = inner / outer;
                        // std::cout << anchor.width() << 'x' << anchor.height() << ": " << iou << std::endl;
                        for (long r = 0; r < output_tensor.nr(); ++r)
                        {
                            for (long c = 0; c < output_tensor.nc(); ++c)
                            {
                                const auto obj_idx = tensor_index(output_tensor, n, a * num_attribs + 4, r, c);
                                // This cell is responsible for the detection of the object
                                if (iou > options.truth_match_iou_threshold && c == tc.x() && r == tc.y())
                                {
                                    binary_loss_log_and_gradient_pos(out_data[obj_idx], scale * options.lambda_obj, loss, g[obj_idx]);

                                    // Perform bbr
                                    const auto x_idx = tensor_index(output_tensor, n, a * num_attribs + 0, r, c);
                                    const auto y_idx = tensor_index(output_tensor, n, a * num_attribs + 1, r, c);
                                    const auto w_idx = tensor_index(output_tensor, n, a * num_attribs + 2, r, c);
                                    const auto h_idx = tensor_index(output_tensor, n, a * num_attribs + 3, r, c);

                                    double pw = anchors[a].width;
                                    double ph = anchors[a].height;
                                    float dx = out_data[x_idx];
                                    float dy = out_data[y_idx];
                                    float dw = out_data[w_idx];
                                    float dh = out_data[h_idx];
                                    double target_dx = std::log((eps + truth_center.x() - c * stride_x) / (eps + stride_x * (c + 1) - truth_center.x()));
                                    double target_dy = std::log((eps + truth_center.y() - r * stride_y) / (eps + stride_y * (r + 1) - truth_center.y()));
                                    double target_dw = std::log(truth_box.rect.width() / pw);
                                    double target_dh = std::log(truth_box.rect.height() / ph);
                                    // std::cout << "out: " << dx << ' ' << dy << ' ' << dw << ' ' << dh << std::endl;
                                    // std::cout << "target: " << target_dx << ' ' << target_dy << ' ' << target_dw << ' ' << target_dh << std::endl;

                                    // Compute smoothed L1 loss
                                    dx = dx - target_dx;
                                    dy = dy - target_dy;
                                    dw = dw - target_dw;
                                    dh = dh - target_dh;
                                    double ldx = std::abs(dx) < 1 ? 0.5 * dx * dx : std::abs(dx) - 0.5;
                                    double ldy = std::abs(dy) < 1 ? 0.5 * dy * dy : std::abs(dy) - 0.5;
                                    double ldw = std::abs(dw) < 1 ? 0.5 * dw * dw : std::abs(dw) - 0.5;
                                    double ldh = std::abs(dh) < 1 ? 0.5 * dh * dh : std::abs(dh) - 0.5;
                                    loss += options.lambda_bbr * (ldx + ldy + ldw + ldh);
                                    // std::cout << "diff: " << dx << ' ' << dy << ' ' << dw << ' ' << dh << std::endl;
                                    // std::cout << "bbr loss: " << options.lambda_bbr * (ldx + ldy + ldw + ldh) << std::endl;

                                    ldx = put_in_range(-1, 1, dx);
                                    ldy = put_in_range(-1, 1, dy);
                                    ldw = put_in_range(-1, 1, dw);
                                    ldh = put_in_range(-1, 1, dh);

                                    g[x_idx] += scale * options.lambda_bbr * ldx;
                                    g[y_idx] += scale * options.lambda_bbr * ldy;
                                    g[w_idx] += scale * options.lambda_bbr * ldw;
                                    g[h_idx] += scale * options.lambda_bbr * ldh;

                                    // Perform class loss
                                    const size_t l_idx = std::find(options.labels.begin(),
                                                         options.labels.end(),
                                                         truth_box.label) - options.labels.begin();

                                    for (size_t k = 0; k < num_classes; ++k)
                                    {
                                        const size_t cls_idx = tensor_index(output_tensor, n, a * num_attribs + 5 + k, r, c);
                                        if (k == l_idx)
                                            binary_loss_log_and_gradient_pos(out_data[cls_idx], scale * options.lambda_cls, loss, g[cls_idx]);
                                        else
                                            binary_loss_log_and_gradient_neg(out_data[cls_idx], scale * options.lambda_cls, loss, g[cls_idx]);
                                    }
                                }
                                else
                                {
                                    binary_loss_log_and_gradient_neg(out_data[obj_idx], scale * options.lambda_noobj, loss, g[obj_idx]);
                                }
                            }
                        }
                    }
                    // std::cin.get();
                }
            }
        };
    }  // namespace impl

    template <template <typename> class... TAG_TYPES>
    class loss_yolo_
    {
        static void list_tags(std::ostream& out) { impl::yolo_helper_impl<TAG_TYPES...>::list_tags(out); }

    public:

        typedef std::vector<yolo_rect> training_label_type;
        typedef std::vector<yolo_rect> output_label_type;

        constexpr static size_t tag_count() { return impl::yolo_helper_impl<TAG_TYPES...>::tag_count(); }

        loss_yolo_() {};

        loss_yolo_(const yolo_options& options) : options(options) { }

        template <
            typename SUB_TYPE,
            typename label_iterator
            >
        void to_label (
            const tensor& input_tensor,
            const SUB_TYPE& sub,
            label_iterator iter
        ) const
        {
            DLIB_CASSERT(sub.sample_expansion_factor() == 1, sub.sample_expansion_factor());
            std::vector<yolo_rect> dets_accum;
            output_label_type final_dets;
            for (long i = 0; i < input_tensor.num_samples(); ++i)
            {
                dets_accum.clear();
                impl::yolo_helper_impl<TAG_TYPES...>::tensor_to_dets(input_tensor, sub, i, options, dets_accum);

                // Do non-max suppression
                std::sort(dets_accum.rbegin(), dets_accum.rend());
                final_dets.clear();
                for (size_t j = 0; j < dets_accum.size(); ++j)
                {
                    if (overlaps_any_box_nms(final_dets, dets_accum[j].rect))
                        continue;

                    final_dets.push_back(dets_accum[j]);
                }

                *iter++ = std::move(final_dets);
            }
        }

        template <
            typename const_label_iterator,
            typename SUBNET
        >
        double compute_loss_value_and_gradient (
            const tensor& input_tensor,
            const_label_iterator truth,
            SUBNET& sub
        ) const
        {
            DLIB_CASSERT(input_tensor.num_samples() != 0);
            DLIB_CASSERT(sub.sample_expansion_factor() == 1);
            DLIB_CASSERT(truth->size() > 0);
            double loss = 0;
            for (long i = 0; i < input_tensor.num_samples(); ++i)
            {
                double sample_loss = 0;
                impl::yolo_helper_impl<TAG_TYPES...>::tensor_to_grad(input_tensor, truth, sub, i, options, sample_loss);
                loss += sample_loss;
            }
            return loss / input_tensor.num_samples();
        }

        void adjust_threshold(float conf_thresh) { options.confidence_threshold = conf_thresh; }

        friend void serialize(const loss_yolo_& item, std::ostream& out)
        {
            serialize("loss_yolo_", out);
            size_t count = tag_count();
            serialize(count, out);
            serialize(item.options, out);
        }

        friend void deserialize(loss_yolo_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "loss_yolo_")
                throw serialization_error("Unexpected version found while deserializing dlib::loss_yolo_.");
            size_t count = 0;
            deserialize(count, in);
            if (count != tag_count())
                throw serialization_error("Invalid number of detection tags " + std::to_string(count) +
                                          ", while deserializing dlib::loss_yolo_, expecting " +
                                          std::to_string(tag_count()) + "tags instead.");
            deserialize(item.options, in);
        }

        friend std::ostream& operator<<(std::ostream& out, const loss_yolo_& )
        {
            out << "loss_yolo\t (" << tag_count() << " output tags: ";
            list_tags(out);
            out << ")";
            return out;
        }

    private:

        static constexpr float sigmoid(const float x) { return 1.0f / (1.0f + std::exp(-x)); }

        yolo_options options;

        template <typename T>
        inline bool overlaps_any_box_nms (
            const std::vector<T>& rects,
            const drectangle& rect
        ) const
        {
            for (const auto& r : rects)
            {
                if (options.overlaps_nms(r.rect, rect))
                    return true;
            }
            return false;
        }
    };

    template <template <typename> class TAG_1, template <typename> class TAG_2, template <typename> class TAG_3, typename SUBNET>
    using loss_yolo = add_loss_layer<loss_yolo_<TAG_1, TAG_2, TAG_3>, SUBNET>;
    // clang-format on
}  // namespace dlib

#endif  // loss_yolo_h_INCLUDED
