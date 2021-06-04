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
        double truth_match_iou_threshold = 0.3;
        double ignore_iou_threshold = 0.7;
        test_box_overlap overlaps_nms = test_box_overlap(0.45, 1.0);
        test_box_overlap overlaps_ignore = test_box_overlap(0.5, 1.0);
        double lambda_obj = 1.0f;
        double lambda_noobj = 1.0f;
        double lambda_bbr = 1.0f;
        double lambda_cls = 1.0f;

    };

    inline void serialize(const yolo_options& item, std::ostream& out)
    {
        int version = 1;
        serialize(version, out);
        serialize(item.anchors, out);
        serialize(item.confidence_threshold, out);
        serialize(item.truth_match_iou_threshold, out);
        serialize(item.ignore_iou_threshold, out);
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
        deserialize(item.ignore_iou_threshold, in);
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

            template <typename SUBNET>
            static void tensor_to_dets(
                const tensor& input_tensor,
                const SUBNET& sub,
                const long n,
                const yolo_options& options,
                std::vector<yolo_rect>& dets
            )
            {
                DLIB_CASSERT(sub.sample_expansion_factor() == 1, sub.sample_expansion_factor());
                const tensor& output_tensor = layer<TAG_TYPE>(sub).get_output();
                const auto stride_x = static_cast<double>(input_tensor.nc()) / output_tensor.nc();
                const auto stride_y = static_cast<double>(input_tensor.nr()) / output_tensor.nr();
                const auto& anchors = options.anchors.at(tag_id<TAG_TYPE>::id);
                const long num_feats = output_tensor.k() / anchors.size();
                const long num_classes = num_feats - 5;
                const float* const out_data = output_tensor.host();

                for (size_t a = 0; a < anchors.size(); ++a)
                {
                    for (long r = 0; r < output_tensor.nr(); ++r)
                    {
                        for (long c = 0; c < output_tensor.nc(); ++c)
                        {
                            const float obj = sigmoid(out_data[tensor_index(output_tensor, n, a * num_feats + 4, r, c)]);
                            if (obj > options.confidence_threshold)
                            {
                                const double x = (sigmoid(out_data[tensor_index(output_tensor, n, a * num_feats + 0, r, c)]) + c) * stride_x;
                                const double y = (sigmoid(out_data[tensor_index(output_tensor, n, a * num_feats + 1, r, c)]) + r) * stride_y;
                                const double w = std::exp(out_data[tensor_index(output_tensor, n, a * num_feats + 2, r, c)]) * anchors[a].width;
                                const double h = std::exp(out_data[tensor_index(output_tensor, n, a * num_feats + 3, r, c)]) * anchors[a].height;
                                yolo_rect d(centered_drect(dpoint(x, y), w, h), 0);
                                for (long k = 0; k < num_classes; ++k)
                                {
                                    const float p = sigmoid(out_data[tensor_index(output_tensor, n, a * num_feats + 5 + k, r, c)]);
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

            // Loss and gradient for a positve sample
            static inline void binary_loss_log_and_gradient_pos(
                const double z,
                const double scale,
                double& loss,
                float& grad
            )
            {
                loss += scale * log1pexp(-z);
                grad = scale * (sigmoid(z) - 1);
            }

            // Loss and gradient for a negative sample
            static inline void binary_loss_log_and_gradient_neg(
                const double z,
                const double scale,
                double& loss,
                float& grad
            )
            {
                loss += scale * (z + log1pexp(-z));
                grad = scale * sigmoid(z);
            }

            static inline double compute_iou(const yolo_rect& a, const yolo_rect& b)
            {
                return a.rect.intersect(b.rect).area() / static_cast<double>((a.rect + b.rect).area());
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
                DLIB_CASSERT(sub.sample_expansion_factor() == 1, sub.sample_expansion_factor());
                const tensor& output_tensor = layer<TAG_TYPE>(sub).get_output();
                const auto stride_x = static_cast<double>(input_tensor.nc()) / output_tensor.nc();
                const auto stride_y = static_cast<double>(input_tensor.nr()) / output_tensor.nr();
                const auto& anchors = options.anchors.at(tag_id<TAG_TYPE>::id);
                const long num_feats = output_tensor.k() / anchors.size();
                const long num_classes = num_feats - 5;
                const float* const out_data = output_tensor.host();
                tensor& grad = layer<TAG_TYPE>(sub).get_gradient_input();
                float* g = grad.host();
                const double scale = 1.0 / (output_tensor.num_samples() * output_tensor.nr() * output_tensor.nc());

                // Compute the objectness loss for all grid cells
                for (long r = 0; r < output_tensor.nr(); ++r)
                {
                    for (long c = 0; c < output_tensor.nc(); ++c)
                    {
                        for (size_t a = 0; a < anchors.size(); ++a)
                        {
                            const auto x_idx = tensor_index(output_tensor, n, a * num_feats + 0, r, c);
                            const auto y_idx = tensor_index(output_tensor, n, a * num_feats + 1, r, c);
                            const auto w_idx = tensor_index(output_tensor, n, a * num_feats + 2, r, c);
                            const auto h_idx = tensor_index(output_tensor, n, a * num_feats + 3, r, c);
                            const auto o_idx = tensor_index(output_tensor, n, a * num_feats + 4, r, c);

                            // The prediction at r, c for anchor a
                            const yolo_rect pred(centered_drect(
                                dpoint((sigmoid(out_data[x_idx]) + c) * stride_x, (sigmoid(out_data[y_idx]) + r) * stride_y),
                                std::exp(out_data[w_idx]) * anchors[a].width,
                                std::exp(out_data[h_idx]) * anchors[a].height)
                            );

                            // Find the best IoU for all ground truth boxes
                            double best_iou = 0;
                            for (const yolo_rect& truth_box : *truth)
                            {
                                if (truth_box.ignore)
                                    continue;
                                best_iou = std::max(best_iou, compute_iou(truth_box, pred));
                            }

                            // Only incur loss for the boxes that are below a certain IoU threshold
                            if (best_iou < options.ignore_iou_threshold)
                                binary_loss_log_and_gradient_neg(out_data[o_idx], scale * options.lambda_noobj, loss, g[o_idx]);
                        }
                    }
                }

                // Now find the best anchor box for each truth box
                for (const yolo_rect& truth_box : *truth)
                {
                    if (truth_box.ignore)
                        continue;
                    const dpoint t_center = dcenter(truth_box);
                    const long c = t_center.x() / stride_x;
                    const long r = t_center.y() / stride_y;
                    double best_iou = 0;
                    size_t best_a = 0;
                    for (size_t a = 0; a < anchors.size(); ++a)
                    {
                        const yolo_rect anchor(centered_drect(t_center, anchors[a].width, anchors[a].height));
                        const double iou = compute_iou(truth_box, anchor);
                        if (iou > best_iou)
                        {
                            best_iou = iou;
                            best_a = a;
                        }
                    }

                    // Only update those anchors that match reasonably well
                    if (best_iou < options.truth_match_iou_threshold)
                        continue;

                    const auto x_idx = tensor_index(output_tensor, n, best_a * num_feats + 0, r, c);
                    const auto y_idx = tensor_index(output_tensor, n, best_a * num_feats + 1, r, c);
                    const auto w_idx = tensor_index(output_tensor, n, best_a * num_feats + 2, r, c);
                    const auto h_idx = tensor_index(output_tensor, n, best_a * num_feats + 3, r, c);
                    const auto o_idx = tensor_index(output_tensor, n, best_a * num_feats + 4, r, c);

                    // This grid cell should detect an object
                    binary_loss_log_and_gradient_pos(out_data[o_idx], scale * options.lambda_obj, loss, g[o_idx]);

                    const double target_dx = t_center.x() / stride_x - c;
                    const double target_dy = t_center.y() / stride_y - r;
                    const double target_dw = std::log(truth_box.rect.width() / static_cast<double>(anchors[best_a].width));
                    const double target_dh = std::log(truth_box.rect.height() / static_cast<double>(anchors[best_a].height));
                    const double dx = sigmoid(out_data[x_idx]) - target_dx;
                    const double dy = sigmoid(out_data[y_idx]) - target_dy;
                    const double dw = out_data[w_idx] - target_dw;
                    const double dh = out_data[h_idx] - target_dh;

                    // Compute MSE loss
                    const double ldx = dx * dx;
                    const double ldy = dy * dy;
                    const double ldw = dw * dw;
                    const double ldh = dh * dh;
                    loss += options.lambda_bbr * scale * (ldx + ldy + ldw + ldh);

                    // Compute the gradient
                    g[x_idx] = scale * options.lambda_bbr * put_in_range(-5, 5, dx);
                    g[y_idx] = scale * options.lambda_bbr * put_in_range(-5, 5, dy);
                    g[w_idx] = scale * options.lambda_bbr * put_in_range(-5, 5, dw);
                    g[h_idx] = scale * options.lambda_bbr * put_in_range(-5, 5, dh);

                    // Compute binary cross-entropy loss
                    for (long k = 0; k < num_classes; ++k)
                    {
                        const auto c_idx = tensor_index(output_tensor, n, best_a * num_feats + 5 + k, r, c);
                        if (truth_box.label == options.labels[k])
                            binary_loss_log_and_gradient_pos(out_data[c_idx], scale * options.lambda_cls, loss, g[c_idx]);
                        else
                            binary_loss_log_and_gradient_neg(out_data[c_idx], scale * options.lambda_cls, loss, g[c_idx]);
                    }
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
                    if (overlaps_any_box_nms(final_dets, dets_accum[j]))
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
                impl::yolo_helper_impl<TAG_TYPES...>::tensor_to_grad(input_tensor, truth, sub, i, options, loss);
            }
            return loss;
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

        yolo_options options;

        inline bool overlaps_any_box_nms (
            const std::vector<yolo_rect>& boxes,
            const yolo_rect& box,
            const bool classwise = false
        ) const
        {
            for (const auto& b : boxes)
            {
                if (options.overlaps_nms(b.rect, box.rect))
                {
                    if (classwise)
                    {
                        if (b.label == box.label)
                            return true;
                    }
                    else
                    {
                        return true;
                    }
                }
            }
            return false;
        }
    };

    template <template <typename> class TAG_1, template <typename> class TAG_2, template <typename> class TAG_3, typename SUBNET>
    using loss_yolo = add_loss_layer<loss_yolo_<TAG_1, TAG_2, TAG_3>, SUBNET>;
    // clang-format on
}  // namespace dlib

#endif  // loss_yolo_h_INCLUDED
