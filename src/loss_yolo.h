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
        double conf_thresh = 0.25;
        test_box_overlap overlaps_nms = test_box_overlap(0.45);
        test_box_overlap overlaps_ignore;

    };

    inline void serialize(const yolo_options& item, std::ostream& out)
    {
        int version = 1;
        serialize(version, out);
        serialize(item.anchors, out);
        serialize(item.conf_thresh, out);
        serialize(item.overlaps_nms, out);
        serialize(item.overlaps_ignore, out);
    }

    inline void deserialize(yolo_options& item, std::istream& in)
    {
        int version = 0;
        deserialize(version, in);
        if (version != 1)
            throw serialization_error("Unexpected version found while deserializing dlib::yolo_options.");
        deserialize(item.anchors, in);
        deserialize(item.conf_thresh, in);
        deserialize(item.overlaps_nms, in);
        deserialize(item.overlaps_ignore, in);
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

            template <typename net_type>
            static void tensor_to_dets(
                const net_type& net,
                const tensor& input,
                const long i,
                const yolo_options& options,
                std::vector<yolo_rect>& dets
            )
            {
                yolo_helper_impl<TAG_TYPE>::tensor_to_dets(net, input, i, options, dets);
                yolo_helper_impl<TAG_TYPES...>::tensor_to_dets(net, input, i, options, dets);
            }
        };

        template <template <typename> class TAG_TYPE>
        struct yolo_helper_impl<TAG_TYPE>
        {
            constexpr static size_t tag_count() { return 1; }

            static void list_tags(std::ostream& out) { out << tag_id<TAG_TYPE>::id; }

            template <typename net_type>
            static void tensor_to_dets(
                const net_type& net,
                const tensor& input,
                const long i,
                const yolo_options& options,
                std::vector<yolo_rect>& dets
            )
            {
                DLIB_CASSERT(net.sample_expansion_factor() == 1, net.sample_expansion_factor());
                const tensor& t = layer<TAG_TYPE>(net).get_output();
                const size_t stride = input.nr() / t.nr();
                const auto& anchors = options.anchors.at(tag_id<TAG_TYPE>::id);
                const size_t num_attribs = t.k() / anchors.size();
                const size_t num_classes = num_attribs - 5;
                const float* const out = t.host();
                for (size_t a = 0; a < anchors.size(); ++a)
                {
                    for (long r = 0; r < t.nr(); ++r)
                    {
                        for (long c = 0; c < t.nc(); ++c)
                        {
                            const float obj = sigmoid(out[tensor_index(t, i, a * num_attribs + 4, r, c)]);
                            if (obj > options.conf_thresh)
                            {
                                const float x = (sigmoid(out[tensor_index(t, i, a * num_attribs + 0, r, c)]) + c) * stride;
                                const float y = (sigmoid(out[tensor_index(t, i, a * num_attribs + 1, r, c)]) + r) * stride;
                                const float w = std::exp(out[tensor_index(t, i, a * num_attribs + 2, r, c)]) * anchors[a].width;
                                const float h = std::exp(out[tensor_index(t, i, a * num_attribs + 3, r, c)]) * anchors[a].height;
                                yolo_rect d(centered_drect(dpoint(x, y), w, h), 0);
                                for (size_t k = 0; k < num_classes; ++k)
                                {
                                    const float p = (sigmoid(out[tensor_index(t, i, a * num_attribs + 5 + k, r, c)]));
                                    if (p > d.detection_confidence)
                                    {
                                        d.detection_confidence = p;
                                        d.label = options.labels[k];
                                    }
                                }
                                d.detection_confidence *= obj;
                                dets.push_back(std::move(d));
                            }
                        }
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
                impl::yolo_helper_impl<TAG_TYPES...>::tensor_to_dets(sub, input_tensor, i, options, dets_accum);

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
            return 0;
        }

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

    template <template <typename> class TAG_8, template <typename> class TAG_16, template <typename> class TAG_32, typename SUBNET>
    using loss_yolo = add_loss_layer<loss_yolo_<TAG_8, TAG_16, TAG_32>, SUBNET>;
    // clang-format on
}  // namespace dlib

#endif  // loss_yolo_h_INCLUDED
