#ifndef darknet_weights_visitor_h_INCLUDED
#define darknet_weights_visitor_h_INCLUDED

#include <dlib/dnn.h>

namespace darknet
{
    using namespace dlib;
    class weights_visitor
    {
        public:
        weights_visitor(const std::string& weights_path) : weights(read_bytes(weights_path))
        {
            int major = 0, minor = 0, dummy = 0;
            (*this) >> major >> minor >> dummy;
            if ((major * 10 + minor) >= 2 and major < 1000 and minor < 1000)
                (*this) >> dummy >> dummy;
            else
                (*this) >> dummy;
            std::cout << "weights file major: " << major << ", minor: " << minor
                      << ", num weights: " << weights.size() << '\n';
        }

        ~weights_visitor()
        {
            std::cout << "read " << offset << " floats of " << weights.size() << '\n';
        }

        // ignore other layers
        template <typename T> void operator()(size_t, T&) {}

        // batch normalization layers
        template <typename SUBNET> void operator()(size_t, add_layer<bn_<CONV_MODE>, SUBNET>& l)
        {
            auto& bn = l.layer_details();
            tensor& bn_t = bn.get_layer_params();
            auto bn_gamma = alias_tensor(1, l.subnet().get_output().k());
            auto bn_beta = alias_tensor(1, l.subnet().get_output().k());
            auto g = bn_gamma(bn_t, 0);
            auto b = bn_beta(bn_t, bn_gamma.size());
            DLIB_CASSERT(bn_t.size() == (g.size() + b.size()) && g.size() == b.size());

            const auto num_b = bn_t.size() / 2;
            auto temp_b = resizable_tensor(1, num_b);
            auto temp_g = resizable_tensor(1, num_b);
            auto temp_m = resizable_tensor(1, num_b);
            auto temp_v = resizable_tensor(1, num_b);

            // bn bias
            float* ptr = temp_b.host();
            for (size_t i = 0; i < num_b; ++i)
                (*this) >> ptr[i];

            // bn weights
            ptr = temp_g.host();
            for (size_t i = 0; i < num_b; ++i)
                (*this) >> ptr[i];

            // bn running mean
            ptr = temp_m.host();
            for (size_t i = 0; i < num_b; ++i)
                (*this) >> ptr[i];

            // bn running var
            ptr = temp_v.host();
            for (size_t i = 0; i < num_b; ++i)
                (*this) >> ptr[i];

            g = pointwise_divide(mat(temp_g), sqrt(mat(temp_v) + DEFAULT_BATCH_NORM_EPS));
            b = mat(temp_b) - pointwise_multiply(mat(g), mat(temp_m));

            // conv weight
            auto& conv = l.subnet().layer_details();
            auto& conv_t = conv.get_layer_params();
            DLIB_CASSERT(conv.bias_is_disabled());
            ptr = conv_t.host();
            for (size_t i = 0; i < conv_t.size(); ++i)
                (*this) >> ptr[i];
        }

        // convolutions
        template <long nf, long nr, long nc, int sy, int sx, int py, int px, typename SUBNET>
        void operator()(size_t, add_layer<con_<nf, nr, nc, sy, sx, py, px>, SUBNET>& l)
        {
            auto& conv = l.layer_details();
            if (not conv.bias_is_disabled())
            {
                tensor& con_t = conv.get_layer_params();
                auto filters = alias_tensor(
                    conv.num_filters(),
                    l.subnet().get_output().k(),
                    conv.nr(),
                    conv.nc());
                auto biases = alias_tensor(1, conv.num_filters());
                auto f = filters(con_t, 0);
                auto b = biases(con_t, filters.size());
                DLIB_CASSERT(con_t.size() == (filters.size() + biases.size()));
                DLIB_CASSERT(f.size() == filters.size());
                DLIB_CASSERT(b.size() == biases.size());

                // conv bias
                float* ptr = b.host();
                for (size_t i = 0; i < b.size(); ++i)
                    (*this) >> ptr[i];

                // conv filters
                ptr = f.host();
                for (size_t i = 0; i < f.size(); ++i)
                    (*this) >> ptr[i];
            }
        }

        private:
        const std::vector<char> weights;
        size_t offset = 0;

        std::vector<char> read_bytes(const std::string& filename)
        {
            std::ifstream file(filename, std::ios::binary);
            std::vector<char> v;
            std::copy(
                std::istreambuf_iterator<char>(file),
                std::istreambuf_iterator<char>(),
                std::back_inserter(v));
            return v;
        }

        template <typename T> weights_visitor& operator>>(T& x)
        {
            T* ptr = (T*)&weights[offset];
            x = *ptr;
            offset += sizeof(T);
            return *this;
        }
    };
}  // namespace darknet

#endif  // darknet_weights_visitor_h_INCLUDED
