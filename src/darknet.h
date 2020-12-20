#ifndef DarkNet_H
#define DarkNet_H

#include <dlib/dnn.h>

namespace darknet
{
    // clang-format off
    using namespace dlib;


    // backbone tags
    template <typename SUBNET> using btag8 = add_tag_layer<8008, SUBNET>;
    template <typename SUBNET> using btag16 = add_tag_layer<8016, SUBNET>;
    template <typename SUBNET> using bskip8 = add_skip_layer<btag8, SUBNET>;
    template <typename SUBNET> using bskip16 = add_skip_layer<btag16, SUBNET>;
    // neck tags
    template <typename SUBNET> using ntag8 = add_tag_layer<6008, SUBNET>;
    template <typename SUBNET> using ntag16 = add_tag_layer<6016, SUBNET>;
    template <typename SUBNET> using ntag32 = add_tag_layer<6032, SUBNET>;
    template <typename SUBNET> using nskip8 = add_skip_layer<ntag8, SUBNET>;
    template <typename SUBNET> using nskip16 = add_skip_layer<ntag16, SUBNET>;
    template <typename SUBNET> using nskip32 = add_skip_layer<ntag32, SUBNET>;
    // head tags
    template <typename SUBNET> using htag8 = add_tag_layer<7008, SUBNET>;
    template <typename SUBNET> using htag16 = add_tag_layer<7016, SUBNET>;
    template <typename SUBNET> using htag32 = add_tag_layer<7032, SUBNET>;
    template <typename SUBNET> using hskip8 = add_skip_layer<htag8, SUBNET>;
    template <typename SUBNET> using hskip16 = add_skip_layer<htag16, SUBNET>;
    template <typename SUBNET> using hskip32 = add_skip_layer<htag32, SUBNET>;
    // yolo tags
    template <typename SUBNET> using ytag8 = add_tag_layer<4008, SUBNET>;
    template <typename SUBNET> using ytag16 = add_tag_layer<4016, SUBNET>;
    template <typename SUBNET> using ytag32 = add_tag_layer<4032, SUBNET>;
    template <typename SUBNET> using yskip8 = add_skip_layer<ytag8, SUBNET>;
    template <typename SUBNET> using yskip32 = add_skip_layer<ytag32, SUBNET>;
    template <typename SUBNET> using yskip32 = add_skip_layer<ytag32, SUBNET>;

    template <template <typename> class ACT, template <typename> class BN>
    struct def
    {
        template <long nf, long ks, int s, typename SUBNET>
        using conblock = ACT<BN<add_layer<con_<nf, ks, ks, s, s, ks/2, ks/2>, SUBNET>>>;

        template <long nf1, long nf2, typename SUBNET>
        using residual = add_prev1<
                         conblock<nf1, 3, 1,
                         conblock<nf2, 1, 1,
                         tag1<SUBNET>>>>;

        template <long nf, typename SUBNET>
        using conblock5 = conblock<nf, 1, 1,
                          conblock<nf * 2, 3, 1,
                          conblock<nf, 1, 1,
                          conblock<nf * 2, 3, 1,
                          conblock<nf, 1, 1, SUBNET>>>>>;

        template <long nf, typename SUBNET> using resv3 = residual<nf, nf / 2, SUBNET>;
        template <long nf, typename SUBNET> using resv4 = residual<nf, nf, SUBNET>;

        template <typename SUBNET> using resv3_64= resv3<64, SUBNET>;
        template <typename SUBNET> using resv3_128 = resv3<128, SUBNET>;
        template <typename SUBNET> using resv3_256 = resv3<256, SUBNET>;
        template <typename SUBNET> using resv3_512 = resv3<512, SUBNET>;
        template <typename SUBNET> using resv3_1024 = resv3<1024, SUBNET>;

        template <typename INPUT>
        using backbone53 = repeat<4, resv3_1024, conblock<1024, 3, 2,
                    btag16<repeat<8, resv3_512,  conblock<512, 3, 2,
                     btag8<repeat<8, resv3_256,  conblock<256, 3, 2,
                           repeat<2, resv3_128,  conblock<128, 3, 2,
                                     resv3_64<   conblock<64, 3, 2,
                                                 conblock<32, 3, 1,
                                                 INPUT>>>>>>>>>>>>>;

       template <typename SUBNET> using resv4_64= resv4<64, SUBNET>;
       template <typename SUBNET> using resv4_128 = resv4<128, SUBNET>;
       template <typename SUBNET> using resv4_256 = resv4<256, SUBNET>;
       template <typename SUBNET> using resv4_512 = resv4<512, SUBNET>;

        template <long nf, long factor, size_t N, template <typename> class RES, typename SUBNET>
        using cspblock = conblock<nf * factor, 1, 1,
                         concat2<tag1, tag2,
                         tag1<conblock<nf, 1, 1,
                         repeat<N, RES,
                         conblock<nf, 1, 1,
                         skip1<
                         tag2<conblock<nf, 1, 1,
                         tag1<conblock<nf * factor, 3, 2,
                         SUBNET>>>>>>>>>>>;

        template <typename INPUT>
        using backbone53csp = cspblock<512, 2, 4, resv4_512,
                       btag16<cspblock<256, 2, 8, resv4_256,
                        btag8<cspblock<128, 2, 8, resv4_128,
                              cspblock<64, 2, 2, resv4_64,
                              cspblock<64, 1, 1, resv3_64,
                              conblock<32, 3, 1,
                              INPUT>>>>>>>>;

        template <typename SUBNET>
        using spp = concat4<tag4, tag3, tag2, tag1, // 113
                    tag4<max_pool<13, 13, 1, 1,     // 112
                    skip1<                          // 111
                    tag3<max_pool<9, 9, 1, 1,       // 110
                    skip1<                          // 109
                    tag2<max_pool<5, 5, 1, 1,       // 108
                    tag1<conblock<512, 1, 1,   // 107
                    conblock<1024, 3, 1,       // 106
                    conblock<512, 1, 1,        // 105
                    SUBNET>>>>>>>>>>>>>;

        template <long nf, int classes, template <typename> class YTAG, template <typename> class NTAG, typename SUBNET>
        using yolo = YTAG<con<3*(classes + 5), 1, 1, 1, 1,
                     conblock<nf, 3, 1,
                     NTAG<conblock5<nf / 2,
                     SUBNET>>>>>;

        template <long nf, int classes, template <typename> class YTAG, template <typename> class NTAG, typename SUBNET>
        using yolo_sam = YTAG<con<3*(classes + 5), 1, 1, 1, 1,
                         conblock<nf, 3, 1,
                         NTAG<conblock<nf/2, 1, 1,
                         mult_prev1<
                         sig<bn_con<con<nf, 1, 1, 1, 1,
                         tag1<conblock<nf, 3, 1,
                         conblock<nf/2, 1, 1,
                         conblock<nf,   3, 1,
                         conblock<nf/2, 1, 1,
                         SUBNET>>>>>>>>>>>>>>;

        template <int classes>
        using yolov3 = yolo<256, classes, ytag8, ntag8,
                       concat2<htag8, btag8,
                       htag8<upsample<2, conblock<128, 1, 1,
                       nskip16<
                       yolo<512, classes, ytag16, ntag16,
                       concat2<htag16, btag16,
                       htag16<upsample<2, conblock<256, 1, 1,
                       nskip32<
                       yolo<1024, classes, ytag32, ntag32,
                       backbone53<tag1<input_rgb_image>>
                       >>>>>>>>>>>>>;

        template <int classes, typename SUBNET>
        using yolov4 = yolo<1024, classes, ytag32, ntag32,  // 161
                       concat2<htag32, ntag32,              // 153
                       htag32<conblock<512, 3, 2,           // 152
                       nskip16<                             // 151
                       yolo<512, classes, ytag16, ntag16,   // 150
                       concat2<htag16, ntag16,              // 142
                       htag16<conblock<256, 3, 2,           // 141
                       nskip8<                              // 140
                       yolo<256, classes, ytag8, ntag8,     // 139
                       concat2<tag1, tag2,                  // 131
                       tag1<conblock<128, 1, 1,             // 130
                       bskip8<                              // 129
                       tag2<upsample<2,                     // 128
                       conblock<128, 1, 1,                  // 127
                       ntag16<conblock5<256,                // 126
                       concat2<tag1, tag2,                  // 121
                       tag1<conblock<256, 1, 1,             // 120
                       bskip16<                             // 119
                       tag2<upsample<2,                     // 118
                       conblock<256, 1, 1,                  // 117
                       ntag32<conblock<512, 1, 1,           // 116
                       conblock<1024, 3, 1,                 // 115
                       conblock<512, 1, 1,                  // 114
                       spp<SUBNET>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>;

        template <int classes, typename SUBNET>
        using yolov4_sam = yolo_sam<1024, classes, ytag32, ntag32,  // 161
                           concat2<htag32, ntag32,                  // 153
                           htag32<conblock<512, 3, 2,               // 152
                           nskip16<                                 // 151
                           yolo_sam<512, classes, ytag16, ntag16,   // 150
                           concat2<htag16, ntag16,                  // 142
                           htag16<conblock<256, 3, 2,               // 141
                           nskip8<                                  // 140
                           yolo_sam<256, classes, ytag8, ntag8,     // 139
                           concat2<tag1, tag2,                      // 131
                           tag1<conblock<128, 1, 1,                 // 130
                           bskip8<                                  // 129
                           tag2<upsample<2,                         // 128
                           conblock<128, 1, 1,                      // 127
                           ntag16<conblock5<256,                    // 126
                           concat2<tag1, tag2,                      // 121
                           tag1<conblock<256, 1, 1,                 // 120
                           bskip16<                                 // 119
                           tag2<upsample<2,                         // 118
                           conblock<256, 1, 1,                      // 117
                           ntag32<conblock<512, 1, 1,               // 116
                           conblock<1024, 3, 1,                     // 115
                           conblock<512, 1, 1,                      // 114
                           spp<SUBNET>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>;

    };

    using yolov3_train = def<leaky_relu, bn_con>::yolov3<80>;
    using yolov3_infer = def<leaky_relu, affine>::yolov3<80>;

    using yolov4_train = def<leaky_relu, bn_con>::yolov4<80, def<mish, bn_con>::backbone53csp<tag1<input_rgb_image>>>;
    using yolov4_infer = def<leaky_relu, affine>::yolov4<80, def<mish, affine>::backbone53csp<tag1<input_rgb_image>>>;

    using yolov4_sam_mish_train = def<mish, bn_con>::yolov4_sam<80, def<mish, bn_con>::backbone53csp<tag1<input_rgb_image>>>;
    using yolov4_sam_mish_infer = def<mish, affine>::yolov4_sam<80, def<mish, affine>::backbone53csp<tag1<input_rgb_image>>>;

    // clang-format on

    template <typename net_type> void setup_detector(net_type& net, int num_classes = 80, size_t img_size = 416)
    {
        // remove bias
        disable_duplicative_biases(net);
        // remove mean from input image
        visit_layers_backwards(net, [](size_t, input_rgb_image& l) {
            l = input_rgb_image(0, 0, 0);
        });
        // setup leaky relus
        visit_computational_layers(net, [](leaky_relu_& l) { l = leaky_relu_(0.1); });
        // set the number of filters
        layer<ytag8, 1>(net).layer_details().set_num_filters(3 * (num_classes + 5));
        layer<ytag16, 1>(net).layer_details().set_num_filters(3 * (num_classes + 5));
        layer<ytag32, 1>(net).layer_details().set_num_filters(3 * (num_classes + 5));
        // allocate the network
        matrix<rgb_pixel> image(img_size, img_size);
        net(image);
    }

    template <typename net_type> void setup_classifier(net_type& net, int num_classes = 1000, size_t img_size = 416)
    {
        // remove bias
        disable_duplicative_bias(net);
        // remove mean from input image
        visit_layers_backwards(net, [](size_t, input_rgb_image& l) {
            l = input_rgb_image(0, 0, 0);
        });
        // setup leaky relus
        visit_computational_layers(net, [](leaky_relu_& l) { l = leaky_relu_(0.1); });
        // set the number of filters
        layer<1>(net).layer_details().set_num_outputs(num_classes);
        // allocate the network
        matrix<rgb_pixel> image(img_size, img_size);
        net(image);
    }

}  // namespace darknet

#endif  // DarkNet_H
