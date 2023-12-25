#include "TransformerCLIP.h"

TransformerCLIP::TransformerCLIP() {
	net.opt.num_threads = 4;
    net.load_param("../models/FrozenCLIPEmbedder.ncnn.param");
    net.load_model("../models/FrozenCLIPEmbedder.ncnn.bin");
}


ncnn::Mat TransformerCLIP::forward(std::vector<int> token_vec){

    ncnn::Mat token_mat = ncnn::Mat(max_length);
	int* token_ptr = token_mat;
    for (int i = 0; i < max_length; i++) {
        token_ptr[i] = int(token_vec[i]);
    }    

    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", token_mat);
    ncnn::Mat out0;
    ex.extract("out0", out0);

    return out0;

}
