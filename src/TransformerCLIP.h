#ifndef TRANSFORMER_CLIP_H
#define TRANSFORMER_CLIP_H

#include "ncnn/net.h"
#include "ncnn/mat.h"

class TransformerCLIP {
    private:
        int max_length = 77;
        ncnn::Net net;
    public:
        TransformerCLIP();
        ncnn::Mat forward(std::vector<int> token_vec);
};

#endif  // TRANSFORMER_CLIP_H
