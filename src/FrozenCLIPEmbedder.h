#ifndef FrozenCLIPEmbedder_H
#define FrozenCLIPEmbedder_H

#include "TransformerCLIP.h"
#include "clip_tokenizer.h"


class FrozenCLIPEmbedder {
    private:
        CLIPTokenizer tokenizer;
        TransformerCLIP transformerCLIP;
    public:
        FrozenCLIPEmbedder();
        ncnn::Mat forward(std::string promot);
};

#endif  // FrozenCLIPEmbedder_H