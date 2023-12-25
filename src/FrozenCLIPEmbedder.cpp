#include "FrozenCLIPEmbedder.h"


FrozenCLIPEmbedder::FrozenCLIPEmbedder(){
    tokenizer = CLIPTokenizer("vocab.txt");
}

ncnn::Mat FrozenCLIPEmbedder::forward(std::string promot){

    TokenizerResult result = tokenizer.tokenize({promot});

    // std::cout << "Tokens: " << std::endl;
    // for (auto& token : result.tokens[0]) {
    //     std::cout << token << " ";
    // }

    return transformerCLIP.forward(result.tokens[0]);



}

