
#include "clip_tokenizer.h"
#include "FrozenCLIPEmbedder.h"
#include <iostream>

void pretty_print(const ncnn::Mat& m)
{
    for (int q=0; q<m.c; q++)
    {
        const float* ptr = m.channel(q);
        for (int y=0; y<m.h; y++)
        {
            for (int x=0; x<m.w; x++)
            {
                printf("%f ", ptr[x]);
            }
            ptr += m.w;
            printf("\n");
        }
        printf("------------------------\n");
    }
}



void test_inference() {

    FrozenCLIPEmbedder frozenCLIPEmbedder;
    ncnn::Mat emb = frozenCLIPEmbedder.forward("hello, world!");
    pretty_print(emb);

}

int main() {
    test_inference();

    return 0;
}


