# frozenCLIP-ncnn



stable diffusion 中的 frozenCLIP 通过 C++与ncnn进行实现



运行：

```shell
# 运行前需编译安装ncnn
mkdir build
cd build
cmake ..
make
./clipTokenizer
```



模型转换参考：https://github.com/Tencent/ncnn/issues/5222

​						   https://github.com/Tencent/ncnn/issues/5194

clip_tokenizer C++实现参考：https://github.com/ozanarmagan/clip_tokenizer_cpp

感谢nihui up耐心解答问题

