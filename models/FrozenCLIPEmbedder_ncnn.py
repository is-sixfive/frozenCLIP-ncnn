import numpy as np
import ncnn
import torch

def test_inference():
    torch.manual_seed(0)
    in0 = torch.full((1, 77), 10, dtype=torch.long)
    in0 = torch.tensor(
    [
49406, 3306, 267, 1002, 256, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407],dtype=torch.long)



    with ncnn.Net() as net:
         net.load_param("./FrozenCLIPEmbedder.ncnn.param")
         net.load_model("./FrozenCLIPEmbedder.ncnn.bin")

         with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(in0.numpy()).clone())

            _, out0 = ex.extract("2")
            out=torch.from_numpy(np.array(out0)).unsqueeze(0)
            
            print(out)

    if len(out) == 1:
        return out[0]
    else:
        return tuple(out)
        
test_inference()
