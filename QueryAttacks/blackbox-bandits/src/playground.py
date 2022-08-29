import numpy as np 

data = np.load("imagenet-resnext101_denoise-None.npy")
correct = (data>0)

def asr_under_k_queries(data, k=100):
    return (data<k).sum()

for k in [100, data.max(0)]:
    print(f"queries:{k} remain:{(correct.sum()-asr_under_k_queries(data[correct], k))/len(data)}")

for k in [100, data.max(0)]:
    print(f"queries:{k} remain:{(correct.sum()-asr_under_k_queries(data[correct], k))/correct.sum()}")