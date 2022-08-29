# **Unifying** **Gradients** to Improve Real-world Robustness for Deep Networks

This repository contains a Pytorch implementation of our paper titled [Unifying Gradients to Improve Real-world Robustness for Deep Networks](https://arxiv.org/abs/2208.06228).

## Environment

Using anaconda and install packages with requirements.txt.

```python-repl
$ pip install -r envs/requirements.txt
```

## Data Prepare

The first time you run this repository, you need to extract subsets of CIFAR10 and ImageNet using function select_cifar10_100 and select_imagenet_1000, see main.py. A slight modification on ImageNet dataloader is required: annotate subdataloader part if you do not extract it before.

## Test

Five score-base query attacks, five defense methods can be tested.

```python
python main.py --dataset cifar10 --batch_size 256 --gpu 0 --max_query 100 --p 0.05 --model_type UniG --delta 0.5 --epochs_ 1 --lr_ 10 --eval
python main.py --dataset imagenet --batch_size 32 --gpu 0 --max_query 100 --p 0.8 --model_type UniG --delta 0.1 --epochs_ 1 --lr_ 1 --eval
# model_type：vanilla,AT,RND,DENT,PNI,UniG
```

## Results

Hereby, we choose [Square attack](https://github.com/max-andr/square-attack), PreResNet18, CIFAR10 as a simple example. The attack setting is: eps=8/255, query=100/2500, p=0.05. More results on different attacks and datasets see [paper](https://arxiv.org/abs/2208.06228).

| Defense | Clean Acc       | Logit-diff     | Robust Acc            |
| ------- | --------------- | -------------- | --------------------- |
| vanilla | 94.26           | -              | 38.79/0.46            |
| AT      | 87.35           | 7.00           | 79.15/67.34           |
| RND     | 91.14           | 1.53           | 65.04/51.22           |
| PNI     | 85.93           | 8.16           | 64.66/51.54           |
| DENT    | 94.25           | 7.35           | 81.78/57.71           |
| UniG    | **94.26** | **1.09** | **81.90/77.80** |

## Contact

If you have any problem with this code, please contact me directly. Email address: yingwen_wu@sjtu.edu.cn
