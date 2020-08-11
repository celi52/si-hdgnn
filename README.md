# HDGNN: Quantifying the Scientific Impact via Heterogeneous Dynamical Graph Neural Network

This repo provides a reference implementation of **HDGNN** as described in the paper:

>  Quantifying the Scientific Impact via Heterogeneous Dynamical Graph Neural Network.  
>  [Xovee Xu](xovee.cn), Fan Zhou, Ce Li, Goce Trajcevski, Ting Zhong and Guanyu Zhu.  
>  Submitted to GLOBECOM 2020 SAC BD, under review

## Requirements
The code was tested with `Python 3.7`, `tensorflow-gpu 2.0.0`, `torch 1.0.1` and `Cuda 10.0`. Install the dependencies via Anaconda: 

```shell script
pip install -r requirements.txt

conda install pytorch==1.0.1 torchvision==0.2.2 cudatoolkit=10.0 -c pytorch
```

## Run the codes

See [README](./codes/README.md) in `./codes/`.

## Todos

I plan to optimize the code in the near future, sorry for the inconvenience that recent codes are hard to read or lack of annotations.

## Cite

If you find **HDGNN** useful for your research, please consider citing us 😘:

    @inproceedings{xovee2020quantifying, 
      author = {Xovee Xu and Fan Zhou and Ce Li and Goce Trajcevski and Ting Zhong and Guanyu Zhu}, 
      title = {Quantifying the Scientific Impact via Heterogeneous Dynamical Graph Neural Network}, 
      booktitle = {arXiv:2003.12042}, 
      year = {2020}, 
    }
      

We also have a [survey paper](https://arxiv.org/abs/2005.11041) you might be interested:

    @article{zhou2020survey,
      author = {Fan Zhou and Xovee Xu and Goce Trajcevski and Kunpeng Zhang}, 
      title = {A Survey of Information Cascade Analysis: Models, Predictions and Recent Advances}, 
      journal = {arXiv:2005.11041}, 
      year = {2020},
      pages = {1--41},
    }
