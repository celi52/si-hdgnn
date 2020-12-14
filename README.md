# SI-HDGNN: Quantifying the Scientific Impact via Heterogeneous Dynamical Graph Neural Network

This repo provides a reference implementation of **SI-HDGNN** as described in the paper:

>  Quantifying the Scientific Impact via Heterogeneous Dynamical Graph Neural Network  
>  [Xovee Xu](https://xovee.cn), Fan Zhou, Ce Li, Goce Trajcevski, Ting Zhong, and Kunpeng Zhang 
>  Submitted for review  

## Requirements
The code was tested with `Python 3.7`, `tensorflow-gpu 2.0.1`, `torch 1.0.1` and `Cuda 10.0`. Install the dependencies via Anaconda: 

```shell
# create conda virtual environment
conda create --name hdgnn python=3.7 cudatoolkit=10.0 cudnn=7.6.5 pytorch=1.0.1 torchvision=0.2.2 -c pytorch

# activate environment
conda activate hdgnn

# install other dependencies
pip install -r requirements.txt
```

## Run the codes

See [README](./codes/README.md) in `./codes/`.

## Todos

I plan to optimize the code in the near future, sorry for the inconvenience that recent codes are hard to read or lack of annotations.

## Cite

If you find **SI-HDGNN** useful for your research, please consider citing us 😘:

    @inproceedings{xovee2020quantifying, 
      author = {Xovee Xu and Fan Zhou and Ce Li and Goce Trajcevski and Ting Zhong and Kunpeng Zhang}, 
      title = {A Heterogeneous Dynamical Graph Neural Networks Approach to Quantify Scientific Impact}, 
      booktitle = {arXiv:2003.12042}, 
      year = {2020}, 
    }
      

We also have a [survey paper](https://arxiv.org/abs/2005.11041) you might be interested:

    @article{zhou2020survey,
      author = {Fan Zhou and Xovee Xu and Goce Trajcevski and Kunpeng Zhang}, 
      title = {A Survey of Information Cascade Analysis: Models, Predictions and Recent Advances}, 
      journal = {ACM Computing Surveys (CSUR)}, 
      year = {2020},
    }
