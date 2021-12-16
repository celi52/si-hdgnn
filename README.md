# SI-HDGNN: Heterogeneous Dynamical Academic Network for Learning Scientific Impact Propagation

This repo provides a reference implementation of **SI-HDGNN**.

> Heterogeneous Dynamical Academic Network for Learning Scientific Impact Propagation  
> [Xovee Xu](https://xovee.cn), Ting Zhong, Ce Li, Goce Trajcevski, and Fan Zhou  
> Knowledge-Based Systems, 2021, Accepted

## Requirements
The code was tested with `Python 3.7`, `tensorflow-gpu 2.4.0`, `pytorch 1.8.1`, `cudnn 8.0.4` and `cuda 11.0`. Install the dependencies via Anaconda: 

```shell
# create conda virtual environment
conda create --name si-hdgnn python=3.7 cudatoolkit=11.0.221 cudnn=8.0.4 pytorch=1.8.1 -c pytorch

# activate environment
conda activate si-hdgnn

# install other dependencies
pip install -r requirements.txt
```

## Datasets

APS and its preprocessd data can be downloaded in [Google Drive](https://drive.google.com/drive/folders/1JPXdSi23VS1lt0O_clxzNvaHgRl9iaIY?usp=sharing).

You can access the original APS dataset [here](https://journals.aps.org/datasets). (Released by *American Physical Society*, obtained at Jan 17, 2019)

Or DBLP-Citation-network V10, and ACM-Citation-network V9 [here](https://www.aminer.org/citation). (Released by *Aminer*)


# Run the code

For a given scientific dataset, you should:

1. Construct a heterogeneous graph
2. Get node embeddings
3. Generate scientific information cascades
4. Training & Evaluating

Detailed pre-process files information can be found [here](https://github.com/celi52/si-hdgnn/tree/master/pre_data).

### 1. Construct heterogeneous graph

This stage may costs a large amount of RAM (~64GB with millions of nodes/edges in graph).


```shell
# build a heterogeneous graph
python codes/gnn_pre/graph_sample.py

# heterogeneous neighboring node sampling save and run
> python codes/gnn_pre/save_rwr.py
> python codes/gnn_pre/run_rwr.py
```

### 2. Generate node embeddings

After graph construction, we now learn node embeddings via a heterogeneous graph neural network. 

```shell script
python codes/gnn_train/gene_node_embeddings.py
```

### 3. Generate scientific information cascades

Once we got the node embeddings, we can generate cascades and corresponding training/validation/test data.

#### Paper Prediction Run scripts:

```shell script
> python codes/predict_paper/1_load_emb.py
> python codes/predict_paper/2_construct_cascade.py
```

#### Author Prediction Run scripts:

```shell script
> python codes/predict_author/1_load_emb.py
> python codes/predict_author/2_x_y.py
```


### 4. Training & Evaluating SI-HDGNN

```shell script
> python codes/predict_paper/paper_prediction.py
> python codes/predict_author/author_prediction.py
```


## Options

You may change the model settings manually in `config.py` or directly into the codes. 

## Cite

If you find **SI-HDGNN** useful for your research, please consider citing us 😘 :)
```bibtex
@article{xu2021heterogeneous, 
  title = {Heterogeneous Dynamical Academic Network for Learning Scientific Impact Propagation}, 
  author = {Xovee Xu and Ting Zhong and Ce Li and Goce Trajcevski and Fan Zhou}, 
  journal = {Knowledge-Based Systems}, 
  year = {2021}, 
  numpages = {20}, 
}
```

## Contact

If you have any questions, feel free to contact us, emails: `xovee@ieee.org` or `ce.lc@outlook.com`. 
