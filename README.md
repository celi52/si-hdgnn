# SI-HDGNN: Heterogeneous Dynamical Academic Network for Learning Scientific Impact Propagation

This repo provides a reference implementation of **SI-HDGNN**.

> Heterogeneous Dynamical Academic Network for Learning Scientific Impact Propagation  
> [Xovee Xu](https://xovee.cn), Ting Zhong, Ce Li, Goce Trajcevski, and Fan Zhou  
> Knowledge-Based Systems, 2021, Accepted

## Requirements
The code was tested with `Python 3.7`, `tensorflow-gpu 2.4.0`, `torch 1.0.1` and `cuda 11.0`. Install the dependencies via Anaconda: 

```shell
# create conda virtual environment
conda create --name SIHDGNN python=3.7 cudatoolkit=11.0.221 cudnn=8.0.4 pytorch=1.0.1 torchvision=0.2.2 -c pytorch

# activate environment
conda activate SIHDGNN

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
4. Training & evaluating

## Construct heterogeneous graph

This stage may costs a large amount of RAM (~64GB with millions of nodes/edges in graph).

### Run scripts:

```shell
# build a heterogeneous graph
python graph_sample.py

# heterogeneous neighboring node sampling
python rwr.py
```

## Generate node embeddings

After graph construction, we now learn node embeddings via a heterogeneous graph neural network. 

### Input fiels:

1. `a_p_list_train.txt`: `author:paper1,paper2,...`, author and papers written by this author
2. `p_a_list_train.txt`: `original_paper:author1,author2,author3,...`, paper and its authors
3. `p_p_citation_list.txt`: `original_paper:paper1,paper2,...`, paper and its citation papers
4. `v_p_list_train.txt`: `venue:paper1,paper2,...`, venue and papers published on this venue
5. `node_net_embedding`: each line is an embedding of a node, trained by DeepWalk
6. `het_neigh_train.txt` and `het_random_walk.txt`: sample neighbors through random walk

```shell script
> cd ./codes/gnn
> python gene_node_embeddings.py
```

## Generate scientific information cascades

Once we got the node embeddings, we can generate cascades and corresponding training/validation/test data.

### Input files for paper prediction:

Here we only include the files related to the paper prediction, followed by the author prediction
1. `node_embedding.txt`: each line is an embedding of a node
2. `p2_cited_citing_lst.txt`: `original_paper:citing_paper1,citing_paper2,...`  (including 2 years of citing papers)
3. `p20_cited_citing_lst.txt`: `original_paper:num_citations`  (including 20 years of citations)
4. `p_a_lst_train.txt`: `original_paper:author1,author2,author3,...`, paper and its authors
5. `p_v.txt`: `original_paper,venue`

### Input files for author prediction:

1. `node_embedding.txt`: each line is an embedding of a node
2. `a2_cited_citing_lst.txt`: `original_author:citing_paper1,citing_paper2,...`  (including 2 years of citing papers)
3. `a20_cited_citing_lst.txt`: `original_author:num_citations`  (including 20 years of citations)
4. `p_a_lst_train.txt`: `original_paper:author1,author2,author3,...`, paper and its authors
5. `paper_addition.pkl`: `original_author:[(publication,[citation1,citation2...])]`, paper and its publication and citation

### Paper Prediction Run scripts:

```shell script
> cd ./codes/paper_prediction
> python 1_load_emb.py
> python 2_construct_cascade.py
> python 3_x_ids.py
> python 4_x_idx.py
> python 5_y.py
```

### Author Prediction Run scripts:

```shell script
> cd ./codes/paper_prediction
> python 1_load_emb.py
> python 2_x_y.py
```


## Training & evaluating SI-HDGNN

```shell script
> python paper_prediction.py
> python author_prediction.py
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
