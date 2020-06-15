
# Run the code

For a given scientific dataset, you should:

1. Construct a heterogeneous graph
2. Get node embeddings
3. Generate scientific information cascades
4. Training & evaluating

## Construct heterogeneous graph

This stage may costs a large amount of RAM (~64GB with millions of nodes/edges in graph), delete some nodes/edges to save space.

### Run scripts:

```shell script
> python graph_sample.py
> python rwr.py
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

### Input files:

Here we only include the files related to paper prediction, however, this can be straightforwardly extended to author prediction.
1. `node_embedding.txt`: each line is an embedding of a node
2. `p2_cited_citing_lst.txt`: `original_paper:citing_paper1,citing_paper2,...`  (including 2 years of citing papers)
3. `p20_cited_citing_lst.txt`: `original_paper:num_citations`  (including 20 years of citations)
4. `p_a_lst_train.txt`: `original_paper:author1,author2,author3,...`, paper and its authors
5. `p_v.txt`: `original_paper,venue`

### Run scripts:

```shell script
> cd ./codes/prediction
> python 1_load_emb.py
> python 2_construct_cascade.py
> python 3_x_ids.py
> python 4_x_idx.py
> python 5_y.py
```

## Training & evaluating HDGNN

```shell script
> python hdgnn.py
```


## Options

You may change the model settings manually in `config.py` or directly into the codes. 

## Datasets

Dataset can be downloaded in [Google Drive](https://drive.google.com/drive/folders/1JPXdSi23VS1lt0O_clxzNvaHgRl9iaIY?usp=sharing).

You can access the original APS dataset [here](https://journals.aps.org/datasets). (Released by *American Physical Society*, obtained at Jan 17, 2019)
