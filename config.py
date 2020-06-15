"""
Paper  : Quantifying the Scientific Impact via Heterogeneous Dynamical Graph Neural Network
Model  : HDGNN
Author : Xovee Xu
Date   : Year 2020
Venue  : Submitted to GLOBECOM 2020 SAC BD
"""


# file paths
dataset = '../../dataset/'

p_p_citation_lst = dataset + 'p_p_citation_list.txt'
graph = dataset + 'graph.pkl'
het_neigh_train = dataset + 'het_neigh_train.txt'
node_emb = dataset + 'node_embedding.txt'
emb_dict = dataset + 'emb_dict.pkl'
cited_citing_lst = dataset + 'p2_cited_citing_list.txt'  # for papers or authors
all_cited_citing_lst = dataset + 'p20_cited_citing_list.txt' # for papers or authors
cited_dict = dataset + 'p2_cited_dict.pkl'
p_a_lst = dataset + 'p_a_list_train.txt'
p2a = dataset + 'p2a.pkl'
p_v_lst = dataset + 'p_v.txt'
p2v = dataset + 'p2v.pkl'
x_ids = dataset + 'x_ids.pkl'
x_idx = dataset + 'x_idx.pkl'
x = dataset + 'x.pkl'
x_authors = dataset + 'x_authors.pkl'
y = dataset + 'y.pkl'


# hyper-settings
author_length = 6
seq_length = 100
