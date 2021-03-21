"""
Author: Ce Li
Arguments
"""

# task to run for author_prediction
task = 'author_prediction'

# file paths
dataset = '../../APS/'
data_prediction = '../../Pre-data/'

# APS (Pretraining files)
p_p_citation_lst = dataset + 'p_p_citation_list.txt'
graph = dataset + 'graph.pkl'
het_neigh_train = dataset + 'het_neigh_train.txt'
node_emb = dataset + 'node_embedding.txt'
p_a_lst = dataset + 'p_a_list_train.txt'
p_v_lst = dataset + 'p_v.txt'

if task == 'paper':
    cited_citing_lst = dataset + 'p2_cited_citing_list.txt'  # for papers or authors
    all_cited_citing_lst = dataset + 'p20_cited_citing_list.txt'  # for papers or authors

else:
    cited_citing_lst = dataset + 'a2_cited_citing_list.txt'  # for papers or authors
    all_cited_citing_lst = dataset + 'a20_cited_citing_list.txt'  # for papers or authors

# Pre-data (predicting files)
if task == 'paper':
    cited_dict = data_prediction + task + '/p2_cited_dict.pkl'
else:
    cited_dict = data_prediction + task + '/a2_cited_dict.pkl'

emb_dict = data_prediction + task + '/emb_dict.pkl'
p2a = data_prediction + task + '/p2a.pkl'
p2v = data_prediction + task + '/p2v.pkl'
x_ids = data_prediction + task + '/x_ids.pkl'
x_idx = data_prediction + task + '/x_idx.pkl'
x = data_prediction + task + '/x.pkl'
x_authors = data_prediction + task + '/x_authors.pkl'
y = data_prediction + task + '/y.pkl'

# author paper addtion
author_paper_paperciting = data_prediction + task + '/paper_addition.pkl'

# hyper-settings
author_length = 6
seq_length = 100

# author add
max_papers = 15

