"""
Author: Xovee Xu
"""
import pickle
import numpy as np
import config


x = list()
x_authors = list()
x_idx = list()

with open(config.p2a, 'rb') as f:
    paper2authors = pickle.load(f)

with open(config.p2v, 'rb') as f:
    paper2venue = pickle.load(f)

with open(config.x_ids, 'rb') as f:
    x_ids = pickle.load(f)

with open(config.emb_dict, 'rb') as f:
    a_emb, p_emb, v_emb = pickle.load(f)


for p_id, p_ids in x_ids.items():
    x_idx.append(p_id)
    temp_x = list()
    # embeddings of the original paper
    temp_x.append(np.concatenate([p_emb[p_id], v_emb[paper2venue[p_id]]]))
    # embeddings of papers who cite original paper
    for p_cited in p_ids[:config.seq_length-1]:
        temp_x.append(np.concatenate([p_emb[p_cited[0]], v_emb[p_cited[1]]]))
    x.append(temp_x)


for p_id, p_ids in x_ids.items():
    temp_x_authors = list()
    # embeddings of the original paper
    temp_x_authors.append([a_emb[author] for author in paper2authors[p_id][:config.author_length]])
    # embeddings of papers who cite original paper
    for p_cited in p_ids[:config.seq_length-1]:
        temp_x_authors.append([a_emb[author] for author in p_cited[2][:config.author_length]])
    x_authors.append(temp_x_authors)


with open(config.x, 'wb') as f:
    pickle.dump(x, f)

with open(config.x_idx, 'wb') as f:
    pickle.dump(x_idx, f)

with open(config.x_authors, 'wb') as f:
    pickle.dump(x_authors, f)
