"""
Author: Xovee Xu
Load node embeddings and save them into Python dictionaries.
"""
import numpy as np
import pickle
import config

a_emb = dict()
p_emb = dict()
v_emb = dict()

with open(config.node_emb, 'r') as f:
    for line in f:
        item = line.strip().split(' ')[0]
        emb = line.split(' ')[1:]
        if item[:1] == 'a':
            a_emb[int(item[1:])] = np.array(emb, dtype='float32')
        elif item[:1] == 'p':
            p_emb[int(item[1:])] = np.array(emb, dtype='float32')
        elif item[:1] == 'v':
            v_emb[int(item[1:])] = np.array(emb, dtype='float32')

with open(config.emb_dict, 'wb') as f:
    pickle.dump((a_emb, p_emb, v_emb), f)
