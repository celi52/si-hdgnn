"""
Author: Xovee Xu
Construct paper sequence for papers/authors predict_paper.
"""
import sys
import os
BASE_PATH = os.path.abspath(os.path.join(os.getcwd()))
sys.path.append(BASE_PATH)
import pickle
import random
import numpy as np
import config


dataset_ids = dict()
min_citations = config.min_citations
seed = 1234

# load
with open(config.p2_cited_citing_lst, 'r') as f:
    num_data = 0
    for line in f:
        num_data += 1
        p_id = line.split(':')[0]
        p_citing = line.split(':')[1].split(',')
        if len(p_citing) < min_citations:  # filter those papers/authors which # citations are less than 'min_citations'
            continue
        p_citing = [int(xovee) for xovee in p_citing]
        dataset_ids[int(p_id)] = p_citing

l = list(dataset_ids.items())
random.seed(seed)
random.shuffle(l)
dataset_ids = dict(l)


with open(config.p2_cited_dict, 'wb') as f:
    print('Number of valid cascades: {}/{}'.format(len(dataset_ids), num_data))
    pickle.dump(dataset_ids, f)


def p2a(input, output):
    paper2authors = dict()
    with open(input, 'r') as f:
        for line in f:
            p_id = line.split(':')[0]
            p_authors = line.split(':')[1].split(',')
            p_authors = [int(xovee) for xovee in p_authors]
            paper2authors[int(p_id)] = p_authors

    max_authors = 0

    for author in paper2authors.values():
        if len(author) > max_authors:
            max_authors = len(author)

    # with open(output, 'wb') as f:
    #     pickle.dump(paper2authors, f)

    return paper2authors

def p2v(input, output):
    paper2venue = dict()

    with open(input, 'r') as f:
        for line in f:
            p_id = int(line.split(',')[0])
            p_v = int(line.split(',')[1])
            paper2venue[p_id] = p_v

    # with open(output, 'wb') as f:
    #     pickle.dump(paper2venue, f)

    return paper2venue


with open(config.p2_cited_dict, 'rb') as f:
    cited = pickle.load(f)

paper2authors = p2a(config.p_a_lst, config.p_p2a)

paper2venue = p2v(config.p_v_lst, config.p_p2v)

x_ids = dict()

err = 0

for p_id, p_citing in cited.items():
    try:
        x_ids[p_id] = [(p_c, paper2venue[p_c], paper2authors[p_c]) for p_c in p_citing]
    except KeyError:
        err += 1

print('# KeyErrors:', err)

max_seq = 0
for v in x_ids.values():
    if len(v) > max_seq:
        max_seq = len(v)

print('Max # sequence:', max_seq)
print('# samples:', len(x_ids))


x = list()
x_authors = list()
x_idx = list()

with open(config.p_emb_dict, 'rb') as f:
    a_emb, p_emb, v_emb = pickle.load(f)


for p_id, p_ids in x_ids.items():
    x_idx.append(p_id)
    temp_x = list()
    # embeddings of the original paper
    temp_x.append(np.concatenate([p_emb[p_id], v_emb[paper2venue[p_id]]]))
    # embeddings of papers who cite original paper
    for p_citing in p_ids[:config.seq_length - 1]:
        temp_x.append(np.concatenate([p_emb[p_citing[0]], v_emb[p_citing[1]]]))
    x.append(temp_x)


for p_id, p_ids in x_ids.items():
    temp_x_authors = list()
    # embeddings of the original paper
    temp_x_authors.append([a_emb[author] for author in paper2authors[p_id][:config.author_length]])
    # embeddings of papers who cite original paper
    for p_citing in p_ids[:config.seq_length - 1]:
        temp_x_authors.append([a_emb[author] for author in p_citing[2][:config.author_length]])
    x_authors.append(temp_x_authors)


with open(config.p_x, 'wb') as f:
    pickle.dump(x, f)

with open(config.p_x_authors, 'wb') as f:
    pickle.dump(x_authors, f)

temp_y = dict()
y = list()

with open(config.p20_cited_citing_lst, 'r') as f:
    for line in f:
        line = line.strip()
        p_id = int(line.split(':')[0])
        temp_y[p_id] = int(line.split(':')[1])


for x_id in x_idx:
    y.append(temp_y[x_id])


with open(config.p_y, 'wb') as f:
    print('# samples:', len(y))
    pickle.dump(y, f)
