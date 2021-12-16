import sys
import os
BASE_PATH = os.path.abspath(os.path.join(os.getcwd()))
sys.path.append(BASE_PATH)
import pickle
import numpy as np
import random
import config

seed = 1234
with open(config.author_Prediction_partition, 'rb') as f:
    author_paper_citing = pickle.load(f)

# load author_cited_citing_list
dataset_ids = dict()
min_citations = config.min_citations
with open(config.a2_cited_citing_lst, 'r') as f:
    num_data = 0
    for line in f:
        num_data += 1
        p_id = int(line.split(':')[0])
        p_citing = line.split(':')[1].split(',')
        if len(p_citing) < min_citations:  # filter those papers/authors which # citations are less than 'min_citations'
            continue

        p_citing = [int(xovee) for xovee in p_citing]
        dataset_ids[int(p_id)] = p_citing

# with open(config.a2_cited_dict, 'wb') as f:
#     # print('Number of valid cascades: {}/{}'.format(len(dataset_ids_new), num_data))
#     pickle.dump(dataset_ids, f)

dataset_ids_new = dict()
for (key, value) in author_paper_citing.items():
    if key in dataset_ids:
        dataset_ids_new[key] = value

l = list(dataset_ids_new.items())
random.seed(seed)
random.shuffle(l)
dataset_ids_new = dict(l)


# load p2a p2v
def p2a(input, output):
    paper2authors = dict()

    with open(input, 'r') as f:
        for line in f:
            line = line.strip()
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
            line = line.strip()
            p_id = int(line.split(',')[0])
            p_v = int(line.split(',')[1])
            paper2venue[p_id] = p_v

    # with open(output, 'wb') as f:
    #     pickle.dump(paper2venue, f)

    return paper2venue


paper2authors = p2a(config.p_a_lst, config.a_p2a)
paper2venue = p2v(config.p_v_lst, config.a_p2v)

x_ids = {}
err = 0

for author_id, author_paper in dataset_ids_new.items():
    try:
        num = 0
        x_ids[author_id] = []
        for temp in author_paper:
            x_ids[author_id].append([])
            x_ids[author_id][num].append((temp[0], paper2venue[temp[0]], paper2authors[temp[0]]))
            for temp_paper in temp[1]:
                temp_paper = int(temp_paper)
                x_ids[author_id][num].append((temp_paper, paper2venue[temp_paper], paper2authors[temp_paper]))
            num += 1
    except KeyError:
        err += 1
print('# KeyErrors:', err)


with open(config.a_x_ids, 'wb') as f:
    pickle.dump(x_ids, f)

# (2) x x_authors x_idx
# hyper_parameter
max_papers = config.max_papers

# with open(config.a_p2a, 'rb') as f:
#     paper2authors = pickle.load(f)
#
# with open(config.a_p2v, 'rb') as f:
#     paper2venue = pickle.load(f)

with open(config.a_emb_dict, 'rb') as f:
    a_emb, p_emb, v_emb = pickle.load(f)

print('ok')

x = []
x_authors = list()
x_idx = list()
for a_id, a_ids in x_ids.items():
    x_idx.append(a_id)
    temp_x_2list = []
    for writing_citing in a_ids[:config.max_papers]:
        temp_x = []
        for paper in writing_citing[:config.seq_length]:
            temp_x.append(np.concatenate([p_emb[paper[0]], v_emb[paper[1]]]))
        temp_x_2list.append(temp_x)
    x.append(temp_x_2list)

for a_id, a_ids in x_ids.items():
    temp_x_authors = []
    for writing_citing in a_ids[:config.max_papers]:
        temp_x = []
        for paper in writing_citing[:config.seq_length]:
            temp_x.append([a_emb[author] for author in paper[2][:config.author_length]])
        temp_x_authors.append(temp_x)
    x_authors.append(temp_x_authors)

with open(config.a_x, 'wb') as f:
    pickle.dump(x, f)

with open(config.a_x_idx, 'wb') as f:
    pickle.dump(x_idx, f)

with open(config.a_x_authors, 'wb') as f:
    pickle.dump(x_authors, f)

# (3) y - label
temp_y = dict()
y = list()

print(x_idx[int(len(x_idx)*.5)])
print(x_idx[int(len(x_idx)*.75)])


with open(config.a20_cited_citing_lst, 'r') as f:
    for line in f:
        line = line.strip()
        p_id = int(line.split(':')[0])
        temp_y[p_id] = int(line.split(':')[1])


for x_id in x_idx:
    y.append(temp_y[x_id])


with open(config.a_y, 'wb') as f:
    print('# samples:', len(y))
    pickle.dump(y, f)