"""
Author: Xovee Xu
Construct paper sequence for papers/authors paper_prediction.
"""
import pickle
import config


dataset_ids = dict()
min_citations = 10

# load
with open(config.cited_citing_lst, 'r') as f:
    num_data = 0
    for line in f:
        num_data += 1
        p_id = line.split(':')[0]
        p_cited = line.split(':')[1].split(',')
        if len(p_cited) < min_citations:  # filter those papers/authors which # citations are less than 'min_citations'
            continue
        p_cited = [int(xovee) for xovee in p_cited]
        dataset_ids[int(p_id)] = p_cited


with open(config.cited_dict, 'wb') as f:
    print('Number of valid cascades: {}/{}'.format(len(dataset_ids), num_data))
    pickle.dump(dataset_ids, f)
