"""
Author: Xovee Xu
"""
import pickle
import config


temp_y = dict()
y = list()

with open(config.x_idx, 'rb') as f:
    x_idx = pickle.load(f)

with open(config.x_ids, 'rb') as f:
    x_ids = pickle.load(f)


print(x_idx[int(len(x_idx)*.75)])

with open(config.all_cited_citing_lst, 'r') as f:
    for line in f:
        line = line.strip()
        p_id = int(line.split(':')[0])
        temp_y[p_id] = int(line.split(':')[1])


for x_id in x_idx:
    y.append(temp_y[x_id])


with open(config.y, 'wb') as f:
    print('# samples:', len(y))
    pickle.dump(y, f)
