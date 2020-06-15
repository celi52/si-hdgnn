"""
Author: Xovee Xu
"""
import pickle
import config


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

    with open(output, 'wb') as f:
        pickle.dump(paper2authors, f)

    return paper2authors

def p2v(input, output):
    paper2venue = dict()

    with open(input, 'r') as f:
        for line in f:
            p_id = int(line.split(',')[0])
            p_v = int(line.split(',')[1])
            paper2venue[p_id] = p_v

    with open(output, 'wb') as f:
        pickle.dump(paper2venue, f)

    return paper2venue


with open(config.cited_dict, 'rb') as f:
    cited = pickle.load(f)

paper2authors = p2a(config.p_a_lst, config.p2a)

paper2venue = p2v(config.p_v_lst, config.p2v)

x_ids = dict()

err = 0

for p_id, p_cited in cited.items():
    try:
        x_ids[p_id] = [(p_c, paper2venue[p_c], paper2authors[p_c]) for p_c in p_cited]
    except KeyError:
        err += 1

print('# KeyErrors:', err)

max_seq = 0
for v in x_ids.values():
    if len(v) > max_seq:
        max_seq = len(v)

print('Max # sequence:', max_seq)

with open(config.x_ids, 'wb') as f:
    print('# samples:', len(x_ids))
    pickle.dump(x_ids, f)
