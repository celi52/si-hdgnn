import sys
import os
BASE_PATH = os.path.abspath(os.path.join(os.getcwd()))
sys.path.append(BASE_PATH)
import re
import pickle
import numpy as np
from itertools import *
from collections import Counter
import random
import config

from args import read_args
args = read_args()

dir_path = os.path.join(os.getcwd(), config.pre_data_path, config.gnn_train_path)
if not os.path.isdir(dir_path):
    os.mkdir(dir_path)


P_n = config.P_n
A_n = config.A_n
V_n = config.V_n
data_path = config.pre_data_path + config.base_path
in_f_d = args.in_f_d


a_p_list_train = [[] for k in range(A_n)]
p_a_list_train = [[] for k in range(P_n)]
p_p_cite_list_train = [[] for k in range(P_n)]
v_p_list_train = [[] for k in range(V_n)]

relation_f = ["a_p_list_train.txt", "p_a_list_train.txt", \
              "p_p_citation_list.txt", "v_p_list_train.txt"]

for i in range(len(relation_f)):
    f_name = relation_f[i]
    neigh_f = open(data_path + f_name, "r")
    for line in neigh_f:
        line = line.strip()
        node_id = int(re.split(':', line)[0])
        neigh_list = re.split(':', line)[1]
        neigh_list_id = re.split(',', neigh_list)
        if f_name == 'a_p_list_train.txt':
            for j in range(len(neigh_list_id)):
                a_p_list_train[node_id].append('p' + str(neigh_list_id[j]))
        elif f_name == 'p_a_list_train.txt':
            for j in range(len(neigh_list_id)):
                p_a_list_train[node_id].append('a' + str(neigh_list_id[j]))
        elif f_name == 'p_p_citation_list.txt':
            for j in range(len(neigh_list_id)):
                if neigh_list_id[j]:
                    p_p_cite_list_train[node_id].append('p' + str(neigh_list_id[j]))
        else:
            for j in range(len(neigh_list_id)):
                v_p_list_train[node_id].append('p' + str(neigh_list_id[j]))
    neigh_f.close()


# p_v
p_v = [0] * P_n
p_v_f = open(data_path + 'p_v.txt', "r")
for line in p_v_f:
    line = line.strip()
    p_id = int(re.split(',', line)[0])
    v_id = int(re.split(',', line)[1])
    p_v[p_id] = v_id
p_v_f.close()

p_t_e_f = open(config.p_title_emb, 'rb')
p_title_embed = pickle.load(p_t_e_f)
p_t_e_f.close()


a_net_embed = np.zeros((A_n, in_f_d))
p_net_embed = np.zeros((P_n, in_f_d))
v_net_embed = np.zeros((V_n, in_f_d))
net_e_f = open(data_path + 'deepwalk_apv.emb', "r")

for line in islice(net_e_f, 0, None):
    line = line.strip()
    index = re.split(' ', line)[0]
    if len(index) and (index[0] == 'a' or index[0] == 'v' or index[0] == 'p'):
        embeds = np.asarray(re.split(' ', line)[1:], dtype='float32')
        if index[0] == 'a':
            a_net_embed[int(index[1:])] = embeds
        elif index[0] == 'v':
            v_net_embed[int(index[1:])] = embeds
        else:
            p_net_embed[int(index[1:])] = embeds
net_e_f.close()

p_v_net_embed = np.zeros((P_n, in_f_d))
p_v = [0] * P_n
p_v_f = open(data_path + "p_v.txt", "r")
for line in p_v_f:
    line = line.strip()
    p_id = int(re.split(',', line)[0])
    v_id = int(re.split(',', line)[1])
    p_v[p_id] = v_id
    p_v_net_embed[p_id] = v_net_embed[v_id]
p_v_f.close()

p_a_net_embed = np.zeros((P_n, in_f_d))
for i in range(P_n):
    if len(p_a_list_train[i]):
        for j in range(len(p_a_list_train[i])):
            a_id = int(p_a_list_train[i][j][1:])
            p_a_net_embed[i] = np.add(p_a_net_embed[i], a_net_embed[a_id])
        p_a_net_embed[i] = p_a_net_embed[i] / len(p_a_list_train[i])
    else:
        p_a_net_embed[i] = np.zeros(in_f_d)

p_ref_net_embed = np.zeros((P_n, in_f_d))
for i in range(P_n):
    if len(p_p_cite_list_train[i]):
        for j in range(len(p_p_cite_list_train[i])):
            if (p_p_cite_list_train[i][j][1:] != ''):
                p_id = int(p_p_cite_list_train[i][j][1:])
            p_ref_net_embed[i] = np.add(p_ref_net_embed[i], p_net_embed[p_id])
        p_ref_net_embed[i] = p_ref_net_embed[i] / len(p_p_cite_list_train[i])
    else:
        p_ref_net_embed[i] = p_net_embed[i]

a_text_embed = np.zeros((A_n + 1, in_f_d * 5))
for i in range(A_n):
    if len(a_p_list_train[i]):
        feature_temp = []
        if len(a_p_list_train[i]) >= 5:
            for j in range(5):
                feature_temp.append(p_title_embed[int(a_p_list_train[i][j][1:])])
        else:
            for j in range(len(a_p_list_train[i])):
                feature_temp.append(p_title_embed[int(a_p_list_train[i][j][1:])])
            for k in range(len(a_p_list_train[i]), 5):
                feature_temp.append(np.zeros(in_f_d))
        feature_temp = np.reshape(np.asarray(feature_temp), [1, -1])
        a_text_embed[i] = feature_temp
feature_temp = []
for k in range(5):
    feature_temp.append(np.zeros(in_f_d))
feature_temp = np.reshape(np.asarray(feature_temp), [1, -1])
a_text_embed[A_n] = feature_temp

v_text_embed = np.zeros((V_n, in_f_d * 5))
for i in range(V_n):
    if len(v_p_list_train[i]):
        feature_temp = []
        if len(v_p_list_train[i]) >= 5:
            for j in range(5):
                feature_temp.append(p_title_embed[int(v_p_list_train[i][j][1:])])
        else:
            for j in range(len(v_p_list_train[i])):
                feature_temp.append(p_title_embed[int(v_p_list_train[i][j][1:])])
            for k in range(len(v_p_list_train[i]), 5):
                feature_temp.append(p_title_embed[int(v_p_list_train[i][-1][1:])])

        feature_temp = np.reshape(np.asarray(feature_temp), [1, -1])
        v_text_embed[i] = feature_temp


a_neigh_list_train = [[[] for i in range(A_n)] for j in range(3)]
p_neigh_list_train = [[[] for i in range(P_n)] for j in range(3)]

het_neigh_train_f = open(config.het_neigh_train, "r")
for line in het_neigh_train_f:
    line = line.strip()
    node_id = re.split(' ', line)[0]
    neigh_list = re.split(' ', line)[1:]
    # neigh_list = re.split(',', neigh)
    if node_id[0] == 'a' and len(node_id) > 1:
        for j in range(len(neigh_list)):
            if neigh_list[j][0] == 'a':
                a_neigh_list_train[0][int(node_id[1:])].append(int(neigh_list[j][1:]))
            elif neigh_list[j][0] == 'p':
                a_neigh_list_train[1][int(node_id[1:])].append(int(neigh_list[j][1:]))
            elif neigh_list[j][0] == 'v':
                a_neigh_list_train[2][int(node_id[1:])].append(int(neigh_list[j][1:]))
    elif node_id[0] == 'p' and len(node_id) > 1:
        for j in range(len(neigh_list)):
            if neigh_list[j][0] == 'a':
                p_neigh_list_train[0][int(node_id[1:])].append(int(neigh_list[j][1:]))
            if neigh_list[j][0] == 'p':
                p_neigh_list_train[1][int(node_id[1:])].append(int(neigh_list[j][1:]))
            if neigh_list[j][0] == 'v':
                p_neigh_list_train[2][int(node_id[1:])].append(int(neigh_list[j][1:]))
        # if len((p_neigh_list_train[0][int(node_id[1:])])) == 0:
        #     p_neigh_list_train[0][int(node_id[1:])].append(1904247)
        if len((p_neigh_list_train[1][int(node_id[1:])])) == 0:
            p_neigh_list_train[1][int(node_id[1:])].append(node_id[1:])
    # elif node_id[0] == 'v' and len(node_id) > 1:
    #     for j in range(len(neigh_list)):
    #         if neigh_list[j][0] == 'a':
    #             v_neigh_list_train[0][int(node_id[1:])].append(int(neigh_list[j][1:]))
    #         if neigh_list[j][0] == 'p':
    #             v_neigh_list_train[1][int(node_id[1:])].append(int(neigh_list[j][1:]))
    #         if neigh_list[j][0] == 'v':
    #             v_neigh_list_train[2][int(node_id[1:])].append(int(neigh_list[j][1:]))
het_neigh_train_f.close()

a_neigh_list_train_top = [[[] for i in range(A_n)] for j in range(3)]
p_neigh_list_train_top = [[[] for i in range(P_n)] for j in range(3)]
top_k = [10, 10, 3]
for i in range(A_n):
    for j in range(3):
        a_neigh_list_train_temp = Counter(a_neigh_list_train[j][i])
        top_list = a_neigh_list_train_temp.most_common(top_k[j])
        neigh_size = 0
        if j == 0 or j == 1:
            neigh_size = 10
        else:
            neigh_size = 3
        for k in range(len(top_list)):
            a_neigh_list_train_top[j][i].append(int(top_list[k][0]))
        if len(a_neigh_list_train_top[j][i]) and len(a_neigh_list_train_top[j][i]) < neigh_size:
            for l in range(len(a_neigh_list_train_top[j][i]), neigh_size):
                a_neigh_list_train_top[j][i].append(random.choice(a_neigh_list_train_top[j][i]))

for i in range(P_n):
    for j in range(3):
        p_neigh_list_train_temp = Counter(p_neigh_list_train[j][i])
        top_list = p_neigh_list_train_temp.most_common(top_k[j])
        neigh_size = 0
        if j == 0 or j == 1:
            neigh_size = 10
        else:
            neigh_size = 3
        for k in range(len(top_list)):
            p_neigh_list_train_top[j][i].append(int(top_list[k][0]))
        if len(p_neigh_list_train_top[j][i]) and len(p_neigh_list_train_top[j][i]) < neigh_size:
            for l in range(len(p_neigh_list_train_top[j][i]), neigh_size):
                p_neigh_list_train_top[j][i].append(random.choice(p_neigh_list_train_top[j][i]))

# venue
from collections import Counter

add_paper_num = 46
add_venue_num = 100-46*2
top_p_a_num = 10
top_v_num = 3

cited_citing_num_dict = {}
for i in range(args.P_n):
    cited_citing_num_dict[i] = 0


for i in range(len(p_p_cite_list_train)):
    citing_id = i
    cited_id_lst = p_p_cite_list_train[citing_id]
    for cited_id in cited_id_lst:
        cited_id = int(cited_id[1:])
        cited_citing_num_dict[cited_id] = cited_citing_num_dict[cited_id] + 1


def get_p_a_for_v(v_id):
    papers = v_p_list_train[v_id]
    paper_citation_num_dict = {}
    neigh_paper_lst = []  # top cited papers
    top_cited_papers_author_lst = []  # top cited authors

    for paper_id in papers:
        paper_id = int(paper_id[1:])
        paper_citation_num_dict[paper_id] = cited_citing_num_dict[paper_id]

    for k, v in sorted(paper_citation_num_dict.items(), key=lambda item: item[1], reverse=True)[:add_paper_num]:
        neigh_paper_lst.append(k)
    for j in neigh_paper_lst:
        top_cited_papers_author_lst.append(int(p_a_list_train[j][0][1:]))

    return neigh_paper_lst, top_cited_papers_author_lst

v_neigh_list_train = [[[] for i in range(V_n)] for j in range(3)]
v_neigh_list_train_top = [[[] for i in range(V_n)] for j in range(3)]

for i in range(V_n):
    add_paper_lst, add_author_lst = get_p_a_for_v(i)
    # how to find the v
    add_venue_lst = []
    for p_id in add_paper_lst: # 2 for venue
        add_venue_lst = add_venue_lst + p_neigh_list_train_top[2][p_id]
    for a_id in add_author_lst:
        add_venue_lst = add_venue_lst + a_neigh_list_train_top[2][a_id]

    add_venue_temp = Counter(add_venue_lst)
    top_venue_list = add_venue_temp.most_common(add_venue_num)

    if len(add_author_lst) == 0:
        print('author_lst=0')
    if len(add_paper_lst) == 0:
        print('paper_lst=0')
    if len(top_venue_list) == 0:
        print('venue_lst=0')

    if len(top_venue_list) < add_venue_num:
        for _ in range(len(top_venue_list), add_venue_num):
            top_venue_list.append(random.choice(top_venue_list))

    for add_p_id in add_paper_lst:
        v_neigh_list_train[1][i].append(add_p_id)

    for add_a_id in add_author_lst:
        v_neigh_list_train[0][i].append(add_a_id)

    for add_v_id in top_venue_list:
        v_neigh_list_train[2][i].append(add_v_id[0])

    for add_p_id in add_paper_lst[:top_p_a_num]:
        v_neigh_list_train_top[1][i].append(add_p_id)

    if len(add_paper_lst) < top_p_a_num:
        for _ in range(len(add_paper_lst), top_p_a_num):
            v_neigh_list_train_top[1][i].append(random.choice(v_neigh_list_train_top[1][i]))

    for add_a_id in add_author_lst[:top_p_a_num]:
        v_neigh_list_train_top[0][i].append(add_a_id)

    if len(add_author_lst) < top_p_a_num:
        for _ in range(len(add_paper_lst), top_p_a_num):
            v_neigh_list_train_top[0][i].append(random.choice(v_neigh_list_train_top[0][i]))

    for add_v_id in top_venue_list[:top_v_num]:
        v_neigh_list_train_top[2][i].append(add_v_id[0])

    if len(v_neigh_list_train_top[0][i]) != top_p_a_num or len(v_neigh_list_train_top[1][i]) != top_p_a_num \
            or len(v_neigh_list_train_top[2][i]) != top_v_num:
        print('wrong p_a_v length')


train_id_list = [[] for i in range(3)]
for i in range(3):
    if i == 0:
        for l in range(A_n):
            if len(a_neigh_list_train_top[i][l]):
                train_id_list[i].append(l)
        a_train_id_list = np.array(train_id_list[i])
    elif i == 1:
        for l in range(P_n):
            if len(p_neigh_list_train_top[i][l]):
                train_id_list[i].append(l)
        p_train_id_list = np.array(train_id_list[i])
    else:
        for l in range(V_n):
            if len(v_neigh_list_train_top[i][l]):
                train_id_list[i].append(l)
        v_train_id_list = np.array(train_id_list[i])

a_net_embed = a_net_embed.astype('float32')
v_net_embed = v_net_embed.astype('float32')
p_net_embed = p_net_embed.astype('float32')

a_text_embed = a_text_embed.astype('float32')
v_text_embed = v_text_embed.astype('float32')
p_a_net_embed = p_a_net_embed.astype('float32')
p_ref_net_embed = p_ref_net_embed.astype('float32')
p_title_embed = p_title_embed.astype('float32')
p_v_net_embed = p_title_embed.astype('float32')

with open(config.gnn_pre_file, 'wb') as fs:
    pickle.dump((a_p_list_train, p_a_list_train, p_p_cite_list_train, v_p_list_train,\
                 a_neigh_list_train_top, p_neigh_list_train_top, v_neigh_list_train_top,\
                 p_title_embed, p_v_net_embed, p_a_net_embed, p_ref_net_embed, p_net_embed,\
                 a_net_embed, a_text_embed, v_net_embed, v_text_embed,\
                 a_train_id_list, p_train_id_list, v_train_id_list), fs, pickle.HIGHEST_PROTOCOL)