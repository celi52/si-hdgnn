import sys
import os
BASE_PATH = os.path.abspath(os.path.join(os.getcwd()))
sys.path.append(BASE_PATH)
import pickle
import math
import numpy as np
import config

# input graph
with open(config.graph, 'rb') as f:
    g = pickle.load(f)

P_n = config.P_n
A_n = config.A_n
V_n = config.V_n

a_neigh = [[] for k in range(A_n)]
p_neigh = [[] for k in range(P_n)]
v_neigh = [[] for k in range(V_n)]

a_neigh_w = [[] for k in range(A_n)]
p_neigh_w = [[] for k in range(P_n)]
v_neigh_w = [[] for k in range(V_n)]

a_percent = [0.65, 0.05, 0, 0.15, 0.15]
# probability
# a_percent[0]: author write paper
# a_percent[1]: author cite paper
# a_percent[3]: author cooperate with author
# a_percent[4]: author publish in venue
for num in range(A_n):
    nod = 'a' + str(num)
    a_p_write_list = []
    a_p_write_list_w = []

    a_p_cite_list = []
    a_p_cite_list_w = []

    a_a_cooperate_list = []
    a_a_cooperate_list_w = []

    a_v_write_list = []
    a_v_write_list_w = []

    all_node_list = []
    all_weight_list = []

    w_a_p_write = 0
    w_a_p_cite = 0
    w_a_a_cooperate = 0
    w_a_v_write = 0

    # author node
    for i in g[nod]:  # i:neighbor
        for j in g[nod][i]:  # j:key relation
            if int(j) == 0:
                a_p_write_list.append(i)
                a_p_write_list_w.append(g[nod][i][0]['w'] * math.log(g.in_degree(i)+1))

            elif int(j) == 1:
                a_p_cite_list.append(i)
                a_p_cite_list_w.append(g[nod][i][1]['w'] * math.log(g.in_degree(i)+1))

            elif int(j) == 3:
                a_a_cooperate_list.append(i)
                a_a_cooperate_list_w.append(g[nod][i][3]['w'] + math.log(g.in_degree(i)+1))

            elif int(j) == 4:
                a_v_write_list.append(i)
                a_v_write_list_w.append(g[nod][i][4]['w'] * math.log(g.in_degree(i)+1))
    w_a_p_write = sum(a_p_write_list_w)
    w_a_p_cite = sum(a_p_cite_list_w)
    w_a_a_cooperate = sum(a_a_cooperate_list_w)
    w_a_v_write = sum(a_v_write_list_w)

    for k in range(len(a_p_write_list)):
        all_node_list.append(a_p_write_list[k])
        all_weight_list.append(float(a_p_write_list_w[k]) / w_a_p_write * a_percent[0])
    for k in range(len(a_p_cite_list)):
        all_node_list.append(a_p_cite_list[k])
        all_weight_list.append(float(a_p_cite_list_w[k]) / w_a_p_cite * a_percent[1])
    for k in range(len(a_a_cooperate_list)):
        all_node_list.append(a_a_cooperate_list[k])
        all_weight_list.append(float(a_a_cooperate_list_w[k]) / w_a_a_cooperate * a_percent[3])
    for k in range(len(a_v_write_list)):
        all_node_list.append(a_v_write_list[k])
        all_weight_list.append(float(a_v_write_list_w[k]) / w_a_v_write * a_percent[4])

    a_neigh[num] = all_node_list
    a_neigh_w[num] = all_weight_list

    if num % 10000 == 0:
        print(num, A_n)

save_a_neigh = np.array(a_neigh, dtype="object")
save_a_neigh_w = np.array(a_neigh_w, dtype="object")
with open(config.a_neigh, 'wb') as f:
    np.save(f, (save_a_neigh, save_a_neigh_w))


p_percent = [0.4, 0.025, 0.2, 0.025, 0.2, 0.15]
# probability
# p_percent[0]: paper written by author
# p_percent[1]: paper cite paper
# p_percent[2]: paper cited by paper
# p_percent[3]: paper cite author
# p_percent[4]: paper cited by author
# p_percent[5]: paper publish in venue
for num in range(P_n):

    nod = 'p' + str(num)

    p_a_writed_list = []
    p_a_writed_list_w = []

    p_p_cite_list = []
    p_p_cite_list_w = []

    p_p_cited_list = []
    p_p_cited_list_w = []

    p_a_cite_list = []
    p_a_cite_list_w = []

    p_a_cited_list = []
    p_a_cited_list_w = []

    p_v_published_list = []
    p_v_published_list_w = []

    all_node_list = []
    all_weight_list = []

    w_p_a_writed = 0
    w_p_p_cite = 0
    w_p_p_cited = 0
    w_p_a_cite = 0
    w_p_a_cited = 0
    w_p_v_published = 0

    # author node
    for i in g[nod]:  # i:neighbor
        for j in g[nod][i]:  # j:key relation
            if int(j) == 5:
                p_a_writed_list.append(i)
                p_a_writed_list_w.append(g[nod][i][5]['w'] * math.log(g.in_degree(i))+1)
            elif int(j) == 6:
                p_p_cite_list.append(i)
                p_p_cite_list_w.append(g[nod][i][6]['w'])
            elif int(j) == 7:
                p_p_cited_list.append(i)
                p_p_cited_list_w.append(g[nod][i][7]['weight'] + g.in_degree(i)+1)
            elif int(j) == 8:
                p_a_cite_list.append(i)
                p_a_cite_list_w.append(g[nod][i][8]['w'] * math.log(g.in_degree(i))+1)
            elif int(j) == 9:
                p_a_cited_list.append(i)
                p_a_cited_list_w.append(g[nod][i][9]['w'] * g.in_degree(i))
            elif int(j) == 10:
                p_v_published_list.append(i)
                p_v_published_list_w.append(g[nod][i][10]['w'])
                w_p_v_published = w_p_v_published + int(g[nod][i][10]['w'])

    w_p_a_writed = sum(p_a_writed_list_w)
    w_p_p_cite = sum(p_p_cite_list_w)
    w_p_p_cited = sum(p_p_cited_list_w)
    w_p_a_cite = sum(p_a_cite_list_w)
    w_p_a_cited = sum(p_a_cited_list_w)

    for k in range(len(p_a_writed_list)):
        all_node_list.append(p_a_writed_list[k])
        all_weight_list.append(float(p_a_writed_list_w[k]) / w_p_a_writed * p_percent[0])
    for k in range(len(p_p_cite_list)):
        all_node_list.append(p_p_cite_list[k])
        all_weight_list.append(float(p_p_cite_list_w[k]) / w_p_p_cite * p_percent[1])
    for k in range(len(p_p_cited_list)):
        all_node_list.append(p_p_cited_list[k])
        all_weight_list.append(float(p_p_cited_list_w[k]) / w_p_p_cited * p_percent[2])
    for k in range(len(p_a_cite_list)):
        all_node_list.append(p_a_cite_list[k])
        all_weight_list.append(float(p_a_cite_list_w[k]) / w_p_a_cite * p_percent[3])
    for k in range(len(p_a_cited_list)):
        all_node_list.append(p_a_cited_list[k])
        all_weight_list.append(float(p_a_cited_list_w[k]) / w_p_a_cited * p_percent[4])
    for k in range(len(p_v_published_list)):
        all_node_list.append(p_v_published_list[k])
        all_weight_list.append(float(p_v_published_list_w[k]) / w_p_v_published * p_percent[5])

    p_neigh[num] = all_node_list
    p_neigh_w[num] = all_weight_list
    if num % 10000 == 0:
        print(num, P_n)

save_p_neigh = np.array(p_neigh, dtype="object")
save_p_neigh_w = np.array(p_neigh_w, dtype="object")
with open(config.p_neigh, 'wb') as f:
    np.save(f, (save_p_neigh, save_p_neigh_w))
