import sys
import os
BASE_PATH = os.path.abspath(os.path.join(os.getcwd()))
sys.path.append(BASE_PATH)
import random
import numpy as np
import config

np_load_old = np.load
# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

P_n = config.P_n
A_n = config.A_n
V_n = config.V_n

a_neigh_list_train = [[] for k in range(A_n)]
p_neigh_list_train = [[] for k in range(P_n)]
v_neigh_list_train = [[] for k in range(V_n)]
node_n = [A_n, P_n, V_n]

# load saved rwr files
with open(config.a_neigh, 'rb') as f:
    a_neigh, a_neigh_w = np.load(f)
with open(config.p_neigh, 'rb') as f:
    p_neigh, p_neigh_w = np.load(f)

len_limit ={}
len_limit['a'] = 40
len_limit['p'] = 45
len_limit['v'] = 15

len_apv = {}
for i in range(2):
    for j in range(node_n[i]):
        if i == 0:
            neigh_train = a_neigh_list_train[j]
            curNode = "a" + str(j)
        elif i == 1:
            neigh_train = p_neigh_list_train[j]
            curNode = "p" + str(j)
        neigh_L = 0
        len_apv['a'] = 0
        len_apv['p'] = 0
        len_apv['v'] = 0
        rej_times = 0
        while neigh_L < 99:  # maximum neighbor size = 100
            if rej_times >= 150:
                    if len_apv['a'] > 0 and len_apv['p'] > 0 and len_apv['v'] > 0:
                        break
                    elif rej_times >= 300:
                        print(j)
                        break
            rand_p = random.random()  # return p
            if rand_p > 0.5:
                if curNode[0] == "a":
                    chooseNode = random.choices(a_neigh[int(curNode[1:])], a_neigh_w[int(curNode[1:])])[0]
                    if chooseNode[1:] != '':
                        if len_apv[chooseNode[0]] <= len_limit[chooseNode[0]]:
                            neigh_train.append(chooseNode)
                            curNode = chooseNode
                            len_apv[curNode[0]] += 1
                            neigh_L += 1
                        else:
                            rej_times += 1
                            continue
                    else:
                        continue

                elif curNode[0] == "p":
                    chooseNode = random.choices(p_neigh[int(curNode[1:])], p_neigh_w[int(curNode[1:])])[0]
                    if chooseNode[1:] != '':
                        if len_apv[chooseNode[0]] <= len_limit[chooseNode[0]]:
                            neigh_train.append(chooseNode)
                            curNode = chooseNode
                            len_apv[curNode[0]] += 1
                            neigh_L += 1
                        else:
                            rej_times += 1
                            continue
                    else:
                        continue

                elif curNode[0] == "v":
                    if i == 0 :
                        curNode = ('a' + str(j))
                    elif i ==1 :
                        curNode = ('p' + str(j))

            else:
                if i == 0:
                    curNode = ('a' + str(j))
                elif i == 1:
                    curNode = ('p' + str(j))
        if i == 0:
            curNode = "a" + str(j)
        elif i == 1:
            curNode = "p" + str(j)

        if j % 10000 == 0:
            print(j, node_n[i])


for i in range(2):
    for j in range(node_n[i]):
        if i == 0:
            a_neigh_list_train[i] = list(a_neigh_list_train[i])
        elif i == 1:
            p_neigh_list_train[j] = list(p_neigh_list_train[j])


neigh_f = open(config.het_neigh_train, "w")
for i in range(2):
    for j in range(node_n[i]):
        if i == 0:
            neigh_train = a_neigh_list_train[j]
            curNode = "a" + str(j)
        elif i == 1:
            neigh_train = p_neigh_list_train[j]
            curNode = "p" + str(j)

        if len(neigh_train):
            neigh_f.write(curNode + " ")
            for k in range(len(neigh_train)-1):
                neigh_f.write(neigh_train[k] + " ")
            neigh_f.write(neigh_train[-1] + "\n")
neigh_f.close()