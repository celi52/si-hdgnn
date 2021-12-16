import sys
import os
BASE_PATH = os.path.abspath(os.path.join(os.getcwd()))
sys.path.append(BASE_PATH)
import pickle
import networkx as nx
import config

dir_path = os.path.join(os.getcwd(), config.pre_data_path, config.rwr_path)
if not os.path.isdir(dir_path):
    os.mkdir(dir_path)


# a_relation = ['a_write_p', 'a_cite_p', 'a_cite_a', 'a_cooperate_a', 'a_write_v']
# p_relation = ['p_writen_by_a', 'p_cite_p', 'p_cited_by_p', 'p_cite_a', 'p_cited_by_a', 'p_published_v']
# v_relation = ['v_publish_p']

a_relation = [0, 1, 2, 3, 4]
p_relation = [5, 6, 7, 8, 9, 10]
v_relation = [11]

g = nx.MultiDiGraph()

paper_num = config.P_n
author_num = config.A_n
venue_num = config.V_n

print('Start loading data.')

print('Generate paper&paper citation dictionary!')
p_p_citation_dict = dict()
with open(config.p_p_citation_lst, 'r') as fs:
    for line in fs:
        line = line.strip()
        paper_citing = int(line.split(':')[0])
        try:
            paper_cited = [int(id) for id in line.split(':')[1].split(',')]
        except ValueError:
            paper_cited = []
        p_p_citation_dict[paper_citing] = paper_cited

print('Generate paper&author list dictionary!')
p_a_list_dict = dict()
with open(config.p_a_lst, 'r') as fs:
    for line in fs:
        line = line.strip()
        paper_id = int(line.split(':')[0])
        try:
            authors = [int(id) for id in line.split(':')[1].split(',')]
        except ValueError:
            authors = []
        p_a_list_dict[paper_id] = authors

len_p_p_citation_dict = len(p_p_citation_dict)
print('Length of p_p_citation_dict:', len_p_p_citation_dict)

print('Starting add nodes and edges to graph G!')

xovee = 0
num_a_p = 0

for paper_id, authors in p_a_list_dict.items():
    if xovee % int(paper_num / 1000) == 0:
        print('Progress {:.2f}%: {}/{}'.format((xovee/paper_num)*100, xovee, paper_num))

    paper_node = ('p' + str(paper_id))

    for paper in p_p_citation_dict[paper_id]:
        g.add_edge(paper_node, 'p'+str(paper), key=p_relation[1], w=1)
        g.add_edge('p' + str(paper), paper_node, key=p_relation[2], weight=1)

        for author in authors:
            # a_relation[1] a_cite_p
            if g.has_edge('a' + str(author), 'p' + str(paper), key=a_relation[1]):
                g['a' + str(author)]['p' + str(paper)][a_relation[1]]['w'] += 1
            else:
                g.add_edge('a' + str(author), 'p' + str(paper), key=a_relation[1], w=1)
                num_a_p += 1

            # add cited paper to author
            if g.has_edge('p' + str(paper), 'a' + str(author), key=p_relation[4]):
                g['p' + str(paper)]['a' + str(author)][p_relation[4]]['w'] += 1
            else:
                g.add_edge('p' + str(paper), 'a' + str(author), key=p_relation[4], w=1)


        for another_author in p_a_list_dict[paper]:
            # add author to author
            # if g.has_edge('a'+str(author), 'a'+str(another_author), key=a_relation[2]):
            #     g['a'+str(author)]['a'+str(another_author)][a_relation[2]]['w'] += 1
            # else:
            #     g.add_edge('a'+str(author), 'a'+str(another_author), key=a_relation[2], w=1)

            # add paper to author
            if g.has_edge(paper_node, 'a'+str(another_author), key=p_relation[3]):
                g[paper_node]['a'+str(another_author)][p_relation[3]]['w'] += 1
            else:
                g.add_edge(paper_node, 'a'+str(another_author), key=p_relation[3], w=1)

    g.add_edges_from([('a'+str(author), paper_node, a_relation[0], dict(w=1)) for author in authors])
    g.add_edges_from([(paper_node, 'a'+str(author), p_relation[0], dict(w=1)) for author in authors])

    for i in range(len(authors)):
        for j in range(len(authors)):
            if i != j:
                if g.has_edge('a'+str(authors[i]), 'a'+str(authors[j]), key=a_relation[3]):
                    g['a'+str(authors[i])]['a'+str(authors[j])][a_relation[3]]['w'] += 1
                else:
                    g.add_edge('a'+str(authors[i]), 'a'+str(authors[j]), key=a_relation[3], w=1)

    xovee+=1

print(num_a_p)

xovee = 0
with open(config.p_v_lst, 'r') as fs:
    for line in fs:
        if xovee % int(paper_num / 1000) == 0:
            print('Progress {:.2f}%: {}/{}'.format((xovee / paper_num) * 100, xovee, paper_num))
        line = line.strip()
        paper_id, venue_id = [str(i) for i in line.split(',')]
        paper_node = ('p' + paper_id)
        venue_node = ('v' + venue_id)

        g.add_edge(paper_node, venue_node, key=p_relation[5], w=1)
        g.add_edge(venue_node, paper_node, key=v_relation[0], w=1)

        if p_a_list_dict.get(int(paper_id)):
            for author_id in p_a_list_dict[int(paper_id)]:
                if g.has_edge('a' + str(author_id), venue_node, key=a_relation[4]):
                    g.add_edge('a' + str(author_id), venue_node, key=a_relation[4],
                               w=1+g['a' + str(author_id)][venue_node][a_relation[4]]['w'])
                else:
                    g.add_edge('a' + str(author_id), venue_node, key=a_relation[4], w=1)

        xovee+=1

print('Adding paper&venue nodes and edges to graph G!')
print('Number of nodes', g.number_of_nodes())
print('Number of edges', g.number_of_edges())

print('Save graph G!')

with open(config.graph, 'wb') as f:
    pickle.dump(g, f)
