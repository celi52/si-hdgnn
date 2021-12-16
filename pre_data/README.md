### Input fiels:

1. `a_p_list_train.txt`: `author:paper1,paper2,...`, author and papers written by this author
2. `p_a_list_train.txt`: `original_paper:author1,author2,author3,...`, paper and its authors
3. `p_p_citation_list.txt`: `original_paper:paper1,paper2,...`, paper and its citation papers
4. `v_p_list_train.txt`: `venue:paper1,paper2,...`, venue and papers published on this venue
5. `p_v.txt`: `original_paper,venue` paper and its venue
6. `deepwalk_apv.emb`: each line is an embedding of a node, trained by DeepWalk
7. `het_neigh_train.txt` and `het_random_walk.txt`: sample neighbors through random walk

### Input files for paper prediction:

Here we only include the files related to the paper prediction, followed by the author prediction
1. `node_embedding.txt`: each line is an embedding of a node
2. `p2_cited_citing_lst.txt`: `original_paper:citing_paper1,citing_paper2,...`  (including 2 years of citing papers)
3. `p20_cited_citing_lst.txt`: `original_paper:num_citations`  (including 20 years of citations)

### Input files for author prediction:

1. `node_embedding.txt`: each line is an embedding of a node
2. `a2_cited_citing_lst.txt`: `original_author:citing_paper1,citing_paper2,...`  (including 2 years of citing papers)
3. `a20_cited_citing_lst.txt`: `original_author:num_citations`  (including 20 years of citations)
5. `paper_addition.pkl`: `original_author:[(publication,[citation1,citation2...])]`, paper and its publication and citation
