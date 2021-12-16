"""
Author: Ce Li
Tool for generator
"""
import copy
import math
import numpy as np
from tensorflow.keras import utils as np_utils


EPSILON = 1e-7


class Generator(np_utils.Sequence):

    def __init__(self, x, x_authors, y, b_size, max_papers, max_seq, max_authors):
        self.x, self.x_authors, self.y = x, x_authors, y
        self.batch_size = b_size
        self.max_papers = max_papers
        self.max_seq = max_seq
        self.max_authors = max_authors
        self.author_emb_dim = 128
        self.paper_emb_dim = 256

    def __len__(self):
        return math.ceil(len(self.x)/self.batch_size)   # ceil or floor

    def __getitem__(self, idx):
        b_x = copy.deepcopy(
            self.x[idx*self.batch_size:(idx+1)*self.batch_size])
        b_x_authors = copy.deepcopy(
            self.x_authors[idx * self.batch_size:(idx + 1) * self.batch_size])
        b_y = copy.deepcopy(self.y[idx*self.batch_size:(idx+1)*self.batch_size])

        for temp in b_x_authors:
            for tem in temp:
                for te in tem:
                    while len(te) < self.max_authors:
                        te.append(np.zeros(self.author_emb_dim))
                while len(tem) < self.max_seq:
                    tem.append(np.zeros(shape=(self.max_authors, self.author_emb_dim)))
            while len(temp) < self.max_papers:
                temp.append(np.zeros(shape=(self.max_seq, self.max_authors, self.author_emb_dim)))

        b_x_authors = np.array(b_x_authors)

        for temp in b_x:
            for tem in temp:
                while len(tem) < self.max_seq:
                    tem.append(np.zeros(tem[0].shape))

            while len(temp) < self.max_papers:
                temp.append(np.zeros(shape=(self.max_seq, self.paper_emb_dim)))
        b_x = np.array(b_x)

        return (b_x, b_x_authors), np.array(b_y)
