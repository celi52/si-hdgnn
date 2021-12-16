import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):

    def __init__(self, args, concat=True):
        super(GraphAttentionLayer, self).__init__()

        self.concat = concat
        embed_d = args.embed_d

        self.a_neigh_att = nn.Parameter(torch.ones(embed_d * 2, 1), requires_grad=True)
        self.p_neigh_att = nn.Parameter(torch.ones(embed_d * 2, 1), requires_grad=True)
        self.v_neigh_att = nn.Parameter(torch.ones(embed_d * 2, 1), requires_grad=True)

        self.softmax = nn.Softmax(dim=1)
        self.act = nn.LeakyReLU()
        self.drop = nn.Dropout(p=0.5)
        self.embed_d = embed_d
        self.bn = nn.BatchNorm1d(embed_d)


    def forward(self, node_type, c_agg_batch, concate_embed, a_agg_batch, p_agg_batch, v_agg_batch):

        if node_type == 1:
            atten_w = self.act(torch.bmm(concate_embed, self.a_neigh_att.unsqueeze(0).expand(len(c_agg_batch), \
                *self.a_neigh_att.size())))

        elif node_type == 2:
            atten_w = self.act(torch.bmm(concate_embed, self.p_neigh_att.unsqueeze(0).expand(len(c_agg_batch), \
                *self.p_neigh_att.size())))
        else:
            atten_w = self.act(torch.bmm(concate_embed, self.v_neigh_att.unsqueeze(0).expand(len(c_agg_batch), \
                *self.v_neigh_att.size())))

        atten_w = self.softmax(atten_w).view(len(c_agg_batch), 1, 4)


        concate_embed = torch.cat((c_agg_batch, a_agg_batch, p_agg_batch, \
                                   v_agg_batch), 1).view(len(c_agg_batch), 4, self.embed_d)

        weight_agg_batch = torch.bmm(atten_w, concate_embed).view(len(c_agg_batch), self.embed_d)
        return weight_agg_batch

