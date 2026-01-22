import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from Contrast import Contrast
from Metapath_Augmentation.mp_encoder import Mp_encoder
from graph_transformer import GraphTransformer
import os


class MRSHGNN(nn.Module):
    def __init__(self, args, g_m_drug, g_m_disease, g_s_drug, g_s_disease):
        super(MRSHGNN, self).__init__()
        self.args = args
        self.mp = Mp_encoder(self.args, g_m_drug, g_m_disease)
        self.contrast_r = Contrast(self.args)
        self.contrast_d = Contrast(self.args)
        self.n_neg = self.args.n_neg
        self.pool = self.args.pool

        self.drug_linear = nn.Linear(args.embedding_dim, args.hgt_in_dim)
        self.disease_linear = nn.Linear(args.embedding_dim, args.hgt_in_dim)
        self.protein_linear = nn.Linear(args.embedding_dim, args.hgt_in_dim)

        self.drug_linear = nn.Linear(args.drug_number, args.hgt_in_dim)
        self.disease_linear = nn.Linear(args.disease_number, args.hgt_in_dim)
        self.protein_linear = nn.Linear(args.protein_number, args.hgt_in_dim)

        self.gt_drug = GraphTransformer(args.hgt_layer, args.drug_number, args.hgt_in_dim, args.hgt_in_dim,
                                        args.hgt_head, args.dropout)
        self.gt_disease = GraphTransformer(args.hgt_layer, args.disease_number, args.hgt_in_dim, args.hgt_out_dim,
                                           args.hgt_head, args.dropout)

        self.hidden_dim = args.hgt_out_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.args.hgt_out_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2)
        )

    def forward(self, g_rm, g_dm, g_r, g_d, pos_r, pos_d,
                drug_feature, disease_feature, protein_feature, pos_sample, neg_sample, neg_candidates_r, neg_candidates_d, drdr_graph, didi_graph):

        dr_sim, dr_layer_emb = self.gt_drug(drdr_graph)
        di_sim, di_layer_emb = self.gt_disease(didi_graph)


        r_mp, d_mp = self.mp(g_rm, g_dm, drug_feature, disease_feature, protein_feature)

        l_r = self.contrast_r(r_mp, dr_sim, pos_r)
        l_d = self.contrast_d(d_mp, di_sim, pos_d)

        r_mp = torch.add(r_mp, dr_sim)
        d_mp = torch.add(d_mp, di_sim)

        emb_neg_r = self.negative_sampling(r_mp, d_mp, pos_sample, neg_sample, neg_candidates_r)
        emb_neg_d = self.negative_sampling(d_mp, r_mp, pos_sample[:, [1, 0]], neg_sample[:, [1, 0]], neg_candidates_d)

        r_emb = r_mp
        d_emb = d_mp


        drdi_emb_pos = torch.mul(r_emb[pos_sample[:, 0]], d_emb[pos_sample[:, 1]])
        drdi_emb_neg = torch.mul(emb_neg_r, emb_neg_d)
        drdi_emb = torch.cat((drdi_emb_pos, drdi_emb_neg), dim=0)
        output = self.mlp(drdi_emb)

        return l_r + l_d, output, drdi_emb_pos, drdi_emb_neg



    def negative_sampling(self, emb_r, emb_d, pos_sample, neg_sample, neg_candidates):

        neg_num = neg_sample.shape[0]

        pos_anchor = neg_sample[:, 0]
        pos_target = pos_sample[:, 1]

        emb_anchor = emb_r[pos_anchor]
        emb_pos = emb_d[pos_target]
        emb_neg = emb_d[neg_candidates]

        if self.pool != 'concat':
            emb_anchor = self.pooling(emb_anchor).unsqueeze(dim=1)

        if self.args.negative_rate != 1.0:
            for i in range(int(self.args.negative_rate) - 1):
                emb_pos = torch.cat((emb_pos, emb_pos), dim=0)

        seed = torch.rand(neg_num, 1, 1).to(emb_anchor.device)
        emb_neg = seed * emb_pos.unsqueeze(dim=1) + (1 - seed) * emb_neg

        scores = (emb_anchor.unsqueeze(dim=1) * emb_neg).sum(dim=-1)
        indices = torch.max(scores, dim=1)[1].detach().unsqueeze(dim=1)

        neg_items_emb_ = emb_neg.permute(0, 2, 1)
        emb_neg = neg_items_emb_[[[i] for i in range(neg_num)], range(neg_items_emb_.shape[1]), indices]
        return emb_neg

    def pooling(self, emb):
        if self.pool == 'mean':
            return emb.mean(dim=1)
        elif self.pool == 'sum':
            return emb.sum(dim=1)
        elif self.pool == 'concat':
            return emb.view(emb.shape[0], -1)
        else:  # final
            return emb[:, -1, :]









