import torch
import torch.nn as nn
from SRHGN import SRHGN

class Mp_encoder(nn.Module):
    def __init__(self, args, g_m_drug, g_m_disease):
        super(Mp_encoder, self).__init__()

        self.hidden_dim = args.hgt_out_dim
        self.drug_num = args.drug_number
        self.disease_num = args.disease_number
        self.protein_num = args.protein_number
        self.output_dim = args.hgt_in_dim
        self.num_layers = 2
        self.num_node_heads = 4
        self.num_type_heads = 4
        self.alpha = 0.5


        node_dict1 = {}
        edge_dict1 = {}
        node_dict2 = {}
        edge_dict2 = {}

        for ntype in g_m_drug.ntypes:
            node_dict1[ntype] = len(node_dict1)
        for etype in g_m_drug.etypes:
            edge_dict1[etype] = len(edge_dict1)


        for ntype in g_m_disease.ntypes:
            node_dict2[ntype] = len(node_dict2)
        for etype in g_m_disease.etypes:
            edge_dict2[etype] = len(edge_dict2)

        input_dims = {'disease': args.disease_number, 'drug': args.drug_number, 'protein': args.protein_number}

        self.SRHGN_encoder1 = SRHGN(g_m_drug, node_dict1, edge_dict1,
                                    input_dims,
                                    self.hidden_dim,
                                    self.output_dim,
                                    self.num_layers,
                                    self.num_node_heads,
                                    self.num_type_heads,
                                    self.alpha)

        self.SRHGN_encoder2 = SRHGN(g_m_disease, node_dict2, edge_dict2,
                                    input_dims,
                                    self.hidden_dim,
                                    self.output_dim,
                                    self.num_layers,
                                    self.num_node_heads,
                                    self.num_type_heads,
                                    self.alpha)

    def init_feat(self, g, features):

        input_dims = {}

        for ntype in g.ntypes:
            feats = features[ntype]
            g.nodes[ntype].data['x'] = feats
            input_dims[ntype] = feats.shape[1]

        return g, input_dims

    def forward(self, g_r, g_d, drug_feature, disease_feature, protein_feature):

        feature_dict = {
            'drug': drug_feature,
            'disease': disease_feature,
            'protein': protein_feature
        }

        G1, input_dims = self.init_feat(g_r, feature_dict)
        out_dict1, _, _1 = self.SRHGN_encoder1(G1, 'drug')

        G2, input_dims = self.init_feat(g_d, feature_dict)
        out_dict2, _, _2 = self.SRHGN_encoder2(G2, 'disease')

        emb_r = out_dict1['drug']
        emb_d = out_dict2['disease']

        return emb_r, emb_d
