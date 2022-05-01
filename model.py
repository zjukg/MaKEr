import torch.nn as nn
import torch
import dgl
from ext_gnn import ExtGNN


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.dim = args.dim

        self.rel_comp = nn.Parameter(torch.Tensor(args.num_rel, args.num_rel_bases))
        nn.init.xavier_uniform_(self.rel_comp, gain=nn.init.calculate_gain('relu'))

        self.rel_feat = nn.Parameter(torch.Tensor(args.num_rel_bases, self.args.rel_dim))
        nn.init.xavier_uniform_(self.rel_feat, gain=nn.init.calculate_gain('relu'))

        self.ent_feat = nn.Parameter(torch.Tensor(args.num_ent, self.args.ent_dim))
        nn.init.xavier_uniform_(self.ent_feat, gain=nn.init.calculate_gain('relu'))

        self.rel_head_feat = nn.Parameter(torch.Tensor(args.num_rel_bases, self.args.ent_dim))
        self.rel_tail_feat = nn.Parameter(torch.Tensor(args.num_rel_bases, self.args.ent_dim))
        nn.init.xavier_uniform_(self.rel_head_feat, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.rel_tail_feat, gain=nn.init.calculate_gain('relu'))

        # for initializing relation in pattern graph (relation position graph)
        self.pattern_rel_ent = nn.Parameter(torch.Tensor(4, args.num_rel_bases))
        nn.init.xavier_uniform_(self.pattern_rel_ent, gain=nn.init.calculate_gain('relu'))

        self.ext_gnn = ExtGNN(args)

    # relation feature representation
    def init_pattern_g(self, pattern_g):
        with pattern_g.local_scope():
            etypes = pattern_g.edata['rel']
            pattern_g.edata['edge_h'] = self.pattern_rel_ent[etypes]

            message_func = dgl.function.copy_e('edge_h', 'msg')
            reduce_func = dgl.function.mean('msg', 'h')
            pattern_g.update_all(message_func, reduce_func)
            pattern_g.edata.pop('edge_h')

            # for observed rel
            obs_idx = (pattern_g.ndata['ori_idx'] != -1)
            pattern_g.ndata['h'][obs_idx] = self.rel_comp[pattern_g.ndata['ori_idx'][obs_idx]]

            rel_coef = pattern_g.ndata['h']

        return rel_coef

    # entity feature representation
    def init_g(self, g, rel_coef):
        with g.local_scope():
            num_edge = g.num_edges()
            etypes = g.edata['b_rel']

            rel_head_emb = torch.matmul(rel_coef, self.rel_head_feat)
            rel_tail_emb = torch.matmul(rel_coef, self.rel_tail_feat)

            g.edata['edge_h'] = torch.zeros(num_edge, self.args.ent_dim).to(self.args.gpu)

            non_inv_idx = (g.edata['inv'] == 0)
            inv_idx = (g.edata['inv'] == 1)
            g.edata['edge_h'][inv_idx] = rel_head_emb[etypes[inv_idx]]
            g.edata['edge_h'][non_inv_idx] = rel_tail_emb[etypes[non_inv_idx]]

            message_func = dgl.function.copy_e('edge_h', 'msg')
            reduce_func = dgl.function.mean('msg', 'h')
            g.update_all(message_func, reduce_func)
            g.edata.pop('edge_h')

            # for observed ent
            obs_idx = (g.ndata['ori_idx'] != -1)
            g.ndata['h'][obs_idx] = self.ent_feat[g.ndata['ori_idx'][obs_idx]]

            ent_feat = g.ndata['h']

        return ent_feat

    def forward(self, g, pattern_g):
        rel_coef = self.init_pattern_g(pattern_g)
        init_ent_feat = self.init_g(g, rel_coef)
        init_rel_feat = torch.matmul(rel_coef, self.rel_feat)
        ent_emb, rel_emb = self.ext_gnn(g, ent_feat=init_ent_feat, rel_feat=init_rel_feat)

        return ent_emb, rel_emb