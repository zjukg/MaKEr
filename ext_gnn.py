import torch.nn as nn
import torch
import dgl.function as fn
import torch.nn.functional as F


class ExtGNNLayer(nn.Module):
    def __init__(self, args, act=None):
        super(ExtGNNLayer, self).__init__()
        self.args = args
        self.act = act

        # define in/out/loop transform layer
        self.W_O = nn.Linear(args.rel_dim + args.ent_dim, args.ent_dim)
        self.W_I = nn.Linear(args.rel_dim + args.ent_dim, args.ent_dim)
        self.W_S = nn.Linear(args.ent_dim, args.ent_dim)

        # define relation transform layer
        self.W_R = nn.Linear(args.rel_dim, args.rel_dim)

    def msg_func(self, edges):
        comp_h = torch.cat((edges.data['h'], edges.src['h']), dim=-1)

        non_inv_idx = (edges.data['inv'] == 0)
        inv_idx = (edges.data['inv'] == 1)

        msg = torch.zeros_like(edges.src['h'])
        msg[non_inv_idx] = self.W_I(comp_h[non_inv_idx])
        msg[inv_idx] = self.W_O(comp_h[inv_idx])

        return {'msg': msg}

    def apply_node_func(self, nodes):
        h_new = self.W_S(nodes.data['h']) + nodes.data['h_agg']

        if self.act is not None:
            h_new = self.act(h_new)

        return {'h': h_new}

    def edge_update(self, rel_emb):
        h_edge_new = self.W_R(rel_emb)

        if self.act is not None:
            h_edge_new = self.act(h_edge_new)

        return h_edge_new

    def forward(self, g, ent_emb, rel_emb):
        with g.local_scope():
            g.edata['h'] = rel_emb[g.edata['b_rel']]
            g.ndata['h'] = ent_emb

            g.update_all(self.msg_func, fn.mean('msg', 'h_agg'), self.apply_node_func)

            rel_emb = self.edge_update(rel_emb)
            ent_emb = g.ndata['h']

        return ent_emb, rel_emb


class ExtGNN(nn.Module):
    # knowledge extrapolation with GNN
    def __init__(self, args):
        super(ExtGNN, self).__init__()
        self.args = args
        self.layers = nn.ModuleList()

        for idx in range(args.num_layers):
            if idx == args.num_layers - 1:
                self.layers.append(ExtGNNLayer(args, act=None))
            else:
                self.layers.append(ExtGNNLayer(args, act=F.relu))

    def forward(self, g, **param):
        rel_emb = param['rel_feat']
        ent_emb = param['ent_feat']
        for layer in self.layers:
            ent_emb, rel_emb = layer(g, ent_emb, rel_emb)

        return ent_emb, rel_emb
