import torch
import dgl
import numpy as np
from scipy import sparse
from collections import defaultdict as ddict
from torch.utils.data import Dataset
import lmdb
from utils import deserialize
import random


class TrainSubgraphDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.env = lmdb.open(args.db_path, readonly=True, max_dbs=1, lock=False)
        self.subgraphs_db = self.env.open_db("train_subgraphs".encode())

    def __len__(self):
        return self.args.num_train_subgraph

    @staticmethod
    def collate_fn(data):
        return data

    def get_train_g(self, sup_tri, ent_map_list, ent_mask):
        triples = torch.LongTensor(sup_tri)
        num_tri = triples.shape[0]
        g = dgl.graph((torch.cat([triples[:, 0].T, triples[:, 2].T]),
                       torch.cat([triples[:, 2].T, triples[:, 0].T])))
        g.edata['rel'] = torch.cat([triples[:, 1].T, triples[:, 1].T])
        g.edata['inv'] = torch.cat([torch.zeros(num_tri), torch.ones(num_tri)])

        ent_mask_list = np.array(list(map(lambda x: x in ent_mask, np.arange(len(ent_map_list)))))
        ent_map_list = np.array(ent_map_list)
        ent_map_list[ent_mask_list] = -1

        g.ndata['ori_idx'] = torch.tensor(ent_map_list)

        return g

    def get_pattern_g(self, pattern_tri, rel_map_list, rel_mask):
        triples = torch.LongTensor(pattern_tri)
        g = dgl.graph((triples[:, 0].T, triples[:, 2].T))
        g.edata['rel'] = triples[:, 1].T

        rel_mask_list = np.array(list(map(lambda x: x in rel_mask, np.arange(len(rel_map_list)))))
        rel_map_list = np.array(rel_map_list)
        rel_map_list[rel_mask_list] = -1

        g.ndata['ori_idx'] = torch.tensor(rel_map_list)

        return g

    def __getitem__(self, idx):
        with self.env.begin(db=self.subgraphs_db) as txn:
            str_id = '{:08}'.format(idx).encode('ascii')
            sup_tri, pattern_tri, que_tri, hr2t, rt2h, ent_reidx_list, rel_reidx_list = deserialize(txn.get(str_id))

        nentity = len(ent_reidx_list)

        que_neg_tail_ent = [np.random.choice(np.delete(np.arange(nentity), hr2t[(h, r)]),
                                        self.args.metatrain_num_neg) for h, r, t in que_tri]

        que_neg_head_ent = [np.random.choice(np.delete(np.arange(nentity), rt2h[(r, t)]),
                                        self.args.metatrain_num_neg) for h, r, t in que_tri]


        ent_mask = np.random.choice(np.arange(len(ent_reidx_list)),
                                    int(len(ent_reidx_list) * random.randint(3, 8) * 0.1), replace=False)
        rel_mask = np.random.choice(np.arange(len(rel_reidx_list)),
                                    int(len(rel_reidx_list) * random.randint(3, 8) * 0.1), replace=False)

        g = self.get_train_g(sup_tri, ent_reidx_list, ent_mask)
        pattern_g = self.get_pattern_g(pattern_tri, rel_reidx_list, rel_mask)

        return g, pattern_g, torch.tensor(que_tri), \
               torch.tensor(que_neg_tail_ent), torch.tensor(que_neg_head_ent)


class EvalDataset(Dataset):
    def __init__(self, args, data, que_triples):
        self.args = args

        self.hr2t = data.hr2t_all
        self.rt2h = data.rt2h_all
        self.triples = que_triples

        self.num_ent = data.num_ent

        self.num_cand = 'all'

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        pos_triple = self.triples[idx]
        h, r, t = pos_triple
        if self.num_cand == 'all':
            tail_label, head_label = self.get_label(self.hr2t[(h, r)], self.rt2h[(r, t)])
            pos_triple = torch.LongTensor(pos_triple)

            return pos_triple, tail_label, head_label
        else:
            neg_tail_cand = np.random.choice(np.delete(np.arange(self.num_ent), self.hr2t[(h, r)]),
                                             self.num_cand)

            neg_head_cand = np.random.choice(np.delete(np.arange(self.num_ent), self.rt2h[(r, t)]),
                                             self.num_cand)
            tail_cand = torch.from_numpy(np.concatenate((neg_tail_cand, [t])))
            head_cand = torch.from_numpy(np.concatenate((neg_head_cand, [h])))

            pos_triple = torch.LongTensor(pos_triple)

            return pos_triple, tail_cand, head_cand

    def get_label(self, true_tail, true_head):
        y_tail = np.zeros([self.num_ent], dtype=np.float32)
        for e in true_tail:
            y_tail[e] = 1.0
        y_head = np.zeros([self.num_ent], dtype=np.float32)
        for e in true_head:
            y_head[e] = 1.0

        return torch.FloatTensor(y_tail), torch.FloatTensor(y_head)

    @staticmethod
    def collate_fn(data):
        pos_triple = torch.stack([_[0] for _ in data], dim=0)
        tail_label_or_cand = torch.stack([_[1] for _ in data], dim=0)
        head_label_or_cand = torch.stack([_[2] for _ in data], dim=0)
        return pos_triple, tail_label_or_cand, head_label_or_cand


class Data(object):
    def __init__(self, args, data):
        self.args = args

        self.entity_dict = data['ent2id']
        self.relation_dict = data['rel2id']

        self.num_ent = len(self.entity_dict)
        self.num_rel = len(self.relation_dict)

    def get_train_g(self, sup_tri, ent_reidx_list=None):
        triples = torch.LongTensor(sup_tri)
        num_tri = triples.shape[0]
        g = dgl.graph((torch.cat([triples[:, 0].T, triples[:, 2].T]),
                       torch.cat([triples[:, 2].T, triples[:, 0].T])))
        g.edata['rel'] = torch.cat([triples[:, 1].T, triples[:, 1].T])
        g.edata['b_rel'] = torch.cat([triples[:, 1].T, triples[:, 1].T])
        g.edata['inv'] = torch.cat([torch.zeros(num_tri), torch.ones(num_tri)])

        if ent_reidx_list is None:
            g.ndata['ori_idx'] = torch.tensor(np.arange(g.num_nodes()))
        else:
            g.ndata['ori_idx'] = torch.tensor(ent_reidx_list)

        return g

    def get_pattern_g(self, pattern_tri, rel_reidx_list=None):
        triples = torch.LongTensor(pattern_tri)
        g = dgl.graph((triples[:, 0].T, triples[:, 2].T))
        g.edata['rel'] = triples[:, 1].T

        if rel_reidx_list is None:
            g.ndata['ori_idx'] = torch.tensor(np.arange(g.num_nodes()))
        else:
            g.ndata['ori_idx'] = torch.tensor(rel_reidx_list)

        return g

    def get_pattern_tri(self, sup_tri):
        # adjacency matrix for rel and ent
        rel_head = torch.zeros((self.num_rel, self.num_ent), dtype=torch.int)
        rel_tail = torch.zeros((self.num_rel, self.num_ent), dtype=torch.int)
        for tri in sup_tri:
            h, r, t = tri
            rel_head[r, h] += 1
            rel_tail[r, t] += 1

        # adjacency matrix for rel and rel of different pattern
        tail_head = torch.matmul(rel_tail, rel_head.T)
        head_tail = torch.matmul(rel_head, rel_tail.T)
        tail_tail = torch.matmul(rel_tail, rel_tail.T) - torch.diag(torch.sum(rel_tail, axis=1))
        head_head = torch.matmul(rel_head, rel_head.T) - torch.diag(torch.sum(rel_head, axis=1))

        # construct pattern graph from adjacency matrix
        src = torch.LongTensor([])
        dst = torch.LongTensor([])
        p_rel = torch.LongTensor([])
        p_w = torch.LongTensor([])
        for p_rel_idx, mat in enumerate([tail_head, head_tail, tail_tail, head_head]):
            sp_mat = sparse.coo_matrix(mat)
            src = torch.cat([src, torch.from_numpy(sp_mat.row)])
            dst = torch.cat([dst, torch.from_numpy(sp_mat.col)])
            p_rel = torch.cat([p_rel, torch.LongTensor([p_rel_idx] * len(sp_mat.data))])
            p_w = torch.cat([p_w, torch.from_numpy(sp_mat.data)])

        return torch.stack([src, p_rel, dst]).T.tolist()

    def get_hr2t_rt2h(self, triples):
        hr2t = ddict(list)
        rt2h = ddict(list)
        for tri in triples:
            h, r, t = tri
            hr2t[(h, r)].append(t)
            rt2h[(r, t)].append(h)

        return hr2t, rt2h


class TrainData(Data):
    def __init__(self, args, data):
        super(TrainData, self).__init__(args, data)
        self.train_triples = data['triples']

        self.hr2t_train, self.rt2h_train = self.get_hr2t_rt2h(self.train_triples)

        # g and pattern g
        self.g = self.get_train_g(self.train_triples).to(args.gpu)

        self.pattern_tri = self.get_pattern_tri(self.train_triples)
        self.pattern_g = self.get_pattern_g(self.pattern_tri).to(args.gpu)


class ValidData(Data):
    def __init__(self, args, data):
        super(ValidData, self).__init__(args, data)
        self.sup_triples = data['support']
        self.que_triples = data['query']

        self.ent_map_list = data['ent_map_list']
        self.rel_map_list = data['rel_map_list']

        self.hr2t_all, self.rt2h_all = self.get_hr2t_rt2h(self.sup_triples + self.que_triples)

        # g and pattern g
        self.g = self.get_train_g(self.sup_triples, ent_reidx_list=self.ent_map_list).to(args.gpu)

        self.pattern_tri = self.get_pattern_tri(self.sup_triples)
        self.pattern_g = self.get_pattern_g(self.pattern_tri, rel_reidx_list=self.rel_map_list).to(args.gpu)


class TestData(Data):
    def __init__(self, args, data):
        super(TestData, self).__init__(args, data)
        self.sup_triples = data['support']
        self.que_triples = data['query_uent'] + data['query_urel'] + data['query_uboth']
        self.que_uent = data['query_uent']
        self.que_urel = data['query_urel']
        self.que_uboth = data['query_uboth']

        self.ent_map_list = data['ent_map_list']
        self.rel_map_list = data['rel_map_list']

        self.hr2t_all, self.rt2h_all = self.get_hr2t_rt2h(self.sup_triples + self.que_triples)

        # g and pattern g
        self.g = self.get_train_g(self.sup_triples, ent_reidx_list=self.ent_map_list).to(args.gpu)

        self.pattern_tri = self.get_pattern_tri(self.sup_triples)
        self.pattern_g = self.get_pattern_g(self.pattern_tri, rel_reidx_list=self.rel_map_list).to(args.gpu)


class TrainDatasetMode(Dataset):
    def __init__(self, args, data, mode):
        self.args = args
        self.triples = data.train_triples
        self.num_ent = data.num_ent
        self.num_neg = args.kge_num_neg
        self.hr2t = data.hr2t_train
        self.rt2h = data.rt2h_train
        self.mode = mode

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        positive_sample = self.triples[idx]

        head, relation, tail = positive_sample

        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.num_neg:
            negative_sample = np.random.randint(self.num_ent, size=self.num_neg * 2)
            if self.mode == 'head-batch':
                mask = np.in1d(
                    negative_sample,
                    self.rt2h[(relation, tail)],
                    assume_unique=True,
                    invert=True
                )
            elif self.mode == 'tail-batch':
                mask = np.in1d(
                    negative_sample,
                    self.hr2t[(head, relation)],
                    assume_unique=True,
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:self.num_neg]

        negative_sample = torch.LongTensor(negative_sample)

        positive_sample = torch.LongTensor(positive_sample)

        return positive_sample, negative_sample, self.mode


    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        mode = data[0][2]
        return positive_sample, negative_sample, mode


class OneShotIterator(object):
    def __init__(self, dataloader):
        self.iterator = self.one_shot_iterator(dataloader)

    def __next__(self):
        data = next(self.iterator)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data


class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data
