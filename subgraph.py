import pickle
import numpy as np
from utils import get_g, serialize
import torch
import lmdb
import dgl
from collections import defaultdict as ddict
from tqdm import tqdm
import random
from scipy import sparse
import multiprocessing as mp


def gen_subgraph_datasets(args):
    print('----------generate tasks(sub-KGs) for meta-training----------')
    data = pickle.load(open(args.data_path, 'rb'))
    bg_train_g = get_g(data['train']['triples'])

    BYTES_PER_DATUM = get_average_subgraph_size(args, args.num_sample_for_estimate_size, bg_train_g) * 2
    map_size = (args.num_train_subgraph) * BYTES_PER_DATUM
    env = lmdb.open(args.db_path, map_size=map_size, max_dbs=1)
    train_subgraphs_db = env.open_db("train_subgraphs".encode())

    with mp.Pool(processes=10, initializer=intialize_worker, initargs=(args, bg_train_g)) as p:
        idx_ = range(args.num_train_subgraph)
        for (str_id, datum) in tqdm(p.imap(sample_one_subgraph, idx_), total=args.num_train_subgraph):
            with env.begin(write=True, db=train_subgraphs_db) as txn:
                txn.put(str_id, serialize(datum))


def intialize_worker(args, bg_train_g):
    global args_, bg_train_g_
    args_, bg_train_g_ = args, bg_train_g


def sample_one_subgraph(idx_):
    args = args_
    bg_train_g = bg_train_g_

    # get graph with bi-direction
    bg_train_g_undir = dgl.graph((torch.cat([bg_train_g.edges()[0], bg_train_g.edges()[1]]),
                                  torch.cat([bg_train_g.edges()[1], bg_train_g.edges()[0]])))

    # induce sub-graph by sampled nodes
    while True:
        while True:
            sel_nodes = []
            for i in range(args.rw_0):
                if i == 0:
                    cand_nodes = np.arange(bg_train_g.num_nodes())
                else:
                    cand_nodes = sel_nodes
                try:
                    rw, _ = dgl.sampling.random_walk(bg_train_g_undir,
                                                     np.random.choice(cand_nodes, 1, replace=False).repeat(args.rw_1),
                                                     length=args.rw_2)
                except ValueError:
                    print(cand_nodes)
                sel_nodes.extend(np.unique(rw.reshape(-1)))
                sel_nodes = list(np.unique(sel_nodes)) if -1 not in sel_nodes else list(np.unique(sel_nodes))[1:]
            sub_g = dgl.node_subgraph(bg_train_g, sel_nodes)

            if sub_g.num_nodes() >= 50:
                break

        sub_tri = torch.stack([sub_g.ndata[dgl.NID][sub_g.edges()[0]],
                               sub_g.edata['rel'],
                               sub_g.ndata[dgl.NID][sub_g.edges()[1]]])

        sub_tri = sub_tri.T.tolist()

        random.shuffle(sub_tri)

        ent_freq = ddict(int)
        rel_freq = ddict(int)
        triples_reidx = []

        rel_reidx = dict()
        relidx = 0

        ent_reidx = dict()
        entidx = 0

        for tri in sub_tri:
            h, r, t = tri
            if h not in ent_reidx.keys():
                ent_reidx[h] = entidx
                entidx += 1
            if t not in ent_reidx.keys():
                ent_reidx[t] = entidx
                entidx += 1
            if r not in rel_reidx.keys():
                rel_reidx[r] = relidx
                relidx += 1
            ent_freq[ent_reidx[h]] += 1
            ent_freq[ent_reidx[t]] += 1
            rel_freq[rel_reidx[r]] += 1
            triples_reidx.append([ent_reidx[h], rel_reidx[r], ent_reidx[t]])

        ent_reidx_inv = {v: k for k, v in ent_reidx.items()}
        rel_reidx_inv = {v: k for k, v in rel_reidx.items()}
        ent_map_list = [ent_reidx_inv[i] for i in range(len(ent_reidx))]
        rel_map_list = [rel_reidx_inv[i] for i in range(len(rel_reidx))]

        # randomly get query triples
        que_tris = []
        sup_tris = []
        for idx, tri in enumerate(triples_reidx):
            h, r, t = tri
            if ent_freq[h] > 2 and ent_freq[t] > 2 and rel_freq[r] > 2:
                que_tris.append(tri)
                ent_freq[h] -= 1
                ent_freq[t] -= 1
                rel_freq[r] -= 1
            else:
                sup_tris.append(tri)

            if len(que_tris) >= int(len(triples_reidx)*0.1):
                break

        sup_tris.extend(triples_reidx[idx+1:])

        if len(que_tris) >= int(len(triples_reidx)*0.1):
            break

    # hr2t, rt2h
    hr2t, rt2h, rel_head, rel_tail = get_hr2t_rt2h_sup_que(sup_tris, que_tris)
    pattern_tris = get_train_pattern_g(rel_head, rel_tail)

    str_id = '{:08}'.format(idx_).encode('ascii')

    return str_id, (sup_tris, pattern_tris, que_tris, hr2t, rt2h, ent_map_list, rel_map_list)


def get_train_pattern_g(rel_head, rel_tail):
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


def get_average_subgraph_size(args, sample_size, bg_train_g):
    total_size = 0

    with mp.Pool(processes=10, initializer=intialize_worker, initargs=(args, bg_train_g)) as p:
        idx_ = range(sample_size)
        for (str_id, datum) in p.imap(sample_one_subgraph, idx_):
            total_size += len(serialize(datum))

    return total_size / sample_size


def get_hr2t_rt2h_sup_que(sup_tris, que_tris):
    hr2t = ddict(list)
    rt2h = ddict(list)

    triples = torch.LongTensor(sup_tris)
    num_rel = torch.unique(triples[:, 1]).shape[0]
    num_ent = torch.unique(torch.cat((triples[:, 0], triples[:, 2]))).shape[0]
    rel_head = torch.zeros((num_rel, num_ent), dtype=torch.int)
    rel_tail = torch.zeros((num_rel, num_ent), dtype=torch.int)

    for tri in sup_tris:
        h, r, t = tri
        hr2t[(h, r)].append(t)
        rt2h[(r, t)].append(h)

        rel_head[r, h] += 1
        rel_tail[r, t] += 1

    for tri in que_tris:
        h, r, t = tri
        hr2t[(h, r)].append(t)
        rt2h[(r, t)].append(h)

    que_hr2t = dict()
    que_rt2h = dict()
    for tri in que_tris:
        h, r, t = tri
        que_hr2t[(h, r)] = hr2t[(h, r)]
        que_rt2h[(r, t)] = rt2h[(r, t)]

    return que_hr2t, que_rt2h, rel_head, rel_tail