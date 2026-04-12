import os
import torch
import numpy as np
from scipy.sparse import csr_matrix
from collections import defaultdict
import logging

class DataLoader:
    def __init__(self, args):
        self.args = args
        self.task_dir = args.data_path
        
        # Load mappings
        self.entity2id = self._read_map('entities.txt')
        self.relation2id = self._read_map('relations.txt')
        self.n_ent, self.n_rel = len(self.entity2id), len(self.relation2id)
        
        self.filters = defaultdict(set)
        self.fact_triple = self.read_triples('facts.txt')
        self.train_triple = self.read_triples('train.txt')
        self.valid_triple = self.read_triples('valid.txt')
        self.test_triple = self.read_triples('test.txt')
        
        self.all_triple = np.concatenate([np.array(self.fact_triple), np.array(self.train_triple)], axis=0)
        self.valid_data = self.double_triple(self.valid_triple)
        self.test_data = self.double_triple(self.test_triple)
        
        self.shuffle_train()
        self.load_test_graph(self.double_triple(self.fact_triple) + self.double_triple(self.train_triple))
        self.valid_q, self.valid_a = self.load_query(self.valid_data)
        self.test_q, self.test_a = self.load_query(self.test_data)
        self.n_valid, self.n_test = len(self.valid_q), len(self.test_q)

    def _read_map(self, filename):
        with open(os.path.join(self.task_dir, filename)) as f:
            return {line.strip(): i for i, line in enumerate(f)}

    def read_triples(self, filename):
        triples = []
        path = os.path.join(self.task_dir, filename)
        if not os.path.exists(path): return []
        with open(path) as f:
            for line in f:
                h_s, r_s, t_s = line.strip().split()
                h, r, t = self.entity2id[h_s], self.relation2id[r_s], self.entity2id[t_s]
                triples.append([h, r, t])
                self.filters[(h, r)].add(t)
                self.filters[(t, r + self.n_rel)].add(h)
        return triples

    def double_triple(self, triples):
        return triples + [[t, r + self.n_rel, h] for h, r, t in triples]

    def load_graph(self, triples):
        idd = np.concatenate([np.expand_dims(np.arange(self.n_ent), 1), 
                             2 * self.n_rel * np.ones((self.n_ent, 1)), 
                             np.expand_dims(np.arange(self.n_ent), 1)], 1)
        self.KG = np.concatenate([np.array(triples), idd], 0)
        self.M_sub = csr_matrix((np.ones((len(self.KG),)), (np.arange(len(self.KG)), self.KG[:, 0])), 
                                shape=(len(self.KG), self.n_ent))

    def load_test_graph(self, triples):
        idd = np.concatenate([np.expand_dims(np.arange(self.n_ent), 1), 
                             2 * self.n_rel * np.ones((self.n_ent, 1)), 
                             np.expand_dims(np.arange(self.n_ent), 1)], 1)
        self.tKG = np.concatenate([np.array(triples), idd], 0)
        self.tM_sub = csr_matrix((np.ones((len(self.tKG),)), (np.arange(len(self.tKG)), self.tKG[:, 0])), 
                                 shape=(len(self.tKG), self.n_ent))

    def load_query(self, triples):
        trip_hr = defaultdict(list)
        for h, r, t in triples: trip_hr[(h, r)].append(t)
        queries = np.array(list(trip_hr.keys()))
        answers = [np.array(v) for v in trip_hr.values()]
        return queries, answers

    def get_neighbors(self, nodes, batchsize, mode='train'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        KG, M_sub = (self.KG, self.M_sub) if mode == 'train' else (self.tKG, self.tM_sub)
        node_1hot = csr_matrix((np.ones(len(nodes)), (nodes[:, 1], nodes[:, 0])), shape=(self.n_ent, nodes.shape[0]))
        edge_1hot = M_sub.dot(node_1hot)
        edges = np.nonzero(edge_1hot)
        sampled_edges = np.concatenate([np.expand_dims(edges[1], 1), KG[edges[0]]], axis=1)
        sampled_edges = torch.LongTensor(sampled_edges).to(device)
        
        head_nodes, head_idx = torch.unique(sampled_edges[:, [0, 1]], dim=0, return_inverse=True)
        tail_nodes, tail_index = torch.unique(sampled_edges[:, [0, 3]], dim=0, return_inverse=True)
        sampled_edges = torch.cat([sampled_edges, head_idx.unsqueeze(1), tail_index.unsqueeze(1)], 1)
        
        mask = sampled_edges[:, 2] == (self.n_rel * 2)
        old_nodes_new_idx = tail_index[mask].sort()[0]
        return tail_nodes.cpu().numpy(), sampled_edges, old_nodes_new_idx

    def get_batch(self, batch_idx, data='train'):
        if data == 'train': return self.train_data[batch_idx]
        query, answer = (self.valid_q, self.valid_a) if data == 'valid' else (self.test_q, self.test_a)
        subs, rels = query[batch_idx, 0], query[batch_idx, 1]
        objs = np.zeros((len(batch_idx), self.n_ent))
        for i, idx in enumerate(batch_idx): objs[i][answer[idx]] = 1
        return subs, rels, objs

    def shuffle_train(self):
        all_trip = self.all_triple.copy()
        np.random.shuffle(all_trip)
        bar = int(len(all_trip) * self.args.fact_ratio)
        self.fact_data = self.double_triple(all_trip[:bar].tolist())
        self.train_data = np.array(self.double_triple(all_trip[bar:].tolist()))
        self.load_graph(self.fact_data)
        self.n_train = len(self.train_data)