import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .layers import MSGAGNNLayer

class LMAGNN_LogicModel(nn.Module):
    def __init__(self, params, loader, device):
        super().__init__()
        self.hidden_dim = params.hidden_dim
        self.n_layer = params.n_layer
        self.loader = loader
        self.device = device
        
        self.ent_embed = nn.Embedding(loader.n_ent, params.hidden_dim)
        self.rel_embed = nn.Embedding(2 * loader.n_rel + 1, params.hidden_dim)
        self.gamma = nn.Parameter(torch.ones(params.n_layer)) 
        
        self.gnn_layers = nn.ModuleList([
            MSGAGNNLayer(params.hidden_dim, params.attn_dim, loader.n_rel, 
                         loader.n_ent, params.n_node_topk[i], params.tau) 
            for i in range(params.n_layer)
        ])
        
        self.gate = nn.GRU(params.hidden_dim, params.hidden_dim)
        self.W_logic = nn.Linear(params.hidden_dim, 1, bias=False)
        self.W_struct = nn.Linear(params.hidden_dim, params.hidden_dim, bias=False)
        self.dropout = nn.Dropout(params.dropout)

    def forward(self, subs, rels, mode='train'):
        batch_size = len(subs)
        q_sub_ts = torch.LongTensor(subs).to(self.device)
        q_rel_ts = torch.LongTensor(rels).to(self.device)
        
        hidden = self.ent_embed(q_sub_ts)
        q_rel_embed = self.rel_embed(q_rel_ts)
        h_gru_state = torch.zeros((1, batch_size, self.hidden_dim)).to(self.device)
        nodes = np.stack([np.arange(batch_size), subs], axis=1)
        
        hop_features, hop_nodes = [], []

        for i in range(self.n_layer):
            nodes_np, edges, old_idx = self.loader.get_neighbors(nodes, batch_size, mode=mode)
            nodes_ts = torch.LongTensor(nodes_np).to(self.device)
            
            hidden, nodes_ts, sampled_mask = self.gnn_layers[i](
                q_rel_embed, hidden, edges, nodes_ts, old_idx, batch_size, self.device
            )
            
            # 路径GRU更新
            h_gru_expanded = torch.zeros((1, sampled_mask.size(0), self.hidden_dim), device=self.device).index_copy_(1, old_idx, h_gru_state)
            h_gru_curr = h_gru_expanded[:, sampled_mask, :]
            hidden, h_gru_state = self.gate(self.dropout(hidden).unsqueeze(0), h_gru_curr)
            hidden = hidden.squeeze(0)
            
            hop_features.append(hidden)
            hop_nodes.append(nodes_ts)
            nodes = nodes_ts.cpu().numpy()
            
        # 多跳特征融合 (Eq. 17)
        gamma_norm = F.softmax(self.gamma, dim=0)
        full_scores = torch.zeros((batch_size, self.loader.n_ent), device=self.device)
        q_semantic = self.W_struct(self.ent_embed(q_sub_ts) * q_rel_embed)

        for i in range(self.n_layer):
            h_i, n_i = hop_features[i], hop_nodes[i]
            s_logic = self.W_logic(h_i).squeeze(-1)
            target_embeds = self.ent_embed(n_i[:, 1])
            s_struct = torch.sum(q_semantic[n_i[:, 0]] * target_embeds, dim=1)
            full_scores[n_i[:, 0], n_i[:, 1]] += gamma_norm[i] * (s_logic + s_struct)
            
        return full_scores