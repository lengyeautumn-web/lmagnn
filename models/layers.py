import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from .operators import LogicOperators

class MSGAGNNLayer(nn.Module):
    def __init__(self, in_dim, attn_dim, n_rel, n_ent, n_node_topk, tau):
        super().__init__()
        self.n_ent = n_ent
        self.n_node_topk = n_node_topk
        self.tau = tau
        self.rela_embed = nn.Embedding(2 * n_rel + 1, in_dim)
        
        # Eq. 7: Logical Message Attention
        self.W_attn = nn.Sequential(
            nn.Linear(in_dim * 3, attn_dim),
            nn.ReLU(),
            nn.Linear(attn_dim, 1),
            nn.Sigmoid()
        )
        
        self.logic_ops = LogicOperators(in_dim)
        self.W_msg = nn.Linear(in_dim, in_dim, bias=False)
        self.W_samp = nn.Linear(in_dim, 1)

    def forward(self, q_rel_embed, h_prev_all, edges, nodes, old_nodes_new_idx, batchsize, device):
        sub_idx, rel_type, obj_idx, b_idx = edges[:, 4], edges[:, 2], edges[:, 5], edges[:, 0]
        
        hs = h_prev_all[sub_idx]
        hr = self.rela_embed(rel_type)
        h_qr = q_rel_embed[b_idx]
        
        # 1. 逻辑注意力
        alpha = self.W_attn(torch.cat([hs, hr, h_qr], dim=-1))
        
        # 2. 消息聚合 (Eq. 6)
        message = alpha * (hs + hr)
        h_curr_agg = scatter(message, index=obj_idx, dim=0, dim_size=nodes.shape[0], reduce='sum')
        
        # 3. 逻辑演化注入 (Eq. 10)
        h_logic = h_curr_agg.clone()
        if len(old_nodes_new_idx) > 0:
            h_logic[old_nodes_new_idx] = self.logic_ops(
                h_prev_all, h_curr_agg[old_nodes_new_idx], op_type='always'
            )
        
        hidden_new = F.relu(self.W_msg(h_logic))

        # 4. 动态 Top-k 路径剪枝
        if self.n_node_topk <= 0: return hidden_new, nodes, None
        
        mask_diff = torch.ones(nodes.shape[0], device=device)
        mask_diff[old_nodes_new_idx] = 0
        diff_idx = mask_diff.bool()
        
        node_scores = torch.full((batchsize, self.n_ent), float('-inf'), device=device)
        node_scores[nodes[diff_idx, 0], nodes[diff_idx, 1]] = self.W_samp(hidden_new[diff_idx]).squeeze(-1)
        
        probs = F.gumbel_softmax(node_scores, tau=self.tau, hard=False) if self.training else F.softmax(node_scores, dim=1)
        topk_indices = probs.topk(min(self.n_node_topk, self.n_ent), dim=1).indices.reshape(-1)
        b_repeat = torch.arange(batchsize, device=device).repeat_interleave(self.n_node_topk)
        
        sampled_mask_ent = torch.zeros((batchsize, self.n_ent), device=device)
        sampled_mask_ent[b_repeat, topk_indices] = 1
        
        final_mask = (~diff_idx).clone()
        final_mask[diff_idx] = sampled_mask_ent[nodes[diff_idx, 0], nodes[diff_idx, 1]].bool()
        
        return hidden_new[final_mask], nodes[final_mask], final_mask