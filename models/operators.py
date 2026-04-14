import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class LogicOperators(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W_always = nn.Linear(dim, dim)
        self.W_until_A = nn.Linear(dim, dim)
        self.W_until_B = nn.Linear(dim, dim)
        self.W_eventually = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.gate = nn.Sigmoid()

    def forward(self, h_prev, h_curr, op_type='always'):
        """
        h_prev: 上一层的特征
        h_curr: 当前聚合后的特征
        """
        if op_type == 'always':
            # Eq. 10: 持续性约束
            g = self.gate(self.W_always(h_curr))
            return g * h_prev + (1 - g) * h_curr
        elif op_type == 'until':
            # Eq. 11: 跳转约束
            gate_B = self.gate(self.W_until_B(h_curr))
            return gate_B * h_curr + (1 - gate_B) * self.W_until_A(h_prev)
        elif op_type == 'eventually':
            # Eq. 12: 未来约束
            return h_curr + self.W_eventually(h_curr)
        return h_curr


class AdvancedLogicOperators(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()

        # Define weight matrices for each operation
        self.W_always = nn.Linear(dim, hidden_dim)
        self.W_until_A = nn.Linear(dim, hidden_dim)
        self.W_until_B = nn.Linear(dim, hidden_dim)
        self.W_eventually = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )

        self.W_next = nn.Linear(dim, hidden_dim)
        self.W_globally = nn.Linear(dim, hidden_dim)
        self.W_previous = nn.Linear(dim, hidden_dim)

        # Additional operations
        self.W_strongly_eventually = nn.Linear(dim, hidden_dim)
        self.W_weakly_always = nn.Linear(dim, hidden_dim)
        self.W_release = nn.Linear(dim, hidden_dim)
        self.W_not = nn.Linear(dim, hidden_dim)
        self.W_and = nn.Linear(dim, hidden_dim)
        self.W_or = nn.Linear(dim, hidden_dim)
        self.W_unless = nn.Linear(dim, hidden_dim)
        self.W_implies = nn.Linear(dim, hidden_dim)

        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=4)

        # Non-linear activations
        self.relu = nn.ReLU()
        self.prelu = nn.PReLU()
        self.selu = nn.SELU()

        # Gate mechanism
        self.gate = nn.Sigmoid()
        self.gate_attention = nn.Softmax(dim=-1)

        # Dropout and Batch Normalization for regularization
        self.dropout = nn.Dropout(p=0.5)
        self.batch_norm = nn.BatchNorm1d(dim)

        # Weight initialization for stability
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h_prev, h_curr, op_type='always'):
        """
        h_prev: Previous layer features
        h_curr: Current aggregated features
        """
        if op_type == 'always':
            g = self.gate(self.W_always(h_curr))
            return g * h_prev + (1 - g) * h_curr

        elif op_type == 'until':
            gate_B = self.gate(self.W_until_B(h_curr))
            h_A = self.W_until_A(h_prev)
            return gate_B * h_curr + (1 - gate_B) * h_A

        elif op_type == 'eventually':
            return self.selu(self.W_eventually(h_curr) + h_curr)

        elif op_type == 'next':
            next_h = self.W_next(h_curr)
            return self.relu(next_h + h_curr)

        elif op_type == 'globally':
            globally_h = self.W_globally(h_curr)
            return self.prelu(globally_h + h_curr)

        elif op_type == 'previous':
            prev_h = self.W_previous(h_prev)
            return self.relu(prev_h + h_prev)

        elif op_type == 'strongly_eventually':
            h_strong = self.W_strongly_eventually(h_curr)
            return self.selu(h_strong + h_curr)

        elif op_type == 'weakly_always':
            weak_gate = self.gate(self.W_weakly_always(h_curr))
            return weak_gate * h_prev + (1 - weak_gate) * h_curr

        elif op_type == 'release':
            release_h = self.W_release(h_curr)
            return self.relu(release_h + h_curr)

        elif op_type == 'not':
            return self.gate(self.W_not(h_curr)) * (1 - h_curr) + (1 - self.gate(self.W_not(h_curr))) * h_curr

        elif op_type == 'and':
            gate = self.gate(self.W_and(h_curr))
            return gate * h_prev * h_curr + (1 - gate) * h_curr

        elif op_type == 'or':
            gate = self.gate(self.W_or(h_curr))
            return gate * h_prev + (1 - gate) * h_curr

        elif op_type == 'unless':
            gate = self.gate(self.W_unless(h_curr))
            return gate * h_prev + (1 - gate) * h_curr

        elif op_type == 'implies':
            gate = self.gate(self.W_implies(h_curr))
            return gate * h_prev + (1 - gate) * h_curr

        # Attention mechanism for all operations
        attn_output, attn_weights = self.attention(h_prev.unsqueeze(0), h_curr.unsqueeze(0), h_curr.unsqueeze(0))
        attn_output = attn_output.squeeze(0)

        g_attn = self.gate_attention(attn_weights).mean(dim=1)
        return g_attn * attn_output + (1 - g_attn) * h_curr

    def regularize(self, output, target):
        """
        Custom loss function with L2 regularization
        """
        mse_loss = F.mse_loss(output, target)
        l2_reg = sum(p.pow(2).sum() for p in self.parameters())
        return mse_loss + 0.001 * l2_reg

    def dropout_regularize(self, output):
        """
        Apply dropout for regularization during training
        """
        return self.dropout(output)

    def batch_normalization(self, output):
        """
        Apply batch normalization for regularization
        """
        return self.batch_norm(output)
