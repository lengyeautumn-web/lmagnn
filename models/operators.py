import torch
import torch.nn as nn

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