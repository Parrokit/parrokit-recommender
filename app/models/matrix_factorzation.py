import torch

class MatrixFactorization(torch.nn.Module):
    def __init__(self, num_users, num_items, factors=32):
        super().__init__()
        self.user_factors = torch.nn.Embedding(num_users,factors) # (80000, 32)
        self.item_factors = torch.nn.Embedding(num_items,factors) # (16471, 32)
        torch.nn.init.normal_(self.user_factors.weight, std=0.05)
        torch.nn.init.normal_(self.item_factors.weight, std=0.05)
    
    def forward(self,user_idx, item_idx):
        u = self.user_factors(user_idx) # (batch_size, 32)
        v = self.item_factors(item_idx) # (batch_size, 32)
        return (u*v).sum(dim=1) # 원소별 곱한 후 sigma{32개} -> (batch_size)