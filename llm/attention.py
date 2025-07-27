import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

    def forward(self, x):
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        attn_scores = q @ k.T / (self.d_model ** 0.5)
        attn_scores = torch.softmax(attn_scores, dim=-1)
        return 0
        

k = torch.randn(3, 100, 768)
print(k.shape)
print(k.T.shape)
print(k.transpose(-1, -2).shape)







