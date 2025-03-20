import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.nn import Module, Linear, LayerNorm, MultiheadAttention

class Meta(nn.Module):
   def __init__(self, d_model=768, nhead=12, mlp_ratio=4):
       super().__init__()
       self.nhead = nhead
       self.theta = nn.Parameter(torch.ones(1))
       self.eta = nn.Parameter(torch.ones(1))
       self.attention = nn.MultiheadAttention(2*d_model, nhead, batch_first=True)
       self.mlp = nn.Sequential(
           nn.Linear(2*d_model, 2*d_model*mlp_ratio),
           nn.GELU(),
           nn.Linear(2*d_model*mlp_ratio, 2*d_model)
       )
       self.norm1 = nn.LayerNorm(2*d_model)
       self.norm2 = nn.LayerNorm(2*d_model)
       
   def forward(self, weights, grads):
       x = torch.cat([weights, grads], dim=-1)
       attn_out, _ = self.attention(x, x, x)
       x = self.norm1(x + attn_out)
       x = self.norm2(x + self.mlp(x))
       w, g = x.chunk(2, dim=-1)
       return self.theta * w + self.eta * g

class Maestro(torch.optim.Optimizer):
   def __init__(self, param_groups):
       for group in param_groups:
           group['meta'] = Meta()
       super().__init__(param_groups, dict())
       self.meta_opt = AdamW(sum([list(g['meta'].parameters()) for g in param_groups], []))
           
   def meta_step(self, model, batch_A, batch_B, sliding_window_size):
       loss_A = model(batch_A, batch_A, sliding_window_size)
       loss_A.backward()

       for group in self.param_groups:
           for p in group['params']:
               if p.grad is None: continue
               rows = p.view(-1, p.shape[-1])
               grad_rows = p.grad.view(-1, p.shape[-1])
               p.data.sub_(group['meta'](rows, grad_rows).view_as(p))

       loss_AB = model(torch.cat([batch_A, batch_B]), torch.cat([batch_A, batch_B]), sliding_window_size)
       self.meta_opt.zero_grad()
       loss_AB.backward()
       self.meta_opt.step()

       return loss_A