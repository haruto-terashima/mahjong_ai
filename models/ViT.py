import torch
import torch.nn as nn

from einops import repeat
from einops.layers.torch import Rearrange


class Patching(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.net = Rearrange("b c (h ph) (w pw) -> b (h W) (ph pw c)", ph = patch_size, pw = patch_size)

    def forward(self, x):
        x = self.net(x)

        return x
    
class LinearProjection(nn.Module):
    def __init__(self, patch_dim, dim):
        super().__init__()
        self.net = nn.Linear(patch_dim, dim)
        
    def forward(self, X):
        x = self.net(x)

        return x  

class Embedding(nn.Module):
    def __init__(self, dim, n_patches):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, n_patches + 1, dim))

    def forward(self, x):
        batch_size, _, __ = x.shape

        cls_token = repeat(self.cls_token, "1 1 d -> b 1 d", b = batch_size)
        x = torch.concat([cls_token, x], dim=1)

        x += self.pos_embedding

        return x        


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        x = self.net(x)

        return x    


class MultiHeadAttention(nn.Module):

    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.dim_heads = dim //n_heads

        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)

        self.split_into_heads = Rearrange("b n (h d) -> b h n d", h = self.n_heads)
        self.softmax = nn.Softmax(dim = -1)
        self.concat = Rearrange("b h n d -> b n (h d)", h = self.n_heads)
    
    def forward(self, x):
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        q = self.split_into_heads(q)
        k = self.split_into_heads(k)
        v = self.split_into_heads(v)

        logit = torch.matmul(q, k.transpose(-1, -2))*(self.dim_heads**-0.5)
        attention_weight = self.softmax(logit)

        output = torch.matmul(attention_weight, v)
        output = self.concat(output)

        return output

        
class TransformerEncoder(nn.Module):

    def __init__(self, dim, n_heads, mlp_dim, depth):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.multi_head_attention = MultiHeadAttention(dim=dim, n_heads=n_heads)
        self.mlp = MLP(dim=dim, hidden_dim = mlp_dim)
        self.depth = depth

    def forward(self, x):
        for _ in range(self.depth):
            x = self.multi_head_attention(self.norm(x)) + x 
            x = self.mlp(self.norm) + x

        return x
           

class MLPHead(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.net = nn.Sequetial(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_dim)
        )

    def forward(self, x):
        x = self.net(x)
        return x    

class ViT(nn.Module):

    def __init__(self, image_size, patch_size, n_classes, dim, depth, n_heads, channels=3, mlp_dim = 256):
        super().__init__()
        n_patches = (image_size // patch_size)**2
        patch_dim = channels * patch_size ** 2
        self.depth = depth

        self.patching = Patching(patch_size = patch_size)
        self.linear_projection_of_flattened_patches = LinearProjection(patch_dim = patch_dim, dim = dim)
        self.embedding = Embedding(dim = dim, n_patches = n_patches)
        self.transformer_encoder = TransformerEncoder(dim = dim, n_heads = n_heads, mlp_dim = mlp_dim, depth = depth)
        self.mlp_head = MLPHead(dim = dim, out_dim = n_classes)

    def forward(self, img):

        x = img

        x = self.patching(x)
        x = self.linear_projection_of_flattened_patches(x)
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = x[:, 0]
        x = self.mlp_head(x)

        return x