import torch
import torch.nn as nn
from . import clip
from torch.utils.checkpoint import checkpoint
from collections import OrderedDict
from slowfast.models.build import MODEL_REGISTRY



@MODEL_REGISTRY.register()
class BasicClip(nn.Module):
    def __init__(self, config):
        super(BasicClip, self).__init__()
        embed_dim = 512
        context_length = 77
        # vocab_size = 49408
        transformer_width = 512
        transformer_heads = transformer_width // 64
        self.num_frames = config.data.num_segments
        self.cfg = config
        self._construct_network(self.cfg)
        self.frame_position_embeddings = nn.Embedding(context_length, embed_dim)
        self.temporal_transformer = TemporalTransformer(width=embed_dim, layers=3, heads=transformer_heads)
        self.model.eval()
    
    def _construct_network(self, cfg):
        if cfg.network.arch == 'ViT-B/16':
            self.model, self.preprocess = clip.load("ViT-B/16", jit=False, )
        elif cfg.network.arch == 'ViT-B/32':
            self.model, self.preprocess = clip.load("ViT-B/32", jit=False, )
        else:
            print("error loading arch")
            exit()
        self.model.float() 
    
    def temporal_modeling(self, x):
        # x (b*t,d)
        x = x.reshape(-1, self.num_frames, x.shape[-1]) # b,t,d
        x = x.contiguous()
        b, t, d = x.size()

        x_original = x
        seq_length = t
        position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(x.size(0), -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        x = x + frame_position_embeddings

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.temporal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x.type(x_original.dtype) + x_original # b,t,d
        x = x.reshape(-1,d) #b*t,d

        return x

    def encode_image(self, x):
        img_encode = self.model.encode_image(x)
        if self.cfg.network.temporal_type == 'TemporalTransformer':
            img_encode = self.temporal_modeling(img_encode) # b*t,d
        return img_encode

    def encode_text(self, x):
        cls_encode = self.model.encode_text(x)
        return cls_encode

    
    def forward(self, image_inputs, text_inputs):
        img_encode = self.encode_image(image_inputs) # b*t,d
        cls_encode = self.encode_text(text_inputs)

        return img_encode, cls_encode


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x



class TemporalTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        self.grad_checkpointing = False

    def forward(self, x: torch.Tensor):
        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x)
            else:
                x = r(x)
        return x