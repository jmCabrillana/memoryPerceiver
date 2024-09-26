import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def full_attn(q, k, v, dropout_fn = None):
    *_, dim = q.shape
    dots = torch.einsum('b i d, b j d -> b i j', q, k) * (dim ** -0.5)
    attn = dots.softmax(dim=-1)
    if dropout_fn is not None:
        attn = dropout_fn(attn)
    return torch.einsum('b i j, b j d -> b i d', attn, v)

# helper classes

class PreNormContext(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)
    
class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.ReLU(), #GEGLU()nn.ReLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, init=False):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        if init:
            nn.init.zeros_(self.to_q.weight)
            nn.init.eye_(self.to_kv.weight)
            nn.init.eye_(self.to_out.weight)


    def forward(self, x, context = None, mask = None, use_x=True):
        h = self.heads

        q = self.to_q(x)
        if context is None:
            full_input = x
        else : 
            if not use_x:
                full_input = context
            else: 
                full_input = torch.concat([context, x], dim=1)
                
        k, v = self.to_kv(full_input).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        if exists(mask):
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
        #     max_neg_value = -torch.finfo(sim.dtype).max
        #     sim.masked_fill_(mask, max_neg_value)
        # sim = einsum('b i d, b j d -> b i j', q, k) / self.scale
        # attn = sim.softmax(dim = -1)
        # out = einsum('b i j, b j d -> b i d', attn, v)

        out = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

# main class

class PerceiverIO(nn.Module):
    def __init__(
        self,
        depth,
        depth_f_state=0,
        device='cpu',
        query_dim=832,
        nb_latents = 512,
        nb_latents_mem = 1024,
        cross_heads = 1,
        latent_heads = 8,
        latent_dim_head = 64,
        input_parameters = None,
        output_parameters = None,
    ):
        super().__init__()
        latent_dim = query_dim
        self.query_dim = query_dim
        self.nb_latents = nb_latents
        self.nb_latents_mem = nb_latents_mem
        self.depth = depth
        self.depth_f_state = depth_f_state
        self.pos_encoding=None

        # F_output

        self.out_embedding = nn.ModuleDict({})
        for part, params in input_parameters.items():
            self.out_embedding[part] = nn.Linear(params, query_dim, device=device)
        self.out_latents = nn.Parameter(torch.randn(nb_latents, latent_dim, device=device))
        self.out_cross_attn = PreNormContext(latent_dim, Attention(latent_dim, query_dim, heads=cross_heads, dim_head=latent_dim_head), query_dim)
        self.out_cross_ff =  FeedForward(latent_dim) #PreNormContext(latent_dim, FeedForward(latent_dim))
        self.out_attn_layers = nn.ModuleList([])
        self.out_ff_layers = nn.ModuleList([])
        for i in range(self.depth):
            self.out_attn_layers.append(PreNormContext(latent_dim, Attention(latent_dim, query_dim, heads=latent_heads, dim_head=latent_dim_head), query_dim))
            self.out_ff_layers.append(PreNormContext(latent_dim, FeedForward(latent_dim)))
        self.out_decoder_cross_attn = PreNormContext(query_dim, Attention(query_dim, latent_dim, heads = cross_heads, dim_head = latent_dim_head), context_dim = query_dim)
        self.modality = {}
        for part, params in output_parameters.items():
            self.modality[part] = nn.Parameter(torch.randn(query_dim, device=device))
        self.queries=None

        # F_state

        self.state_embedding = nn.ModuleDict({})
        for part, params in input_parameters.items():
            self.state_embedding[part] = nn.Linear(params, query_dim, device=device)
        self.state_latents = nn.Parameter(torch.randn(nb_latents_mem, query_dim, device=device))
        self.state_cross_attn = PreNormContext(query_dim, Attention(query_dim, query_dim, heads = cross_heads, dim_head = latent_dim_head, init=True), query_dim)
        self.state_cross_ff = PreNormContext(query_dim, FeedForward(query_dim))
        self.state_attn_layers = nn.ModuleList([])
        self.state_ff_layers = nn.ModuleList([])
        for i in range(self.depth_f_state):
            self.state_attn_layers.append(PreNormContext(query_dim, Attention(query_dim, heads = latent_heads, dim_head = latent_dim_head)))
            self.state_ff_layers.append( PreNormContext(query_dim, FeedForward(query_dim)))

        # Projection to return modalities

        self.out_projections = nn.ModuleDict({})
        for part, params in output_parameters.items():
            self.out_projections[part] = nn.Linear(query_dim, params, device=device)

    def forward(self, data, layers_mem, mask = None, queries = None):

        dummy = next(iter(data.values()))
        b, device = dummy.shape[0], dummy.device

        # Compute output

        ## Embedding
        out_context = torch.tensor([], device=device)
        for key, value in data.items():
            context = self.out_embedding[key](value) + repeat(self.pos_encoding[key], '... -> b ...', b=b)
            out_context = torch.cat([out_context, context], dim=1)

        ## Cross attention on input
        x = repeat(self.out_latents, 'n d -> b n d', b = b)
        attn = self.out_cross_attn(x, context = torch.cat([out_context, layers_mem], dim=1), mask = mask)
        # data = data.cpu() # to save gpu memory
        x = attn + x
        x = self.out_cross_ff(x) + x

        ## layers
        for i in range(1, self.depth):
            attn = self.out_attn_layers[i](x, mask=mask)
            x = attn + x
            x = self.out_ff_layers[i](x) + x

        ## cross attend from decoder queries to latents
        if queries.ndim == 2:
            queries = repeat(queries, 'n d -> b n d', b = b)
        output = self.out_decoder_cross_attn(queries, context = x)
        
        # Compute next state

        # Embedding
        state_context = torch.tensor([], device=device)
        for key, value in data.items():
            context = self.state_embedding[key](value) + repeat(self.pos_encoding[key], '... -> b ...', b=b)
            state_context = torch.cat([state_context, context], dim=1)

        ## Cross attention on input # context = context.cpu() # to save gpu memory
        # x = repeat(self.state_latents, 'n d -> b n d', b = b)
        x = layers_mem
        attn = self.state_cross_attn(x, context = state_context, mask = mask)
        x = attn + x
        x = self.state_cross_ff(x) + x

        ## layers
        for i in range(self.depth_f_state):
            attn = self.state_attn_layers[i](x, mask=mask)
            x = attn + x
            x = self.state_ff_layers[i](x) + x

        ## Update mem
        new_mem = x.mean(dim=1) # 'b n d -> b d'
        new_mem = rearrange(new_mem, 'b d0 -> b () d0')
        new_layers_mem = torch.cat([layers_mem[:, :-1], new_mem], dim=1)
        
        return output, new_layers_mem
