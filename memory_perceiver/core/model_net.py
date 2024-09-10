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

class PreNormMem(nn.Module):
    def __init__(self, dim, fn, mem_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_mem = nn.LayerNorm(mem_dim) if exists(mem_dim) else None
        self.norm_cmem = nn.LayerNorm(mem_dim) if exists(mem_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_mem):
            mem = kwargs['mem']
            normed_mem = self.norm_mem(mem)
            kwargs.update(mem = normed_mem)

        if exists(self.norm_cmem):
            cmem = kwargs['cmem']
            normed_cmem = self.norm_cmem(cmem)
            kwargs.update(cmem = normed_cmem)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.ReLU(), #GEGLU()nn.ReLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) / self.scale

        if exists(mask):
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

# Core

class AttentionMemory(nn.Module):
    def __init__(self, query_dim, mem_dim = None, heads = 8, dim_head = 64, reconstruction_attn_dropout = 0):
        super().__init__()
        inner_dim = dim_head * heads
        mem_dim = default(mem_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(mem_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.compress_mem = Attention(mem_dim)
        self.reconstruction_attn_dropout = nn.Dropout(reconstruction_attn_dropout)

    def forward(self, x, mem = None, cmem = None, mask = None, use_x = True, nb_compressed = 1, special_compression=None):
        h = self.heads
        _, cmem_len, _ = cmem.shape

        """
        move new_cmem, new_mem before
        to improve memory
        """

        # New memories

        if( special_compression is None ):
            new_mem = torch.cat([mem[:, nb_compressed:, :], x[:, 0:nb_compressed, :]], dim=1)
            to_compress = mem[:, 0:nb_compressed, :]
        else:
            new_mem = mem
            to_compress = mem[:, special_compression, :]
        # new_cmem = self.compress_mem(cmem, to_compress)
        new_cmem = cmem

        q = self.to_q(x)
        if(use_x):
            if special_compression is not None:
                full_input = torch.concat([mem, x], dim=1)
            else: 
                full_input = x
        else:
            full_input = torch.concat([cmem, mem], dim=1)
        k, v = self.to_kv(full_input).chunk(2, dim = -1)

        # Merge heads

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) / self.scale

        if exists(mask):
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(mask, max_neg_value)

        # Attention, what we cannot get enough of

        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)

        # Auxiliary loss
        
        self.to_kv.weight.detach()

        cmem_k, cmem_v = self.to_kv(new_cmem.detach()).chunk(2, dim=-1)
        cmem_k, cmem_v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (cmem_k, cmem_v))

        old_mem_k, old_mem_v = map(lambda x: x[:, :cmem_len+nb_compressed, :].clone(), (k, v))

        q, old_mem_k, old_mem_v = map(torch.detach, (q, old_mem_k, old_mem_v))
        out_cmem = full_attn(q, cmem_k, cmem_v, self.reconstruction_attn_dropout)
        out_old_mem = full_attn(q, old_mem_k, old_mem_v, self.reconstruction_attn_dropout)
        out_cmem, out_old_mem = map(lambda t: rearrange(t, '(b h) n d -> b n (h d)', h = h), (out_cmem, out_old_mem))
        aux_loss = F.mse_loss(
            out_cmem,
            out_old_mem
        )
        return self.to_out(out), mem, cmem, aux_loss # new_mem, new_cmem, aux_loss

# main class

class PerceiverIO(nn.Module):
    def __init__(
        self,
        depth,
        queries_dim,
        device,
        nb_latents = 512,
        cross_heads = 1,
        latent_heads = 8,
        cross_dim_head = 64,
        latent_dim_head = 64,
        mem_size = 1024,
        cmem_size = 1024,
        special_compression = None,
        nb_compressed = 512,
        modality_parameters = None,
        projection_parameters = None,
    ):
        super().__init__()
        latent_dim = queries_dim
        self.queries_dim = queries_dim
        self.mem_size = mem_size
        self.cmem_size = cmem_size
        self.depth = depth
        self.latents = nn.Parameter(torch.randn(nb_latents, latent_dim, device=device))
        self.special_compression = special_compression
        self.nb_compressed = nb_compressed

        self.cross_attn = PreNormMem(latent_dim, 
                                Attention(latent_dim, 
                                                queries_dim, 
                                                heads = cross_heads, 
                                                dim_head = cross_dim_head))
        self.cross_ff = PreNormContext(latent_dim, FeedForward(latent_dim))

        self.attn_layers = nn.ModuleList([])
        self.ff_layers = nn.ModuleList([])
        for i in range(self.depth):
            if ( i % 2 == 0):
                self.attn_layers.append(
                    PreNormMem(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head)),
                )
            if ( i % 2 == 1):
                self.attn_layers.append(
                    PreNormMem(latent_dim, AttentionMemory(latent_dim, heads = latent_heads, dim_head = latent_dim_head)),
                )
            self.ff_layers.append(
                PreNormContext(latent_dim, FeedForward(latent_dim)),
            )

        self.decoder_cross_attn = PreNormContext(queries_dim, Attention(queries_dim, latent_dim, heads = cross_heads, dim_head = cross_dim_head), context_dim = latent_dim)
        self.queries=None

        # Projection to return modalities, enhancement

        self.modality = {}
        for part, params in modality_parameters.items():
            self.modality[part] = nn.Parameter(torch.randn(params, device=device))

        self.projections = nn.ModuleDict({})
        for part, params in projection_parameters.items():
            self.projections[part] = nn.Linear(queries_dim, params, device=device)

    def forward(self, data, memories, mask = None, queries = None):

        # init data

        b, *_, device, dtype = *data.shape, data.device, data.dtype
        data.to(device)
        sum_aux_loss = torch.tensor(0, requires_grad = True, device=device, dtype=dtype)
        x = repeat(self.latents, 'n d -> b n d', b = b)

        # Cross attention on input

        attn = self.cross_attn(x, context = data, mask = mask)
        data = data.cpu() # to save gpu memory
        x = attn + x
        x = self.cross_ff(x) + x

        # layers

        new_layers_mem = torch.zeros(b, self.depth//2, self.mem_size, self.queries_dim, device=device)
        new_layers_cmem = torch.zeros(b, self.depth//2, self.cmem_size, self.queries_dim, device=device)
        for i in range(self.depth):
            if ( i % 2 == 0):
                attn = self.attn_layers[i](x, mask=mask)
            if ( i % 2 == 1):
                attn, new_mem, new_cmem, aux_loss = self.attn_layers[i](x, 
                                                                        mem = memories['layers_mem'][:, i//2].clone().detach(), 
                                                                        cmem = memories['layers_cmem'][:, i//2].clone().detach(),
                                                                        nb_compressed = self.nb_compressed,
                                                                        mask=mask
                                                                        )
                mem=memories['layers_mem'][:, i//2].clone().detach().sum()
                cmem=memories['layers_mem'][:, i//2].clone().detach().sum()
                sum_aux_loss= sum_aux_loss + aux_loss
            x = attn + x
            x = self.ff_layers[i](x) + x

        if not exists(queries):
            return x, sum_aux_loss

        # make sure queries contain batch dimension

        queries = queries.to(device)
        if queries.ndim == 2:
            queries = repeat(queries, 'n d -> b n d', b = b)

        # cross attend from decoder queries to latents

        output = self.decoder_cross_attn(queries, context = x)

        # final linear out

        memories = {'cross_cmem':memories['cross_cmem'], 'layers_mem':new_layers_mem, 'layers_cmem':new_layers_cmem}
        
        return output, memories, sum_aux_loss
