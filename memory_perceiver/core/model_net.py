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

def layer_init(layer, std=1.4, bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def fourier_encode_vector(x, max_freq, num_bands):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq / 2, num_bands, device = device, dtype = dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * torch.pi
    x = torch.cat([x.sin(), x.cos()], dim = -1)
    x = torch.cat((x, orig_x), dim = -1)
    return x

def positional_encoding(data, max_freq=1000, num_bands=64):
    b, *axis, d, device = *data.shape, data.device
    axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=device), axis))
    pos = torch.stack(torch.meshgrid(*axis_pos, indexing = 'ij'), dim = -1)
    enc_pos = fourier_encode_vector(pos, max_freq, num_bands)
    enc_pos = rearrange(enc_pos, '... n b -> ... (n b)') # out dim is K*(2b+1), k=nb axis, b=bands
    enc_pos = F.pad(enc_pos, (0, d - enc_pos.shape[-1]))
    return enc_pos

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

class LinearSplit(nn.Module):
    def __init__(self, d_in, d_out, splits=8):
        super().__init__()
        assert(d_in % splits == 0)
        self.splits = splits
        d_in_split = d_in//splits
        self.weights =  nn.ParameterList( [nn.Parameter(torch.randn(d_in_split, d_out)) for _ in range(splits)] )
        self.bias = nn.Parameter(torch.randn(d_out))

    def forward(self, x):
        w = torch.cat(list(self.weights))
        return torch.einsum('i j, ... i -> ... j', w, x) + self.bias


class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.ReLU(), #GEGLU() nn.ReLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class MultiEmbedding(nn.Module):
    def __init__(self, input_scalars, input_vectors, dim, device):
        super().__init__()
        self.input_scalars = input_scalars
        self.out_scalar_embeddings_weight = nn.Parameter(torch.randn(input_scalars, dim, device=device))
        self.out_scalar_embeddings_bias = nn.Parameter(torch.zeros(input_scalars, dim, device=device))
        nn.init.orthogonal_(self.out_scalar_embeddings_weight, 1.4)
        self.out_vector_embedding = nn.ModuleDict({})
        self.dim = dim
        for part, params in input_vectors.items():
            self.out_vector_embedding[part] = nn.Linear(params, dim, device=device)
            layer_init(self.out_vector_embedding[part])
        self.scalar_pos = None
        self.vector_pos = {}

    def forward(self, x):
        scalar_embedding = torch.einsum('b t n, n d -> b t n d', x['scalars'], self.out_scalar_embeddings_weight) + self.out_scalar_embeddings_bias
        if self.scalar_pos is None:
            self.scalar_pos = positional_encoding(scalar_embedding[:, :, 0, :])
        scalar_embedding = rearrange(scalar_embedding, 'b t n d -> b n t d') + self.scalar_pos # rearrange for broadcast
        scalar_embedding = rearrange(scalar_embedding, 'b n t d -> b (n t) d')
        
        out = scalar_embedding
        for key, value in x.items():
            if key != 'scalars':
                vector_embedding = self.out_vector_embedding[key](value)
                if key not in self.vector_pos:
                    self.vector_pos[key] = positional_encoding(vector_embedding)
                vector_embedding += self.vector_pos[key]
                vector_embedding = rearrange(vector_embedding, 'b ... d -> b (...) d')
                out = torch.cat([out, vector_embedding], dim=1)
        return out
    
class MultiQuery(nn.Module):
    def __init__(self, query_scalars, query_vectors, dim, device):
        super().__init__()
        self.input_scalars = query_scalars
        self.query_vectors = query_vectors
        self.out_scalar_embeddings_bias = nn.Parameter(torch.zeros(query_scalars, dim, device=device))
        self.dim = dim
        self.out_vector_embedding = {}
        for key in query_vectors.keys():
            self.out_vector_embedding[key] = nn.Parameter(torch.zeros(dim, device=device))
        self.scalar_pos = None
        self.vector_pos = {}
        self.device=device

    def forward(self, seq_len):
        scalar_embedding = self.out_scalar_embeddings_bias
        if self.scalar_pos is None:
            self.scalar_pos = positional_encoding(torch.zeros(1, seq_len, self.dim, device=self.device))
        scalar_embedding = repeat(self.out_scalar_embeddings_bias, 'n d -> n t d', t=seq_len) + self.scalar_pos
        scalar_embedding = rearrange(scalar_embedding, 'n t d -> (n t) d')
        
        query = scalar_embedding
        for key, _ in self.query_vectors.items():
            if key != 'scalars':
                if key not in self.vector_pos:
                    self.vector_pos[key] = positional_encoding(torch.zeros(1, seq_len, self.dim, device=self.device))
                vector_embedding = self.out_vector_embedding[key] + self.vector_pos[key]
                vector_embedding = rearrange(vector_embedding, '... d -> (...) d')
                query = torch.cat([query, vector_embedding], dim=0)
        return query
    
class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, init=False):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias = False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        if init:
            nn.init.zeros_(self.to_q.weight)
            nn.init.eye_(self.to_k.weight)
            nn.init.eye_(self.to_v.weight)
            nn.init.eye_(self.to_out.weight)


    def forward(self, x, context = None, mask = None, use_x=False):
        h = self.heads

        q = self.to_q(x)
        if context is None:
            full_input = x
        else : 
            if not use_x:
                full_input = context
            else: 
                full_input = torch.concat([context, x], dim=1)
                
        k = self.to_k(full_input)
        v = self.to_v(full_input)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        # if exists(mask):
        #     mask = repeat(mask, 'b j -> (b h) () j', h = h)
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
        input_scalars = 1,
        input_vectors = {},
        output_scalars = 1,
        output_vectors = {},
        depth_f_state=0,
        device='cpu',
        query_dim = 832,
        nb_latents = 512,
        nb_latents_mem = 1024,
        latent_heads = 8,
        latent_dim_head = 64
    ):
        super().__init__()
        latent_dim = query_dim
        self.query_dim = query_dim
        self.nb_latents = nb_latents
        self.nb_latents_mem = nb_latents_mem
        self.depth = depth
        self.depth_f_state = depth_f_state
        self.output_scalars = output_scalars
        self.output_vectors = output_vectors
        
        # F_output

        self.out_embedding = MultiEmbedding(input_scalars, input_vectors, query_dim, device=device)
        self.out_latents = nn.Parameter(torch.randn(nb_latents, latent_dim, device=device))
        self.out_cross_attn = PreNormContext(latent_dim, Attention(latent_dim, query_dim, heads=latent_heads, dim_head=latent_dim_head), query_dim)
        self.out_cross_ff =  PreNormContext(latent_dim, FeedForward(latent_dim))
        self.out_attn_layers = nn.ModuleList([])
        self.out_ff_layers = nn.ModuleList([])
        for i in range(self.depth):
            self.out_attn_layers.append(PreNormContext(latent_dim, Attention(latent_dim, query_dim, heads=latent_heads, dim_head=latent_dim_head)))
            self.out_ff_layers.append(PreNormContext(latent_dim, FeedForward(latent_dim)))
        self.out_decoder_cross_attn = PreNormContext(query_dim, Attention(query_dim, latent_dim, heads = latent_heads, dim_head = latent_dim_head), context_dim = query_dim)


        # F_state

        self.state_embedding = MultiEmbedding(input_scalars, input_vectors, query_dim, device=device)
        self.state_latents = nn.Parameter(torch.randn(nb_latents_mem, query_dim, device=device))
        self.state_cross_attn = PreNormContext(query_dim, Attention(query_dim, query_dim, heads = latent_heads, dim_head = latent_dim_head, init=True), query_dim)
        self.state_cross_ff = PreNormContext(query_dim, FeedForward(query_dim))
        self.state_attn_layers = nn.ModuleList([])
        self.state_ff_layers = nn.ModuleList([])
        for i in range(self.depth_f_state):
            self.state_attn_layers.append(PreNormContext(query_dim, Attention(query_dim, heads = latent_heads, dim_head = latent_dim_head)))
            self.state_ff_layers.append( PreNormContext(query_dim, FeedForward(query_dim)))

        # Projection to return modalities

        self.out_query = MultiQuery(output_scalars, output_vectors, query_dim, device=device)
        self.out_scalar_projections_weight = nn.Parameter(torch.randn(output_scalars, query_dim, device=device))
        nn.init.orthogonal_(self.out_scalar_projections_weight, 0.01)
        self.out_scalar_projections_bias = nn.Parameter(torch.zeros(output_scalars, device=device))
        self.out_projections = nn.ModuleDict({})
        for part, params in output_vectors.items():
            self.out_projections[part] = nn.Linear(query_dim, params, device=device)
            layer_init(self.out_projections[part], std=0.01)

    def forward(self, data, mem, mask = None):

        dummy = next(iter(data.values()))
        b, device = dummy.shape[0], dummy.device

        # Compute output

        ## Modality and position embedding
        out_context = self.out_embedding(data)

        ## Cross attention on input
        x = repeat(self.out_latents, 'n d -> b n d', b = b)
        attn = self.out_cross_attn(x, context = torch.cat([out_context, mem], dim=1), mask = mask)
        # data = data.cpu() # to save gpu memory
        x = attn + x
        x = self.out_cross_ff(x) + x

        ## layers
        for i in range(1, self.depth):
            attn = self.out_attn_layers[i](x)
            x = attn + x
            x = self.out_ff_layers[i](x) + x

        # print(x[0].mean().item(),x[1].mean().item())
        ## cross-attend query and project outputs
        seq_len = 1
        queries = repeat(self.out_query(seq_len), 'n d -> b n d', b = b)
        out = self.out_decoder_cross_attn(queries, context = x)

        output = {}
        output_scalars = rearrange(out[:, :seq_len * self.output_scalars, :], 'b (n t) d -> b t n d', n=self.output_scalars)
        output['scalars'] = torch.einsum('b t n d, n d -> b t n', output_scalars, self.out_scalar_projections_weight) + self.out_scalar_projections_bias
        seq_pos = seq_len * self.output_scalars
        for key, _ in self.output_vectors.items():
            output[key] = self.out_projections[key](out[:, seq_pos: seq_pos+seq_len, :])
            seq_pos += seq_len
        
        # Compute next state

        # Modality and position embedding
        state_context = self.state_embedding(data)

        ## Cross attention on input 
        # context = context.cpu() # to save gpu memory
        # x = repeat(self.state_latents, 'n d -> b n d', b = b)
        x = mem
        attn = self.state_cross_attn(x, context = state_context, mask = mask)
        x = attn + x
        x = self.state_cross_ff(x) + x

        ## layers
        for i in range(self.depth_f_state):
            attn = self.state_attn_layers[i](x)
            x = attn + x
            x = self.state_ff_layers[i](x) + x

        ## Update mem
        new_mem = x.mean(dim=1) # 'b n d -> b d'
        new_mem = rearrange(new_mem, 'b d0 -> b () d0')
        new_mem = torch.cat([mem[:, 1:], new_mem], dim=1)
        
        return output, new_mem

def reset_memories(memories, indexes=None, mem_dim=None):

    # Reset memory
    with torch.no_grad():
        if indexes is None:
            memories *= 0
        else:
            if mem_dim is not None:
                b, l, h = mem_dim
                indexes = sum((list(range(bi*l*h, (bi+1)*l*h)) for bi in indexes), [])
            memories[indexes] *= 0

def copy_memories(layers_mem):
    device = layers_mem.device
    return layers_mem.clone().detach().to(device)

def create_memories(model, batch_size, device=torch.device('cpu')):
    layers_mem = torch.zeros(batch_size, model.nb_latents_mem, model.query_dim,
                              device=device, requires_grad=True)
    return layers_mem