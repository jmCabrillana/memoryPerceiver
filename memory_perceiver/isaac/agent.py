import torch
from einops import rearrange, repeat
from torch import nn, einsum
from torch.distributions import Normal, MultivariateNormal

from memoryPerceiver.memory_perceiver.core.model_net import PerceiverIO
from memoryPerceiver.memory_perceiver.core.miscellaneous import layer_init
from memoryPerceiver.memory_perceiver.processing.vectorize import patchify

from memoryPerceiver.memory_perceiver.core.model_net import PreNormContext, Attention, FeedForward


class ActorCriticSimple(nn.Module):
    def __init__(self, cf, device):
        super().__init__()
        self.shared=cf.shared
        self.state_dim = cf.dim_observations
        self.dim_actions = cf.dim_actions
        self.dim_actions_ppo = cf.dim_actions_ppo
        self.dim_t = cf.frames_future
        self.hidden_dim = cf.hidden_dim
        self.actor = nn.Sequential( *( [layer_init(nn.Linear(self.state_dim, self.hidden_dim)), nn.ELU()] +
                                        [layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)), nn.ELU()]*(cf.depth-1) )
                                    ).to(device)
        self.actor.reset_memories = lambda *args, **kwargs: None
        self.actor_mean = layer_init(nn.Linear(self.hidden_dim, self.dim_actions_ppo, device=device), std=0.01)
        # self.actor_std = layer_init(nn.Linear(self.hidden_dim, self.dim_actions_ppo, device=device), std=0.01)
        self.actor_std = nn.Parameter(torch.zeros(self.dim_actions_ppo, device=device))
        if self.shared:
            self.actor_critic = layer_init(nn.Linear(self.hidden_dim, 1, device=device), std=1)
        else:
            self.critic = nn.Sequential( *( [layer_init(nn.Linear(self.state_dim, self.hidden_dim)), nn.ELU()] +
                                            [layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)), nn.ELU()]*(cf.critic_depth-1) +
                                            [layer_init(nn.Linear(self.hidden_dim, 1), std=1)]  ) ).to(device)
            
        # self.embedding = nn.Linear(self.state_dim, self.hidden_dim, device=device)
        # self.latents = nn.Parameter(torch.randn(2, self.hidden_dim, device=device))
        # self.att = PreNormContext(self.hidden_dim, 
        #                 Attention(self.hidden_dim, self.hidden_dim, heads=cf.latent_heads, dim_head=cf.latent_dim_head)).to(device)
        # self.ff = PreNormContext(self.hidden_dim, FeedForward(self.hidden_dim)).to(device)
        
    def forward(self, states, memories, old_actions=None, mask=None):
        state = states['obs'][:,-1] # last time
        hidden = self.actor(state)
        # b = len(state)
        # query = rearrange(self.embedding(state), 'b d -> b 1 d')
        # x = repeat(self.latents, 'n d -> b n d', b = b)
        # attn = self.att(x, context = query)
        # x = attn + x
        # x = self.ff(x) + x
        # hidden = x.mean(dim=1)

        # Critic
        if self.shared:
            values = self.actor_critic(hidden)
        else:
            values = self.critic(state)
        values = rearrange(values,  'b 1 -> b')

        # Act
        actions_mean  = torch.tanh(self.actor_mean(hidden))
        actions_std = torch.clamp(self.actor_std, -20, 2).exp()
        dist = Normal(actions_mean, actions_std)
        actions = dist.rsample() if old_actions is None else old_actions
        logprobs = dist.log_prob(actions).sum(dim=-1)
        entropies = None if old_actions is None else dist.entropy().sum(dim=-1)
        return {'actions': actions,  'actions_mean': actions_mean, 'actions_std': actions_std,
                'logprobs': logprobs, 'values': values, 'entropies': entropies, 'memories':None}
        
    
    def instanciate_actions(self, actions, **kwargs):

        # default dt
        dt=2
        if 'dt' in kwargs:
            dt = kwargs['dt']

        # Interpolate if dt>=2 otherwise identity
        if dt >= 2:
            t = torch.linspace(0, 1, dt, device=actions.device)
            dim_d = actions.size()[-1]
            action1, action2 = torch.split(actions, dim_d//2, dim=-1)
            actions_star = einsum('t, b d -> t b d', t, action1) + einsum('t, b d -> t b d', 1-t, action2)
        else: 
            actions_star = repeat(actions, 'b d -> t b d', t=1)
        # actions_star = repeat(actions, 'b d -> t b d', t=dt) for non ppo action dim
        return actions_star
            
    
class ActorCriticPerceiver(nn.Module):
    def __init__(self, cf, device):
        super().__init__()
        self.frames_future = cf.frames_future
        self.audio_future = cf.audio_future
        self.dim_observations = cf.dim_observations
        self.dim_actions = cf.dim_actions
        self.patch_height, self.patch_width, self.bands_video, self.bands_video_t = \
            cf.patch_height, cf.patch_width, cf.bands_video, cf.bands_video_t
        self.patch_audio = cf.patch_audio
        self.hidden_dim = cf.hidden_dim
        self.shared = cf.shared
        self.inputs = cf.inputs
        self.flattened_mem = None
        input_scalars = cf.dim_observations + ('actions_star' in cf.inputs) * cf.dim_actions
        input_vectors = {}
        if 'video' in cf.inputs:
            input_vectors['video'] = cf.patch_width * cf.patch_height * cf.channels
        output_scalars = cf.dim_actions_ppo + cf.shared
        self.actor = PerceiverIO(
            cf.depth,
            input_scalars = input_scalars,
            input_vectors = input_vectors,
            output_scalars = output_scalars,
            output_vectors = {},
            depth_f_state=cf.depth_f_state,
            device=device,
            query_dim = cf.query_dim,
            nb_latents = cf.nb_latents,
            nb_latents_mem = cf.nb_latents_mem,
            latent_heads = cf.latent_heads,
            latent_dim_head = cf.latent_dim_head
        ).to(device)
        self.out_std = nn.Parameter(torch.zeros(self.dim_actions, device=device))
        if not self.shared:
            self.critic = nn.Sequential( *( [layer_init(nn.Linear(self.dim_observations, self.hidden_dim)), nn.ELU()] +
                                            [layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)), nn.ELU()]*(cf.critic_depth-1) +
                                            [layer_init(nn.Linear(self.hidden_dim, 1), std=1)]  ) ).to(device)
        else:
             nn.init.orthogonal_(self.actor.out_scalar_projections_weight[:-1], 1.)
        # for i in range(self.dim_actions):
        #     # layer_init(self.actor.out_projections['action_variance_'+str(i)], std=0.01)
        #     layer_init(self.actor.out_projections['action_'+str(i)], std=0.01)   
        # layer_init(self.actor.out_projections['critic'], std=1)          
        
    def forward(self, states, memories, old_actions=None, mask=None):
        
        # Actor
        
        ## Net inference
        
        patchify(states, height=self.patch_height, width=self.patch_width, patch_audio=self.patch_audio)
        input = {}
        if 'actions_star' in self.inputs:
            input['scalars'] = torch.cat([states['obs'], states['actions_star']], dim=-1)
        else:
            input['scalars'] = states['obs']
        if 'video' in self.inputs:
            input['video'] = states['video']
        if 'audio' in self.inputs:
            input['audio'] = states['audio']

        if self.flattened_mem is not None:
            b, l, h = self.flattened_mem
            memories = rearrange(memories, '(b l h) -> b l h', b=b, l=l, h=h)

        output, new_memories = self.actor(input, memories)
        actions_ppo = output['scalars'][:,:,:-1] if self.shared else output['scalars']
        # actions = torch.tanh(actions_ppo)
        dim_t = actions_ppo.size()[1]

        # Average on time or identity
        if dim_t >= 2:
            actions_tmp1, actions_tmp2 = torch.split(actions_ppo, dim_t//2, dim=1)
            actions_mean = torch.cat((actions_tmp1.mean(dim=1), actions_tmp2.mean(dim=1)), dim=-1)
        else : 
            actions_mean = rearrange(actions_ppo, 'b () d -> b d')

        ## result

        # Critic
        if self.shared:
            values = output['scalars'][:,:,-1].mean(dim=-1)
            # values = rearrange(output['critic'], 'b () d -> b d') # For test RTRL anbn
        else : 
            values = self.critic(states['obs'][:,-1])
            values = rearrange(values,  'b 1 -> b')
        if self.flattened_mem is not None:
            new_memories = rearrange(new_memories, 'b l h -> (b l h)')

        # Act
        actions_std = torch.clamp(self.out_std, -20, 2).exp() # [-20, 2]
        dist = Normal(actions_mean, actions_std)
        actions = dist.rsample() if old_actions is None else old_actions
        logprobs = dist.log_prob(actions).sum(dim=-1)
        entropies = None if old_actions is None else dist.entropy().sum(dim=-1)
        return {'actions': actions,  'actions_mean': actions_mean, 'actions_std': actions_std,
                'logprobs': logprobs, 'values': values, 'entropies': entropies, 'memories':None}
    
    def instanciate_actions(self, actions, **kwargs):

        # set default dt
        dt=8
        if 'dt' in kwargs:
            dt = kwargs['dt']

        # Interpolate if dt >= 2 otherwise no modif
        if dt >= 2:
            t = torch.linspace(0, 1, dt, device=actions.device)
            dim_d = actions.size()[-1]
            action1, action2 = torch.split(actions, dim_d//2, dim=-1)
            actions_star = einsum('t, b d -> t b d', t, action1) + einsum('t, b d -> t b d', 1-t, action2)
        else :
            actions_star = repeat(actions, 'b d -> t b d', t=1)
        return actions_star


def create_agent(type, cf, rank):
    if type == "Simple":
        return ActorCriticSimple(cf, rank)
    elif type == "Perceiver":
        return ActorCriticPerceiver(cf, rank)
    else:
        raise ValueError("Unknown actor type")