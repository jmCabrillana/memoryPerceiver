import torch
from einops import rearrange, repeat
from torch import nn, einsum
from torch.distributions import Normal, MultivariateNormal

from memoryPerceiver.memory_perceiver.core.model_net import PerceiverIO
from memoryPerceiver.memory_perceiver.core.model_wrapper import create_multimodal_perceiver, output_projection
from memoryPerceiver.memory_perceiver.core.miscellaneous import layer_init
from memoryPerceiver.memory_perceiver.processing.vectorize import patchify

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, states, memories, old_actions=None):
        pass

    def instanciate_actions(self, actions, **kwargs):
        pass     

class ActorCriticSimple(ActorCritic):
    def __init__(self, cf, device):
        super().__init__()
        self.shared=cf.shared
        self.state_dim = cf.dim_observations
        self.dim_actions = cf.dim_actions
        self.dim_actions_ppo = cf.dim_actions_ppo
        self.dim_t = cf.frames_future
        self.hidden_dim = cf.hidden_dim
        self.actor = nn.Sequential( *( [layer_init(nn.Linear(self.state_dim, self.hidden_dim)), nn.ReLU()] +
                                        [layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)), nn.ReLU()]*(cf.depth-1) )
                                    ).to(device)
        self.actor.reset_memories = lambda *args, **kwargs: None
        self.actor_mean = layer_init(nn.Linear(self.hidden_dim, self.dim_actions_ppo, device=device), std=0.01)
        self.actor_std = layer_init(nn.Linear(self.hidden_dim, self.dim_actions_ppo, device=device), std=0.01)
        if self.shared:
            self.actor_critic = layer_init(nn.Linear(self.hidden_dim, 1, device=device), std=1)
        else:
            self.critic = nn.Sequential( *( [layer_init(nn.Linear(self.state_dim, self.hidden_dim)), nn.ReLU()] +
                                            [layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)), nn.ReLU()]*(cf.critic_depth-1) +
                                            [layer_init(nn.Linear(self.hidden_dim, 1), std=1)]  ) ).to(device)
        
    def forward(self, states, memories, old_actions=None):
        state = states['observations'][:,-1] # last time
        hidden = self.actor(state)
        
        # Critic

        if self.shared:
            values = self.actor_critic(hidden)
        else:
            values = self.critic(state)
        values = rearrange(values,  'b 1 -> b')

        # Actor

        actions_mean  = torch.tanh(self.actor_mean(hidden))
        log_std = self.actor_std(hidden).clip(-20, 2) # ((1+actions_tmp[:, self.dim_actions_ppo:2*self.dim_actions_ppo])/4+1e-5)**2
        covariance = torch.diag_embed(log_std.exp() * log_std.exp())
        dist = MultivariateNormal(actions_mean, covariance) # scale_tril=covariance) # torch.diag(...)
        
        # Act
        if old_actions is None:  
            actions = dist.sample()
            logprobs = dist.log_prob(actions)
            return actions.detach(), values.detach(), logprobs.detach(), None
        # Evaluate
        else:  
            logprobs = dist.log_prob(old_actions)
            entropies = dist.entropy() 
            return logprobs, values, entropies, None
    
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
            
    
class ActorCriticPerceiver(ActorCritic):
    def __init__(self, cf, device):
        super().__init__()
        self.frames_future = cf.frames_future
        self.audio_future = cf.audio_future
        self.dim_observations = cf.dim_observations
        self.dim_actions = cf.dim_actions
        self.body_parts = cf.body_parts
        self.body_outputs = cf.body_outputs
        self.patch_height, self.patch_width, self.bands_video, self.bands_video_t = \
            cf.patch_height, cf.patch_width, cf.bands_video, cf.bands_video_t
        self.patch_audio = cf.patch_audio
        self.hidden_dim = cf.hidden_dim
        self.shared = cf.shared
        self.flattened_mem = None
        self.actor = create_multimodal_perceiver(cf, device)
        if not self.shared:
            self.critic = nn.Sequential( *( [layer_init(nn.Linear(self.dim_observations, self.hidden_dim)), nn.ReLU()] +
                                            [layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)), nn.ReLU()]*(cf.critic_depth-1) +
                                            [layer_init(nn.Linear(self.hidden_dim, 1), std=1)]  ) ).to(device)
        for i in range(self.dim_actions):
            layer_init(self.actor.out_projections['action_variance_'+str(i)], std=0.01)
            layer_init(self.actor.out_projections['action_'+str(i)], std=0.01)   
            layer_init(self.actor.out_projections['critic'], std=1)          
        
    def forward(self, states, memories, old_actions=None):
        
        # Actor
        
        ## Net inference
        
        device = states['observations'].device
        states_chunk= {}
        for i in range(self.dim_observations):
            states_chunk['observation_'+str(i)] = states['observations'][:, :, i:i+1]
        # input = input_embedding(self.actor, states_chunk, device=device,
        #         )
        input = patchify(states_chunk, height=self.patch_height, width=self.patch_width, patch_audio=self.patch_audio)
        if self.flattened_mem is not None:
            b, l, h = self.flattened_mem
            memories = rearrange(memories, '(b l h) -> b l h', b=b, l=l, h=h)
        x, new_memories = self.actor(input, memories, queries=self.actor.queries)
        output = output_projection(self.actor, x, self.frames_future, self.audio_future)

        actions = []
        actions_var = []
        for i in range(self.dim_actions):
            actions.append(output['action_'+str(i)])
            actions_var.append(output['action_variance_'+str(i)])
        actions = torch.tanh(rearrange(actions, 'd b t () -> b t d'))
        actions_var = rearrange(actions_var, 'd b t () -> b t d')
        dim_t = actions.size()[1]

        # Average on time or identity
        if dim_t >= 2:
            actions_tmp1, actions_tmp2 = torch.split(actions, dim_t//2, dim=1)
            actions_mean = torch.cat((actions_tmp1.mean(dim=1), actions_tmp2.mean(dim=1)), dim=-1)
            actions_var_tmp1, actions_var_tmp2 = torch.split(actions_var, dim_t//2, dim=1)
            actions_var_tmp = torch.cat((actions_var_tmp1.mean(dim=1), actions_var_tmp2.mean(dim=1)), dim=-1)
        else : 
            actions_mean = rearrange(actions, 'b () d -> b d')
            actions_var_tmp = rearrange(actions_var, 'b () d -> b d')
        log_std = actions_var_tmp.clip(-20, 2) # actions_var = ((1+actions_var_tmp)/4+1e-5)**2
        covariance = torch.diag_embed(log_std.exp() * log_std.exp())
        dist = MultivariateNormal(actions_mean, covariance)

        ## result

        # Critic
        if self.shared:
            values = rearrange(output['critic'], 'b t () -> b t').mean(dim=-1)
            # values = rearrange(output['critic'], 'b () d -> b d') # For test RTRL anbn
        else : 
            values = self.critic(states['observations'][:,-1])
            values = rearrange(values,  'b 1 -> b')
        
        # Act
        if self.flattened_mem is not None:
            new_memories = rearrange(new_memories, 'b l h -> (b l h)')
        if old_actions is None:  
            actions = dist.sample()
            logprobs = dist.log_prob(actions)
            return actions.detach(), values.detach(), logprobs.detach(), new_memories
        # Evaluate
        else:  
            logprobs = dist.log_prob(old_actions)
            entropies = dist.entropy() 
            return logprobs, values, entropies, new_memories
    
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


def create_agent(cf, rank):
    if cf.agent_type == "Simple":
        return ActorCriticSimple(cf, rank)
    elif cf.agent_type == "Perceiver":
        return ActorCriticPerceiver(cf, rank)
    else:
        raise ValueError("Unknown actor type")