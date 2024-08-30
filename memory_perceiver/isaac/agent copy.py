import torch
from einops import rearrange, repeat
from torch import nn, einsum
from torch.distributions import Normal, MultivariateNormal

from memoryPerceiver.memory_perceiver.core.model_net import PerceiverIO
from memoryPerceiver.memory_perceiver.processing.vectorize import input_to_tensor
from memoryPerceiver.memory_perceiver.core.model_wrapper import create_multimodal_perceiver, output_projection

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
        self.state_dim = cf.dim_observations
        self.dim_actions = cf.dim_actions
        self.dim_actions_ppo = cf.dim_actions_ppo
        self.dim_t = cf.frames_future
        self.hidden_dim = 32
        self.actor = nn.Sequential( nn.Linear(self.state_dim, self.hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(self.hidden_dim, self.hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(self.hidden_dim, 2*self.dim_actions_ppo+1)
                                    ).to(device)
        self.actor.reset_memories = lambda *args, **kwargs: None
        # self.critic = nn.Sequential(nn.Linear(self.state_dim, self.hidden_dim),
        #                             nn.ReLU(),
        #                             nn.Linear(self.hidden_dim, self.hidden_dim),
        #                             nn.ReLU(),
        #                             nn.Linear(self.hidden_dim, 1)
        #                             ).to(device)
        
    def forward(self, states, memories, old_actions=None):
        state = states['observations'][:,-1] # last time
        
        # Critic

        # values = self.critic(state)
        x = self.actor(state)
        values = x[:, 2*self.dim_actions_ppo:]
        values = rearrange(values,  'b 1 -> b')

        # Actor

        actions_tmp = torch.tanh(x[:, :2*self.dim_actions_ppo]) #torch.tanh(self.actor(state))
        actions_mean,  = actions_tmp[:, 0:self.dim_actions_ppo], 
        actions_var = ((1+actions_tmp[:, self.dim_actions_ppo:2*self.dim_actions_ppo])/4+1e-5)**2
        dist = MultivariateNormal(actions_mean, torch.diag_embed(actions_var))
        
        # Act
        if old_actions is None:  
            actions = dist.sample()
            logprobs = dist.log_prob(actions)
            return actions.detach(), logprobs.detach(), values.detach(), None
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
        self.hidden_dim = 64
        self.actor = create_multimodal_perceiver(cf, device)
        # self.critic = nn.Sequential(nn.Linear(cf.dim_observations, self.hidden_dim),
        #                             nn.ReLU(),
        #                             nn.Linear(self.hidden_dim, self.hidden_dim),
        #                             nn.ReLU(),
        #                             nn.Linear(self.hidden_dim, self.hidden_dim),
        #                             nn.ReLU(),
        #                             nn.Linear(self.hidden_dim, self.hidden_dim),
        #                             nn.ReLU(),
        #                             nn.Linear(self.hidden_dim, 1)
        #                             ).to(device)
        
        
    def forward(self, states, memories, old_actions=None):
        
        # Critic

        # values = self.critic(states['observations'][:,-1])
        # values = rearrange(values,  'b 1 -> b')

        # Actor
        
        ## Net inference
        
        device = states['observations'].device
        for i in range(self.dim_observations):
            states['observation_'+str(i)] = states['observations'][:, :, i:i+1]
        input = input_to_tensor(states, self.actor.modality, self.body_parts, device=device,
            height=self.patch_height, width=self.patch_width, bands_video=self.bands_video, bands_video_t=self.bands_video_t)
        x, new_memories, _ = self.actor(input, memories, queries=self.actor.queries)
        output = output_projection(self.actor, x, self.body_outputs, 
                          self.frames_future, self.audio_future)
        actions = []
        actions_var = []
        for i in range(self.dim_actions):
            actions.append(output['action_'+str(i)])
            actions_var.append(output['action_variance_'+str(i)])
        actions = torch.tanh(rearrange(actions, 'd b t () -> b t d'))
        actions_var = torch.tanh(rearrange(actions_var, 'd b t () -> b t d'))
        dim_t = actions.size()[1]

        # Average on time 
        actions_tmp1, actions_tmp2 = torch.split(actions, dim_t//2, dim=1)
        actions_mean = torch.cat((actions_tmp1.mean(dim=1), actions_tmp2.mean(dim=1)), dim=-1)
        actions_var_tmp1, actions_var_tmp2 = torch.split(actions_var, dim_t//2, dim=1)
        actions_var_tmp = torch.cat((actions_var_tmp1.mean(dim=1), actions_var_tmp2.mean(dim=1)), dim=-1)
        actions_var = ((1+actions_var_tmp)/2+1e-4)**2 

        dist = MultivariateNormal(actions_mean, torch.diag_embed(actions_var))

        ## result

        # Critic
        values = rearrange(output['critic'], 'b t () -> b t').mean(dim=-1)
        
        # Act
        if old_actions is None:  
            actions = dist.sample()
            logprobs = dist.log_prob(actions)
            return actions.detach(), logprobs.detach(), values.detach(), new_memories
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