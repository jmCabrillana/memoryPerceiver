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

    def act(self, states, memories):
        pass

    def evaluate(self, states, actions, memories):
        pass

    def instanciate_actions(self, actions, **kwargs):
        pass     

class ActorCriticSimple(ActorCritic):
    def __init__(self, cf, device):
        super().__init__()
        self.state_dim = cf.dim_observations
        self.action_dim = cf.dim_actions
        self.hidden_dim = 64
        self.actor = nn.Sequential( nn.Linear(self.state_dim, self.hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(self.hidden_dim, self.hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(self.hidden_dim, 2*self.action_dim)
                                    ).to(device)
        self.actor.reset_memories = lambda *args, **kwargs: None
        self.critic = nn.Sequential(nn.Linear(self.state_dim, self.hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(self.hidden_dim, self.hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(self.hidden_dim, 1)
                                    ).to(device)
        
    def act(self, states, memories):
        state = states['observations'][:,-1] # last time

        # Actor

        actions_tmp = torch.tanh(self.actor(state))
        actions_mean,  = actions_tmp[:, 0:self.action_dim], 
        actions_var = ((1+actions_tmp[:, self.action_dim:2*self.action_dim+1])/4+1e-5)**2
        dist = MultivariateNormal(actions_mean, torch.diag_embed(actions_var))
        actions = dist.sample()
        logprobs = dist.log_prob(actions)

        # Critic

        values = self.critic(state)
        values = rearrange(values,  'b 1 -> b')

        return actions.detach(), logprobs.detach(), values.detach(), dist, None

    def evaluate(self, states, actions, memories):
        state = states['observations'][:,-1] # ? d=1 ou 0

        # Actor

        actions_tmp = torch.tanh(self.actor(state))
        actions_mean,  = actions_tmp[:, 0:self.action_dim], 
        actions_var = ((1+actions_tmp[:, self.action_dim:2*self.action_dim+1])/4+1e-5)**2
        dist = MultivariateNormal(actions_mean, torch.diag_embed(actions_var))
        logprobs = dist.log_prob(actions)
        entropies = dist.entropy() 

        # Critic

        values = self.critic(state)
        values = rearrange(values,  'b 1 -> b')

        return logprobs, values, entropies, dist, None
    
    def instanciate_actions(self, actions, **kwargs):
        dt=1
        if 'dt' in kwargs:
            dt = kwargs['dt']
        actions_star = repeat(actions, 'b d -> t b d', t=dt)
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
        self.hidden_dim = 64
        self.actor = create_multimodal_perceiver(cf, device)
        self.critic = nn.Sequential(nn.Linear(cf.dim_observations, self.hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(self.hidden_dim, self.hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(self.hidden_dim, self.hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(self.hidden_dim, 1)
                                    ).to(device)
        

    def net_inference(self, states, memories):
        device = states['observations'].device
        for i in range(self.dim_observations):
            states['observation_'+str(i)] = states['observations'][:, :, i:i+1]
        input = input_to_tensor(states, self.actor.modality, self.body_parts, device=device)
        x, new_memories, _ = self.actor(input, memories, queries=self.actor.queries)
        output = output_projection(self.actor, x, self.body_outputs, 
                          self.frames_future, self.audio_future)
        actions = []
        actions_var = []
        for i in range(self.dim_actions):
            actions.append(output['action_'+str(i)])
            actions_var.append(output['action_variance_'+str(i)])
        actions = rearrange(actions, 'd b t () -> b t d')
        actions_var = rearrange(actions_var, 'd b t () -> b t d')
        dim_t = actions.size()[1]
        actions_tmp1, actions_tmp2 = torch.split(torch.tanh(actions), dim_t//2, dim=1)
        actions_mean = torch.cat((actions_tmp1.mean(dim=1), actions_tmp2.mean(dim=1)), dim=-1)
        actions_var_tmp1, actions_var_tmp2 = torch.split(torch.tanh(actions_var), dim_t//2, dim=1)
        actions_var_tmp = torch.cat((actions_var_tmp1.mean(dim=1), actions_var_tmp2.mean(dim=1)), dim=-1)
        actions_var = ((1+actions_var_tmp)/4+1e-5)**2
        dist = MultivariateNormal(actions_mean, torch.diag_embed(actions_var))
        return dist, new_memories
        
    def act(self, states, memories):
        
        # Critic

        values = self.critic(states['observations'][:,-1])
        values = rearrange(values,  'b 1 -> b')

        # Actor

        dist, new_memories = self.net_inference(states)
        actions = dist.sample()
        logprobs = dist.log_prob(actions)

        return actions.detach(), logprobs.detach(), values.detach(), dist, new_memories

    def evaluate(self, states, actions, memories):
        
        # Critic

        values = self.critic(states['observations'][:,-1])
        values = rearrange(values,  'b 1 -> b')

        # Actor

        dist, new_memories = self.net_inference(states, memories)
        logprobs = dist.log_prob(actions)
        entropies = dist.entropy() 

        return logprobs, values, entropies, dist, new_memories
    
    def instanciate_actions(self, actions, **kwargs):
        dt=8
        if 'dt' in kwargs:
            dt = kwargs['dt']
        t = torch.linspace(0, 1, dt, device=actions.device)
        dim_d = actions.size()[-1]
        action1, action2 = torch.split(actions, dim_d//2, dim=-1)
        actions_star = einsum('t, b d -> t b d', t, action1) + einsum('t, b d -> t b d', 1-t, action2)
        return actions_star