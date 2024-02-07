import torch
from einops import rearrange, repeat
from torch import nn, einsum
from torch.distributions import Normal, MultivariateNormal

from memoryPerceiver.memory_perceiver.core.model_net import PerceiverIO
from memoryPerceiver.memory_perceiver.processing.vectorize import input_to_tensor
import memoryPerceiver.memory_perceiver.core.model_wrapper as mw

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()

    def act(self, states):
        pass

    def evaluate(self, states, actions):
        pass

    def instanciate_actions(self, actions, **kwargs):
        pass     

class ActorCriticSimple(ActorCritic):
    def __init__(self, cf, device):
        super().__init__()
        self.state_dim = cf.body_parts['observations']
        self.action_dim = cf.body_outputs['actions']
        self.hidden_dim = 64
        self.actor = nn.Sequential( nn.Linear(self.state_dim, self.hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(self.hidden_dim, self.hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(self.hidden_dim, 2*self.action_dim)
                                    ).to(device)

        self.critic = nn.Sequential(nn.Linear(self.state_dim, self.hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(self.hidden_dim, self.hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(self.hidden_dim, 1)
                                    ).to(device)
        
    def act(self, states):
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

        return actions.detach(), logprobs.detach(), values.detach(), dist

    def evaluate(self, states, actions):
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

        return logprobs, values, entropies, dist
    
    def instanciate_actions(self, actions, **kwargs):
        dt=1
        if 'dt' in kwargs:
            dt = kwargs['dt']
        actions_star = repeat(actions, 'b d -> t b d', t=dt)
        return actions_star
    
class ActorCriticPerceiver(ActorCritic):
    def __init__(self, cf, device):
        super().__init__()
        self.cf = cf
        self.hidden_dim = 64
        self.actor = mw.create_multimodal_perceiver(cf, device)
        self.critic = nn.Sequential(nn.Linear(4, self.hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(self.hidden_dim, self.hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(self.hidden_dim, 1)
                                    ).to(device)
        
    def act(self, states):
        state = states['observations'][:,-1] # ? d=1 ou 0
        batch = input_to_tensor(states, self.actor.modality, self.cf.body_parts)
        

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

        return actions.detach(), logprobs.detach(), values.detach(), dist

    def evaluate(self, states, actions):
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

        return logprobs, values, entropies, dist
    
    def instanciate_actions(self, actions, **kwargs):
        dt=1
        if 'dt' in kwargs:
            dt = kwargs['dt']
        actions_star = repeat(actions, 'b d -> t b d', t=dt)
        return actions_star

# model = ActorCritic(4 ,1)
# model_student = ActorCritic(4 ,1)
# model.act(torch.randn(8, 4, device=device))