import torch
from einops import rearrange, repeat
from torch import nn, einsum
from torch.distributions import Normal, MultivariateNormal

from memoryPerceiver.memory_perceiver.core.model_net import PerceiverIO
from memoryPerceiver.memory_perceiver.processing.vectorize import input_to_tensor
from memoryPerceiver.memory_perceiver.core.model_wrapper import create_multimodal_perceiver

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
        self.body_parts = cf.body_parts
        self.body_outputs = cf.body_outputs
        self.frames_future = cf.frames_future
        self.audio_future = cf.audio_future
        self.hidden_dim = 64
        self.actor = create_multimodal_perceiver(cf, device)
        self.critic = nn.Sequential(nn.Linear(4, self.hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(self.hidden_dim, self.hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(self.hidden_dim, 1)
                                    ).to(device)
        

    def net_inference(self, states):
        device = states['observations'].device
        input = input_to_tensor(states, self.actor.modality, self.body_parts, device='cpu').to(device)
        x, _ = self.actor(input)
        output = {}
        output['audio'] = rearrange(self.actor.projections['audio'](
                            x[:, :self.audio_future, :]), "b t t1 -> b (t t1)" )
        start_modality = self.audio_future
        for part in self.body_outputs.keys():
            output[part] = self.actor.projections[part](
                x[:, start_modality:start_modality+self.frames_future, :] )
            start_modality += self.frames_future
        return output
        
    def act(self, states):

        # Net

        output = self.net_inference(states)

        # Actor

        actions_tmp1, actions_tmp2 = torch.split(torch.tanh(output['actions']), 2, dim=1)
        actions_mean = torch.cat(actions_tmp1.mean(dim=1), actions_tmp2.mean(dim=1), dim=-1)
        actions_var_tmp = torch.cat(actions_tmp1.var(dim=1), actions_tmp2.var(dim=1), dim=-1)
        actions_var = ((1+actions_var_tmp)/4+1e-5)
        dist = MultivariateNormal(actions_mean, torch.diag_embed(actions_var))
        actions = dist.sample()
        logprobs = dist.log_prob(actions)

        # Critic

        values = self.critic(states['observations'][:,-1])
        values = rearrange(values,  'b 1 -> b')

        return actions.detach(), logprobs.detach(), values.detach(), dist

    def evaluate(self, states, actions):

        # Net

        output = self.net_inference(states)

        # Actor

        actions_tmp1, actions_tmp2 = torch.split(torch.tanh(output['actions']), 2, dim=1)
        actions_mean = torch.cat(actions_tmp1.mean(dim=1), actions_tmp2.mean(dim=1), dim=-1)
        actions_var_tmp = torch.cat(actions_tmp1.var(dim=1), actions_tmp2.var(dim=1), dim=-1)
        actions_var = ((1+actions_var_tmp)/4+1e-5)
        dist = MultivariateNormal(actions_mean, torch.diag_embed(actions_var))
        logprobs = dist.log_prob(actions)
        entropies = dist.entropy() 

        # Critic

        values = self.critic(states['observations'][:,-1])
        values = rearrange(values,  'b 1 -> b')

        return logprobs, values, entropies, dist
    
    def instanciate_actions(self, actions, **kwargs):
        dt=8
        if 'dt' in kwargs:
            dt = kwargs['dt']
        t = torch.linspace(0,1,dt)
        action1, action2 = torch.split(actions, 2, dim=-1)
        actions_star = einsum('t, b d -> t b d', t, action1) + einsum('t, b d -> t b d', 1-t, action2)
        return actions_star