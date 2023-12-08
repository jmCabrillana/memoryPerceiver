import torch
from einops import rearrange, repeat
from torch import nn, einsum
from torch.distributions import Normal, MultivariateNormal

from memoryPerceiver.memory_perceiver.core.model_net import PerceiverIO
from memoryPerceiver.memory_perceiver.processing.vectorize import input_to_tensor, fourier_encode

def create_model(cf, device):

    # Projection to return modalities, enhancement

    modality_parameters = {}
    modality_parameters['audio'] = cf.audio_latent_dim
    modality_parameters['video'] = cf.video_latent_dim
    for part, params in cf.body_parts.items():
        modality_parameters[part] = cf.query_dim - 2*128 - params
    for part, params in cf.body_outputs.items():
        modality_parameters[part] = cf.query_dim - 2*128 - params

    projection_parameters = dict(cf.body_outputs)
    projection_parameters.update({'audio': 16})

    # Input compression

    special_compression = [i for i in range(cf.video_future)] + [i for i in range(cf.len_video, cf.len_video+cf.audio_future)]
    for k in range(len(cf.body_parts)):
        special_compression += [ cf.len_video + cf.len_audio + k*cf.frames + i for i in range(cf.frames_future) ]

    # Instantiate model
        
    model = PerceiverIO(
            2,
            cf.query_dim,
            special_compression = special_compression,
            device=device,
            projection_parameters = projection_parameters,
            modality_parameters = modality_parameters,
            max_batch_size = cf.batch_size
        )
    model = model.to(device)

    # Queries

    queries = []
    audio_query = repeat( torch.cat([model.modality['audio'], torch.zeros(16, device=device)]), '... -> n ...', n=cf.audio_future)
    audio_pos = fourier_encode(torch.linspace(-1, 1, cf.audio_future), cf.audio_future/2, num_bands=128).to(device)
    queries.append(torch.cat([audio_query, audio_pos], dim=-1).to(device))
    for part, params in cf.body_outputs.items():
        part_query = repeat( torch.cat([model.modality[part], torch.zeros(params, device=device)]), '... -> n ...', n=cf.frames_future)
        part_pos = fourier_encode(torch.linspace(-1, 1, cf.frames_future), cf.frames_future/2, num_bands=128).to(device)
        queries.append(torch.cat([part_query, part_pos], dim=-1).to(device))
    model.queries=torch.cat(queries)


    return model

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, device, hidden_dim=64):
        super(ActorCritic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.actor = nn.Sequential( nn.Linear(state_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, 2*action_dim)
                                    ).to(device)

        self.critic = nn.Sequential(nn.Linear(state_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, 1)
                                    ).to(device)
        
    def act(self, state):

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

    def evaluate(self, state, actions):

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
    
    def instanciate_actions(action):
        pass     

    def compute_loss( ):
        pass

    

model = ActorCritic(4 ,1)
model_student = ActorCritic(4 ,1)
# model.act(torch.randn(8, 4, device=device))