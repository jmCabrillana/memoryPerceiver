import torch
from torch.autograd import grad, Variable
from .rnn import SimpleRNN, SimpleLSTM
import numpy as np
from einops import rearrange
from memoryPerceiver.memory_perceiver.isaac.agent import ActorCritic, ActorCriticSimple, ActorCriticPerceiver, create_agent


class Param2Vec():
    def __init__(self, model, param_filter=''):
        """
        get a list of trainable variables
        """
        self.param_list = []
        self.size_list = []
        for name, p in model.named_parameters():
            if p.requires_grad and name.startswith(param_filter):
                self.param_list.append(p)
                self.size_list.append(p.size())

        self.num_list = len(self.param_list)

    def merge(self, var_list, device='cpu'):
        """
        merge a list of variables to a vector
        """
        # assert len(var_list) == len(self.size_list)
        theta_list = []
        for i in range(len(var_list)):
            if var_list[i] is not None:
                theta_list.append(var_list[i].flatten())
            else:
                theta_list.append(torch.zeros(self.size_list[i], device=device).flatten())
        return torch.cat(theta_list)

    def split(self, var_vec):
        """
        split a vec to a list
        """
        var_list = []
        count = 0
        for i in range(len(self.size_list)):
            prod_size = np.prod(self.size_list[i])
            var_list.append(var_vec[count:(count+prod_size)].reshape(self.size_list[i]))
            count += prod_size
        return var_list

def jacobian(out_vec, in_vec):
    """
    :param out_vec:
    :param in_vec:
    :return: An (len(out_vec), len(in_vec)) jacobian
    """
    return torch.cat([grad(o, in_vec, retain_graph=True, allow_unused=True,)[0][None] for o in out_vec], dim=0)




