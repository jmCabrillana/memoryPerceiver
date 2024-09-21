import torch
from torch.autograd import grad, Variable
from .rnn import SimpleRNN, SimpleLSTM
import numpy as np
from einops import rearrange
from memoryPerceiver.memory_perceiver.isaac.agent import ActorCritic, ActorCriticSimple, ActorCriticPerceiver, create_agent


class Param2Vec():
    def __init__(self, model):
        """
        get a list of trainable variables
        """
        self.param_list = []
        self.size_list = []
        for p in model.parameters():
            if p.requires_grad:
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


class RTRL():
    def __init__(self, model):

        self.nn_param = Param2Vec(model)
        self.d_state_d_theta = None

    def step(self, model, x, state_new, state_old, loss, **kwargs):
        """
        x: input of size (1, 1, input_size)
        s: previous recurrent state of size (1, 1, hidden_size)
        y: target of size (1,)

        self.s_toupee: column vector of size (state, )
        self.theta_toupee: row vector of size (params, )
        """
        # print('---')
        device = state_old.device

        delta_s = grad(loss, state_old, retain_graph=True)[0]
        delta_theta = grad(loss, self.nn_param.param_list, retain_graph=True, allow_unused=True)
        delta_theta_vec = self.nn_param.merge(delta_theta, device=device)

        len_state = len(state_old)
        len_param = len(delta_theta_vec)

        if self.d_state_d_theta is None:
            self.d_state_d_theta = Variable(torch.zeros(len_state, len_param, device=device))


        # Backprop
        # jacobian(state_new, nn_param_vec)
        d_state_new_d_theta = [grad(outputs=o, inputs=self.nn_param.param_list, allow_unused=True, retain_graph=True)
                                    for o in state_new]
        d_state_new_d_theta = torch.cat([self.nn_param.merge(o, device=device) for o in d_state_new_d_theta])
        d_state_new_d_theta = d_state_new_d_theta.reshape([len_state, len_param])

        d_state_new_d_state_old = [grad(outputs=o, inputs=state_old, allow_unused=True, retain_graph=True)[0]
                                    for o in state_new]
        d_state_new_d_state_old = torch.cat(d_state_new_d_state_old).reshape([len_state, len_state])

        # print('delta_s', delta_s.shape)
        # print('delta_theta_vec', delta_theta_vec.shape)
        # print('d_state_d_theta', self.d_state_d_theta.shape)
        # print('delta_s', delta_s.shape)
        # print('d_state_new_d_theta_vec', d_state_new_d_theta.shape)


        g_t1 = delta_s.unsqueeze(0).mm(self.d_state_d_theta) + delta_theta_vec
        g_t1_list = self.nn_param.split(g_t1[0])
        self.d_state_d_theta = d_state_new_d_state_old.mm(self.d_state_d_theta) + d_state_new_d_theta

        # self.optimizer.zero_grad()
        for i in range(len(self.nn_param.param_list)):
            if self.nn_param.param_list[i].grad is None:
                self.nn_param.param_list[i].grad = g_t1_list[i]
            else:
                self.nn_param.param_list[i].grad += g_t1_list[i]

        # all-reduce grad/ other params 