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
        assert len(var_list) == len(self.size_list)
        theta_list = []
        for i in range(len(var_list)):
            var = var_list[i]
            if var is not None:
                theta_list.append(var.flatten())
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

        if self.s_tilda is None or self.theta_tilda is None:
            self.d_state_d_theta = Variable(torch.zeros(len(state_old), len(delta_theta_vec), device=device))


        # Backprop
        d_state_new_d_theta= grad(outputs=state_new, inputs=self.nn_param.param_list, allow_unused=True, retain_graph=True)
        d_state_new_d_state_old = grad(outputs=state_new, inputs=state_old, allow_unused=True, retain_graph=True)
        d_state_new_d_theta_vec = self.nn_param.merge(d_state_new_d_theta, device=device)


        self.theta.grad = delta_s.view(1, -1).mm(self.dstate_dtheta).view(-1) + delta_theta_vec
        self.d_state_d_theta = d_state_new_d_state_old.mm(self.d_state_d_theta) + d_state_new_d_theta_vec


        # g_t1 = torch.dot(delta_s.squeeze(), self.s_tilda) * self.theta_tilda + delta_theta_vec
        # g_t1_list = self.nn_param.split(g_t1)

        # self.optimizer.zero_grad()
        for i in range(len(self.nn_param.param_list)):
            if self.nn_param.param_list[i].grad is None:
                self.nn_param.param_list[i].grad = g_t1_list[i]
            else:
                self.nn_param.param_list[i].grad += g_t1_list[i]

        # all-reduce grad/ other params 