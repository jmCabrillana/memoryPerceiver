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


class UORO():
    def __init__(self, model, epsilon_perturbation=1e-7, epsilon_stability=1e-7):

        self.epsilon_perturbation = epsilon_perturbation
        self.epsilon_stability = epsilon_stability
        self.criterion = torch.nn.CrossEntropyLoss()

        self.nn_param = Param2Vec(model)
        self.s_tilda = None
        self.theta_tilda = None

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
            self.s_tilda = Variable(torch.zeros(*state_old.squeeze().size(), device=device))  # (batch_size, state_dim)
            self.theta_tilda = Variable(torch.zeros(*delta_theta_vec.size(), device=device))  # (n_params, )

        g_t1 = torch.dot(delta_s.squeeze(), self.s_tilda) * self.theta_tilda + delta_theta_vec
        g_t1_list = self.nn_param.split(g_t1)

        # ForwardDiff
        state_old_perturbed = state_old + self.s_tilda * self.epsilon_perturbation  #.detach()
        state_new_perturbed = model(x, state_old_perturbed, **kwargs)[-1]
        s_forwarddiff = (state_new_perturbed - state_new)/self.epsilon_perturbation

        # Backprop
        nu_vec = Variable(torch.round(torch.rand(*state_old.size(), device=device)) * 2 - 1)
        delta_theta_g = grad(outputs=state_new, inputs=self.nn_param.param_list, grad_outputs=nu_vec, allow_unused=True,
                             retain_graph=True)
        delta_theta_g_vec = self.nn_param.merge(delta_theta_g, device=device)

        rho_0 = torch.sqrt(self.theta_tilda.norm()/(s_forwarddiff.norm() + self.epsilon_stability)) + self.epsilon_stability
        rho_1 = torch.sqrt(delta_theta_g_vec.norm()/(nu_vec.norm() + self.epsilon_stability)) + self.epsilon_stability

        self.s_tilda = (rho_0 * s_forwarddiff.squeeze() + rho_1 * nu_vec.squeeze()).detach()
        self.theta_tilda = (self.theta_tilda / rho_0 + delta_theta_g_vec / rho_1).detach()

        # self.optimizer.zero_grad()
        for i in range(len(self.nn_param.param_list)):
            if self.nn_param.param_list[i].grad is None:
                self.nn_param.param_list[i].grad = g_t1_list[i]
            else:
                self.nn_param.param_list[i].grad += g_t1_list[i]

        # all-reduce grad/ other params 