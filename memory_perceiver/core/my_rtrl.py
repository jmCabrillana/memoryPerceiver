import torch
from torch.autograd import grad, Variable
import numpy as np
from torch.func import jacrev, functional_call

class Param2Vec():
    def __init__(self, model, param_filter=''):
        """
        get a list of trainable variables
        """
        self.param_list = []
        self.name_list = []
        self.size_list = []
        for name, p in model.named_parameters():
            if p.requires_grad and name.startswith(param_filter):
                self.param_list.append(p)
                self.name_list.append(name)
                self.size_list.append(p.size())

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

    def merge_matrix(self, var_list, batch_size, device='cpu', sparsity=None):
        """
        merge a list of variables to a matrix
        """
        assert len(var_list) == len(self.size_list)
        theta_list = []
        for i in range(len(var_list)):
            if var_list[i] is not None:
                var_tmp = var_list[i].flatten(start_dim=1)
                if sparsity is not None:
                    var_tmp = var_tmp.to_sparse_bsr(blocksize=(sparsity, sparsity))
                theta_list.append(var_tmp)
            else:
                if sparsity is not None:
                    zeros = torch.sparse_bsr_tensor([0], [], [], size=(batch_size, np.prod(self.size_list[i])), device=device)
                else:
                    zeros = torch.zeros(batch_size, np.prod(self.size_list[i]), device=device)
                theta_list.append(zeros)

        return torch.cat(theta_list, dim=1)

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


        # Jacobians
        
        d_state_new_d_theta = [grad(outputs=o, inputs=self.nn_param.param_list, allow_unused=True, retain_graph=True)
                                    for o in state_new]
        d_state_new_d_theta = torch.cat([self.nn_param.merge(o, device=device) for o in d_state_new_d_theta])
        d_state_new_d_theta = d_state_new_d_theta.reshape([len_state, len_param])

        d_state_new_d_state_old = [grad(outputs=o, inputs=state_old, allow_unused=True, retain_graph=True)[0]
                                    for o in state_new]
        d_state_new_d_state_old = torch.cat(d_state_new_d_state_old).reshape([len_state, len_state])

        # Recursion formula and compute gradient

        g_t1 = delta_s.unsqueeze(0).mm(self.d_state_d_theta) + delta_theta_vec
        g_t1_list = self.nn_param.split(g_t1[0])
        self.d_state_d_theta = d_state_new_d_state_old.mm(self.d_state_d_theta) + d_state_new_d_theta

        for i in range(len(self.nn_param.param_list)):
            if self.nn_param.param_list[i].grad is None:
                self.nn_param.param_list[i].grad = g_t1_list[i]
            else:
                self.nn_param.param_list[i].grad += g_t1_list[i]

        # all-reduce grad/ other params 

#e.to_sparse_bsr(blocksize=(100, 100))
class RTRL_partial():
    def __init__(self, model, param_state_filter, block_dim=None):

        # Init params
        self.nn_param_state = Param2Vec(model, param_filter=param_state_filter)
        self.nn_param = Param2Vec(model)
        
        # Iterative sensitivity
        self.d_state_d_theta = None
        self.block_dim=block_dim

    def step(self, model, x, state_new, state_old, loss, **kwargs):
        """
        x: input 
        state_old: previous recurrent state of size (state)
        state_new: previous recurrent state of size (state)
        """
        device = state_old.device

        # Update online gradient 

        delta_theta = grad(loss, self.nn_param.param_list, retain_graph=True, allow_unused=True)

        for i in range(len(self.nn_param.param_list)):
            if self.nn_param.param_list[i].grad is None:
                self.nn_param.param_list[i].grad = delta_theta[i]
            else:
                self.nn_param.param_list[i].grad += delta_theta[i]

        # RTRL for state operations

        delta_s = grad(loss, state_old, retain_graph=True)[0]

        len_state = len(state_old)
        len_param_state = len(self.nn_param_state.merge(self.nn_param.param_list, device=device))
        b_dim = self.block_dim

        if self.d_state_d_theta is None:
            self.d_state_d_theta = Variable(torch.zeros(len_state, len_param_state, device=device))


        # Jacobians

        def fmodel(params, old_s): #functional version of model
            _, _, _, new_s = functional_call(model, params, (x, old_s), kwargs)
            return new_s
        dict_param_state = dict(zip(self.nn_param_state.name_list, self.nn_param_state.param_list))

        ## ds/dtheta: dim S T
        d_state_new_d_theta = jacrev(fmodel, argnums=(0))(dict_param_state, state_old)
        d_state_new_d_theta = self.nn_param_state.merge_matrix(list(d_state_new_d_theta.values()), len_state, 
                                    device=device, sparsity=b_dim)
        ## ds_new/ds_old: dim S S
        d_state_new_d_state_old = jacrev(fmodel, argnums=(1))(dict_param_state, state_old)
        if b_dim is not None:
            d_state_new_d_state_old = d_state_new_d_state_old.to_sparse_bsr(blocksize=(b_dim, b_dim))

        # Recursion formula and compute gradient

        g_t1 = delta_s.unsqueeze(0).mm(self.d_state_d_theta)  # + delta_theta_vec # Already done in online update
        g_t1_list = self.nn_param_state.split(g_t1[0])
        self.d_state_d_theta = d_state_new_d_state_old.mm(self.d_state_d_theta) + d_state_new_d_theta

        for i in range(len(self.nn_param_state.param_list)):
            if self.nn_param_state.param_list[i].grad is None:
                self.nn_param_state.param_list[i].grad = g_t1_list[i]
            else:
                self.nn_param_state.param_list[i].grad += g_t1_list[i]
