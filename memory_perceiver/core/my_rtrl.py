import torch
from torch.autograd import grad, Variable
import numpy as np
from torch.func import jacrev, functional_call
import os

def cat_CSR_rows(csr_list):
    device = csr_list[0].device
    dtype = csr_list[0].dtype
    rows = 0
    cols = csr_list[0].size()[1]
    crow_indices_list, col_indices_list, values_list = [], [], []
    ccol_sum = 0
    for i, csr in enumerate(csr_list):
        rows += csr.size()[0]
        crow_indices_tmp = csr.crow_indices()
        crow_indices_list.append(crow_indices_tmp+ccol_sum)
        ccol_sum += crow_indices_tmp[-1]
        col_indices_list.append(csr.col_indices())
        values_list.append(csr.values())
    crow_indices = torch.cat(crow_indices_list)
    col_indices = torch.cat(col_indices_list)
    values = torch.cat(values_list)
    return torch.sparse_csr_tensor(crow_indices, col_indices, values, size=(rows, cols), device=device, dtype=dtype)


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

    def merge_matrix(self, var_list, batch_size, device='cpu'):
        """
        merge a list of variables to a matrix
        """
        assert len(var_list) == len(self.size_list)
        theta_list = []
        for i in range(len(var_list)):
            if var_list[i] is not None:
                var_tmp = var_list[i].flatten(start_dim=1)
                theta_list.append(var_tmp)
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
    def __init__(self, model, param_state_filter, chunks=1, slice_indexes=None, table_save=None):

        # Init params
        self.nn_param_state = Param2Vec(model, param_filter=param_state_filter)
        self.nn_param = Param2Vec(model)
        
        # Iterative sensitivity
        self.d_state_d_theta = None
        self.chunks=chunks
        self.slice_indexes=slice_indexes
        self.table_save = None

    def save_influence(self, table):
        " Save to bigquery"
        pass


    def step(self, model, x, state_old, loss, **kwargs):
        """
        x: input 
        state_old: previous recurrent state of size (state)
        state_new: previous recurrent state of size (state)
        """
        device = state_old.device
        if self.slice_indexes is None:
            self.slice_indexes = list(range(len(state_old)))

        # Update online gradient 

        delta_theta = grad(loss, self.nn_param.param_list, retain_graph=True, allow_unused=True)

        for i in range(len(self.nn_param.param_list)):
            if self.nn_param.param_list[i].grad is None:
                self.nn_param.param_list[i].grad = delta_theta[i]
            else:
                self.nn_param.param_list[i].grad += delta_theta[i]

        # RTRL for state operations

        delta_s = grad(loss, state_old, retain_graph=True)[0][self.slice_indexes]

        state_old_slice = state_old[self.slice_indexes]
        len_state = len(state_old_slice)
        len_param_state = len(self.nn_param_state.merge(self.nn_param_state.param_list, device=device))
        assert(len_state % self.chunks == 0)
        chunk_size = len_state//self.chunks

        if self.d_state_d_theta is None:
            self.d_state_d_theta = torch.zeros(len_state, len_param_state, device=device)
            # self.d_state_d_theta = torch.sparse_csr_tensor([0],[],[], size=(len_state, len_param_state), device=device)


        # Jacobians

        dict_param_state = dict(zip(self.nn_param_state.name_list, self.nn_param_state.param_list))
        dict_param_state = {k:v.detach() for k, v in dict_param_state.items()}
        state_old_detached = state_old_slice.detach()
        d_state_new_d_theta_list, d_state_new_d_state_old_list = [], []
        for i in range(self.chunks):

            def fmodel(params, old_s): #functional version of 
                state_old_clone = state_old.clone().detach()
                state_old_clone[self.slice_indexes] = old_s
                _, _, _, state_new = functional_call(model, params, (x, state_old_clone), kwargs)
                return state_new[self.slice_indexes][i*chunk_size:(i+1)*chunk_size]
            def jacobian(argnum):
                return jacrev(fmodel, argnums=(argnum))(dict_param_state, state_old_detached)
            
            ## ds/dtheta: dim (S/C) T
            d_state_new_d_theta_tmp = jacobian(0)
            d_state_new_d_theta_list.append( self.nn_param_state.merge_matrix(list(d_state_new_d_theta_tmp.values()), 
                                                        chunk_size, device=device).to_sparse_csr() )
            torch.cuda.empty_cache() 

            ## ds_new/ds_old: dim (S/C) S
            d_state_new_d_state_old_list.append( jacobian(1).to_sparse_csr() )
            torch.cuda.empty_cache() 

        d_state_new_d_theta = cat_CSR_rows(d_state_new_d_theta_list)
        d_state_new_d_state_old = cat_CSR_rows(d_state_new_d_state_old_list)

        # Recursion formula and compute gradient

        g_t1 = delta_s.unsqueeze(0).mm(self.d_state_d_theta)  # + delta_theta_vec # Already done in online update
        g_t1_list = self.nn_param_state.split(g_t1[0])
        # print('d_state_new_d_state_old', d_state_new_d_state_old._nnz())
        # print('d_state_new_d_theta', d_state_new_d_theta._nnz())
        try:
            self.d_state_d_theta = d_state_new_d_state_old.mm(self.d_state_d_theta) + d_state_new_d_theta
        except Exception as e:
            print(e)

        for i in range(len(self.nn_param_state.param_list)):
            if self.nn_param_state.param_list[i].grad is None:
                self.nn_param_state.param_list[i].grad = g_t1_list[i]
            else:
                self.nn_param_state.param_list[i].grad += g_t1_list[i]
