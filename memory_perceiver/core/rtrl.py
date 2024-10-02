import torch
from torch.autograd import grad, Variable
import numpy as np
import pandas as pd
import random
import time
from torch.func import jacrev, functional_call
from google.cloud import bigquery
from google.oauth2 import service_account

def cat_CSR_rows(csr_list):
    device = csr_list[0].device
    dtype = csr_list[0].dtype
    rows = 0
    cols = csr_list[0].size()[1]
    crow_indices_list, col_indices_list, values_list = [], [], []
    crow_sum = 0
    for i, csr in enumerate(csr_list):
        rows += csr.size()[0]
        if i == 0:
            crow_indices_tmp = csr.crow_indices()
        else:
            crow_indices_tmp = csr.crow_indices()[1:]
        crow_indices_list.append(crow_indices_tmp+crow_sum)
        crow_sum += crow_indices_tmp[-1]
        col_indices_list.append(csr.col_indices())
        values_list.append(csr.values())
    crow_indices = torch.cat(crow_indices_list)
    col_indices = torch.cat(col_indices_list)
    values = torch.cat(values_list)
    return torch.sparse_csr_tensor(crow_indices, col_indices, values, size=(rows, cols), device=device, dtype=dtype)

def shuffle_split_indexes(nb_latents_mem, query_dim, nb_chunks):
    i_mems = list(range(nb_latents_mem-1))
    len_chunk = int(np.ceil(len(i_mems)/nb_chunks))
    random.shuffle(i_mems)
    slice_indexes_list = []
    end_slice = list(range((nb_latents_mem-1)*query_dim, nb_latents_mem*query_dim))
    for i in range(nb_chunks):
        i_mem = i_mems[i*len_chunk: (i+1)*len_chunk]
        i_mem.sort()
        slice_indexes = sum( (list(range(i*query_dim, (i+1)*query_dim)) for i in i_mem), start=[])
        slice_indexes_list.append( slice_indexes + end_slice )
    return slice_indexes_list

class Param2Vec():
    def __init__(self, model, param_filter='', num_parameters=None):
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

        if num_parameters is not None:
            new_param_list, new_name_list, new_size_list = [], [], []
            cnt_parameters = 0
            while cnt_parameters < num_parameters and len(self.param_list) > 0 :
                i = torch.randint(len(self.param_list), (1,)).item()
                new_param_list.append(self.param_list.pop(i)) 
                new_name_list.append(self.name_list.pop(i))
                size_list = self.size_list.pop(i)
                new_size_list.append(size_list)
                cnt_parameters += np.prod(size_list)

            self.param_list = new_param_list
            self.name_list = new_name_list
            self.size_list = new_size_list
            for name, p in model.named_parameters():
                if name.startswith(param_filter) and name not in self.name_list:
                    p.requires_grad = False
            

    def merge(self, var_list, device='cpu'):
        """
        merge a list of variables to a vector
        """
        assert len(var_list) == len(self.size_list)
        theta_list = []
        for i in range(len(var_list)):
            if var_list[i] is not None:
                theta_list.append(var_list[i].flatten())
            else:
                theta_list.append(torch.zeros(self.size_list[i], device=device).flatten())
        return torch.cat(theta_list)

    def merge_matrix(self, var_list, device='cpu'):
        """
        merge a list of variables to a matrix
        """
        assert len(var_list) == len(self.size_list)
        theta_list = []
        batch_size = len(var_list[0])
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


class RTRL_partial():
    def __init__(self, model, param_state_filter, chunk_size=100, num_parameters=None, slice_indexes=None, table_save=None):

        # Init params
        self.num_parameters = num_parameters
        self.slice_indexes=slice_indexes
        self.param_state_filter = param_state_filter
        self.nn_param_state = Param2Vec(model, param_filter=param_state_filter, num_parameters=num_parameters)
        self.nn_param = Param2Vec(model)
        
        # Iterative sensitivity
        self.d_state_d_theta = None
        self.chunk_size=chunk_size
        self.table_save = None
        self.len_state_all = None

    def get_j_mappings(self):
        n, p = self.d_state_d_theta.shape
        k, j_k, param_count = 0, 0, 0
        param_size = np.prod(self.nn_param_state.size_list[0])
        j_to_k, j_to_j_k = [], []
        for j in range(p):
            j_k = j - param_count
            if j >= param_count + param_size:
                param_count += param_size
                k += 1
                param_size = np.prod(self.nn_param_state.size_list[k])
            j_k = j - param_count
            j_to_k.append(k)
            j_to_j_k.append(j_k)
        return j_to_k, j_to_j_k
    
    def get_inverse_j_mappings(self):
        n, p = self.d_state_d_theta.shape
        device = self.d_state_d_theta.device
        k, j_k, param_count = 0, 0, 0
        param_size = np.prod(self.nn_param_state.size_list[0])
        k_and_j_k_to_j = -1 * torch.ones(len(self.nn_param_state.param_list), p, device=device, dtype=torch.long)
        for j in range(p):
            j_k = j - param_count
            if j >= param_count + param_size:
                param_count += param_size
                k += 1
                param_size = np.prod(self.nn_param_state.size_list[k])
            j_k = j - param_count
            k_and_j_k_to_j[k ,j_k] = j
        return k_and_j_k_to_j

    def get_inverse_slice(self):
        device = self.d_state_d_theta.device
        inv = -1* torch.ones(self.len_state_all, device=device, dtype=torch.long)
        for i, k in enumerate(self.slice_indexes):
            inv[k] = i
        return inv

    def save_influence(self):
        " Save to bigquery"
        if self.table_save is not None:
            device = self.d_state_d_theta.device

            # Set up bigquery
            credentials = service_account.Credentials.from_service_account_file('/isaac-sim/memoryPerceiver/service_account.json')
            client = bigquery.Client(credentials=credentials)
            table_name = f"{credentials.project_id}.influence.{self.table_save}"
            schema =[
                        bigquery.SchemaField("h_i", "INTEGER"),
                        bigquery.SchemaField("param", "INTEGER"),
                        bigquery.SchemaField("p_j", "INTEGER"),
                        bigquery.SchemaField("dh_dtheta", "FLOAT"),
                    ]
            job_config = bigquery.LoadJobConfig(schema=schema, write_disposition="WRITE_APPEND")
            
            # Save influence matrix
            j_to_k, j_to_j_k = map(lambda x: torch.tensor(x, device=device), self.get_j_mappings())
            valid_mask = ~self.d_state_d_theta.isnan() & (self.d_state_d_theta != 0)
            valid_indices = valid_mask.nonzero()
            slice_indexes_t = torch.tensor(self.slice_indexes, device=device)
            name_array = np.array([hash(s) for s in self.nn_param_state.name_list])
            df = pd.DataFrame( {"h_i": slice_indexes_t[valid_indices[:, 0]].cpu().numpy(),
                    "param": name_array[j_to_k[valid_indices[:, 1]].cpu().numpy()],
                    "p_j": j_to_j_k[valid_indices[:, 1]].cpu().numpy(),
                    "dh_dtheta": self.d_state_d_theta[valid_indices[:, 0], valid_indices[:, 1]].cpu().numpy()} )
            client.load_table_from_dataframe(df, table_name, job_config=job_config)

    def load_influence(self):
        " Load influence from bigquery"
        if self.table_save is not None:
            device = self.d_state_d_theta.device

            # Rows to query
            slice_indexes_t = torch.tensor(self.slice_indexes, device=device)
            name_array = np.array([hash(s) for s in self.nn_param_state.name_list])
            params_string = ', '.join(str(x) for x in name_array)
            h_indexes_string = ', '.join(str(hi.item()) for hi in slice_indexes_t)
            
            # try:
            credentials = service_account.Credentials.from_service_account_file('/isaac-sim/memoryPerceiver/service_account.json')
            table_name = f"{credentials.project_id}.influence.{self.table_save}"
            client = bigquery.Client(credentials=credentials)
            query = f"""
                SELECT h_i, param, p_j, dh_dtheta 
                FROM `{table_name}`
                WHERE h_i IN ({h_indexes_string})
                AND param IN ({params_string})
                """
            
            results = client.query(query).result() 
            df = results.to_dataframe()
            df['param'] = df['param'].astype('category')
            df.set_index('param', inplace=True)
        
            # Fill the influence matrix with existing values
            k_and_j_k_to_j = self.get_inverse_j_mappings()
            inverse_slice = self.get_inverse_slice()
            fetched_params = df.index.unique()
            for k, param in enumerate(self.nn_param_state.name_list):
                param_hash = hash(param) 
                if param_hash in fetched_params:
                    df_param = df.loc[param_hash]
                    h_i = inverse_slice[df_param['h_i'].values]
                    p_j = k_and_j_k_to_j[k, df_param['p_j'].values]
                    self.d_state_d_theta[h_i, p_j] = torch.tensor(df_param['dh_dtheta'].values, 
                                                                  device=device, dtype=torch.float32)

            # except Exception as e:
            #     print(e)

    def step_safe(self, model, x, state_old, loss, **kwargs):
        try:
            self.step(model, x, state_old, loss, **kwargs)
        except RuntimeError as e:
            print('RTRL step exception', e)
            if 'out of memory' in str(e):
                print('Reset RTRL with less parameters to prevent out of memory errors')
                self.num_parameters = max(1, int(self.num_parameters / 1.5))
                self.chunk_size = max(1, int(self.chunk_size / 1.5))
                slice_len = max(1, int(len(self.slice_indexes) / 1.5))
                self.slice_indexes = random.sample(self.slice_indexes, slice_len)
                self.slice_indexes.sort()
                self.nn_param_state = Param2Vec(model, param_filter=self.param_state_filter, num_parameters=self.num_parameters)
                self.d_state_d_theta = None
                torch.cuda.empty_cache()
                time.sleep(1)
        

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
            if delta_theta[i] is not None:
                if self.nn_param.param_list[i].grad is None:
                    self.nn_param.param_list[i].grad = delta_theta[i]
                else:
                    self.nn_param.param_list[i].grad += delta_theta[i]

        # RTRL for state operations

        delta_s = grad(loss, state_old, retain_graph=True)[0][self.slice_indexes]

        state_old_slice = state_old[self.slice_indexes]
        len_state = len(state_old_slice)
        self.len_state_all = len(state_old)
        len_param_state = len(self.nn_param_state.merge(self.nn_param_state.param_list, device=device))

        if self.d_state_d_theta is None:
            self.d_state_d_theta = torch.zeros(len_state, len_param_state, device=device)
            # self.d_state_d_theta = torch.sparse_csr_tensor([0],[],[], size=(len_state, len_param_state), device=device)

        # Jacobians

        dict_param_state = dict(zip(self.nn_param_state.name_list, self.nn_param_state.param_list))
        dict_param_state = {k:v.detach() for k, v in dict_param_state.items()}
        state_old_detached = state_old_slice.detach()
        d_state_new_d_theta_list, d_state_new_d_state_old_list = [], []
        chunks = int(np.ceil(len_state/self.chunk_size))
        for i in range(chunks):

                def fmodel(params, old_s): #functional version of 
                    state_old_clone = state_old.clone().detach()
                    state_old_clone[self.slice_indexes] = old_s
                    _, _, _, state_new = functional_call(model, params, (x, state_old_clone), kwargs)
                    return state_new[self.slice_indexes][i*self.chunk_size:(i+1)*self.chunk_size]
                def jacobian(argnum):
                    return jacrev(fmodel, argnums=(argnum))(dict_param_state, state_old_detached)
                
                ## ds/dtheta: dim (S/C) T
                d_state_new_d_theta_tmp = jacobian(0)
                torch.cuda.empty_cache() 
                d_state_new_d_theta_list.append( self.nn_param_state.merge_matrix(list(d_state_new_d_theta_tmp.values()), 
                                                                            device=device).to_sparse_csr() )

                ## ds_new/ds_old: dim (S/C) S
                d_state_new_d_state_old_list.append( jacobian(1).to_sparse_csr() )
                torch.cuda.empty_cache() 

        d_state_new_d_theta = cat_CSR_rows(d_state_new_d_theta_list)
        d_state_new_d_state_old = cat_CSR_rows(d_state_new_d_state_old_list)

        # Recursion formula and compute gradient

        g_t1 = delta_s.unsqueeze(0).mm(self.d_state_d_theta)  # + delta_theta_vec # Already done in online update
        g_t1_list = self.nn_param_state.split(g_t1[0])
        self.d_state_d_theta = d_state_new_d_state_old.mm(self.d_state_d_theta) + d_state_new_d_theta

        for i in range(len(self.nn_param_state.param_list)):
            if self.nn_param_state.param_list[i].grad is None:
                self.nn_param_state.param_list[i].grad = g_t1_list[i]
            else:
                self.nn_param_state.param_list[i].grad += g_t1_list[i]


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


class UORO():
    def __init__(self, model, epsilon_perturbation=1e-7, epsilon_stability=1e-7):

        self.epsilon_perturbation = epsilon_perturbation
        self.epsilon_stability = epsilon_stability

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