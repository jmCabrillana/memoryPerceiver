import torch

def layer_init(layer, std=1.4, bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def batch_dictionary(dicts):
    out_dict = {}
    for key in dicts[0].keys():
        value = []
        for i in range(len(dicts)):
            value.append(dicts[i][key])
        out_dict[key] = value
    return out_dict

def batch_dictionary_to_list(dict):
    out_list = []
    for i in range(len(list(dict.values())[0])):
        dict_row = {}
        for key in dict:
            dict_row.update({key: dict[key][i]})
        out_list.append(dict_row)
    return out_list

def custom_random(seed1, seed2):
    a = 1664525
    c = 1013904223
    m = 2**32
    combined_seed = (seed1 + seed2) << 16
    combined_seed = (a * combined_seed + c) % m
    return combined_seed / m

def print_cuda_tensors():
    objects = []
    mem_total = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                mem = (obj.element_size() * obj.nelement()/1000000)
                mem_total += mem
                objects.append( [type(obj), obj.size(), obj.nelement(), mem])
        except:
            pass
    print(mem_total)
    objects = sorted(objects, key = lambda x: x[2], reverse=True)
    for obj in objects[:5]:
        print(obj)

def freeze_parameters(model, filter):
    for name, p in model.named_parameters():
        if name.startswith(filter):
            p.requires_grad = False
        else:
            p.requires_grad = True