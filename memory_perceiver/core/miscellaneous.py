import torch

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

