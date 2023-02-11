import sys
import os
from os.path import join, dirname
import pickle

sys.path.append(join(dirname(dirname(os.getcwd()))))

from memory_perceiver.main.configuration import cf


def write_seq(dict, dst):
    len_seq = len(dict[list(cf.body_parts.keys())[0]])
    for i in range(len_seq):
        single_seq = []
        single_seq.append(dict['clean_audio'][i*cf.sample_per_frame : (i+1)*cf.sample_per_frame])
        for part in cf.body_parts_extended.keys():
            single_seq.append(dict[part][i])
        pickle.dump(single_seq, dst, protocol=pickle.HIGHEST_PROTOCOL)
    # print(len(pickle.dumps(single_seq, protocol=pickle.HIGHEST_PROTOCOL)))
    dst.flush()

def write_reward(rewards, dst):
    for reward in rewards:
        pickle.dump(reward, dst, protocol=pickle.HIGHEST_PROTOCOL)
    print(len(pickle.dumps(reward, protocol=pickle.HIGHEST_PROTOCOL)))
    dst.flush()

def read_seq(path, start, stop):
    start_frame = int(start * cf.fps)
    end_frame = int(stop * cf.fps)
    seq_file = open(path, 'rb')
    seq_file.seek(start_frame*cf.seq_byte_size, os.SEEK_CUR)
    seq_dict = {}
    seq_dict['clean_audio'] = []
    for part in cf.body_parts_extended.keys():
        seq_dict[part] = []

    # Read pickle file 

    for _ in range(start_frame, end_frame):
        seq_array = pickle.load(seq_file)
        seq_dict['clean_audio'] += seq_array[0]
        part_index = 1
        for part in cf.body_parts_extended.keys():
            seq_dict[part].append(seq_array[part_index])
            part_index+=1

    seq_file.close()

    return seq_dict

def read_reward(path, start, stop):
    start_frame = int(start * cf.fps)
    end_frame = int(stop * cf.fps)
    seq_file = open(path, 'rb')
    seq_file.seek(start_frame*cf.reward_byte_size, os.SEEK_CUR)
    rewards = []

    # Read pickle file 

    for _ in range(start_frame, end_frame):
        rewards.append(pickle.load(seq_file))
    seq_file.close()

    return rewards