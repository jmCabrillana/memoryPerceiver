import sys
import os
from os import listdir
from os.path import isfile, join, getsize, dirname, basename
import random
import torch.utils.data as data

import torch
from pytorchvideo.data.encoded_video import EncodedVideo

sys.path.append(join(dirname(os.getcwd())))

from memory_perceiver.processing.videoTransform import transform_video, transform_audio
from memory_perceiver.main.configuration import cf
from pickle_sequence import read_seq

class Streamer(data.Dataset):

    def __init__(self, path, input_duration, target_duration, total_duration, batch_size):
        self.input_duration = input_duration
        self.total_duration = total_duration
        self.target_duration = target_duration
        self.len_stream = int( (total_duration-input_duration)/target_duration )
        self.batch_size = batch_size

        # data paths

        self.pathes = [join(path, f) for f in listdir(path) if isfile(join(path, f))]

        # Randomly cut clips

        self.start_sec = []
        self.random_cut()
        

    def shuffle_pathes(self):
        random.shuffle(self.pathes)
        self.random_cut()

    def random_cut(self):
        self.start_sec = []
        for path in self.pathes:
            duration = int((getsize(path)/8)/cf.seq_byte_size)
            self.start_sec.append(torch.rand(1).item() * (duration-self.total_duration))

    def __getitem__(self, index):

        # Retrieve path and stream number

        bloc_id = index // (self.batch_size*self.len_stream)
        bloc_index = index % (self.batch_size*self.len_stream)
        path_id_relative = bloc_index % self.batch_size
        path_id = path_id_relative + bloc_id*self.batch_size
        stream_nb = bloc_index // self.batch_size

        # Initialize an EncodedVideo helper class and load the video

        start = self.start_sec[path_id] + stream_nb * self.target_duration
        stop_input = start + self.input_duration
        path = self.pathes[path_id]

        # Load pose amd clean audio as binary

        seq_path = dirname(path) + '/seq/' + basename(path)[:-4] + '.bin'
        input = read_seq(seq_path, start, stop_input)
        target = read_seq(seq_path, stop_input, stop_input + self.target_duration)

        # Load media

        stream = EncodedVideo.from_path(path)
        inputMedia = stream.get_clip(start_sec=start, end_sec = stop_input)
        targetMedia = stream.get_clip(start_sec=stop_input, end_sec = stop_input + self.target_duration)
        input.update(transform_audio(int(self.input_duration*cf.fps))(transform_video(int(self.input_duration*cf.fps))(inputMedia)))
        target.update(transform_audio(int(self.target_duration*cf.fps))(transform_video(int(self.target_duration*cf.fps))(targetMedia)))

        return path_id, stream_nb, input, target

    def __len__(self):
        return len(self.pathes)*self.len_stream

class videoStreamer(data.Dataset):

    def __init__(self, path, input_duration, target_duration, total_duration):
        self.input_duration = input_duration
        self.total_duration = total_duration
        self.target_duration = target_duration
        self.len_stream = int( (total_duration-input_duration)/target_duration )

        # data paths

        self.pathes = [join(path, f) for f in listdir(path) if isfile(join(path, f))]

        # Randomly cut clips

        self.start_sec = []

    def shuffle_pathes(self):
        random.shuffle(self.pathes)
        self.random_cut()

    def random_cut(self):
        self.start_sec = []
        for path in self.pathes:
            stream = EncodedVideo.from_path(path)
            self.start_sec.append(torch.rand(1).item() * (stream.duration-self.total_duration))

    def __getitem__(self, index):

        # Retrieve path and stream number

        path_id = index % len(self.pathes)
        stream_nb = index // len(self.pathes)

        # Initialize an EncodedVideo helper class and load the video

        stream = EncodedVideo.from_path(self.pathes[path_id])
        start = self.start_sec[path_id] + stream_nb * self.target_duration
        stop_input = start + self.input_duration
        input = stream.get_clip(start_sec=start, end_sec = stop_input)
        target = stream.get_clip(start_sec=stop_input, end_sec = stop_input + self.target_duration)

        return path_id, input, target

    def __len__(self):
        return len(self.pathes)*self.len_stream-1