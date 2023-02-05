from __future__ import print_function
import torch.utils.data as data
from os import listdir
from os.path import isfile, join
import torch
from pytorchvideo.data.encoded_video import EncodedVideo

class ThinkFuture(data.Dataset):

    def __init__(self, path, input_duration):
        self.input_duration = input_duration

        self.files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (seq, target) where sampled sequences are splitted into a seq
                    and target part
        """

        # Initialize an EncodedVideo helper class and load the video
        video = EncodedVideo.from_path(self.files[index])

        # Load the desired clip
        start_sec = torch.rand(1).item() * (video.duration-self.input_duration)

        return self.files[index], start_sec

    def __len__(self):
        return len(self.files)