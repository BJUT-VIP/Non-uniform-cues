from __future__ import print_function, division
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
import os


# np.array
class RandomHorizontalFlip(object):
    """Horizontally flip the given Image randomly with a probability of 0.5."""

    def __call__(self, img_faces):
        p = random.random()
        if p < 0.5:
            for i in range(len(img_faces)):
                img_faces[i] = cv2.flip(img_faces[i], 1)
        return img_faces


# np.array
class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __call__(self, img_faces):
        img = np.array(img_faces, dtype=np.float)
        img = img[:, :, :, ::-1].transpose((3, 0, 1, 2))  # C, T, H, W
        return torch.from_numpy(img.copy()).float()


# Tensor
class Normaliztion(object):
    def __call__(self, img_faces):
        return (img_faces - 127.5) / 127.5  # [-1,1]


class Spoofing_train(Dataset):
    def __init__(self, info_list, image_dir, args, transform=None):
        class Record(object):
            def __init__(self, row):
                self._data = row.strip().split(',')
                self.path = self._data[1]
                self.label = eval(self._data[0])  # real 1 photo -1 video -2
                self.video_length = int(self._data[2])  # the num of frames in the entire video

        # self.video_records = [Record(x) for x in open(info_list)]
        self.video_records = [Record(x) for x in open(info_list) if not x.startswith('-2')]  # for motion
        self.image_dir = image_dir
        self.test = args.test
        self.map_size = args.mapsize
        self.input_size = args.inputsize
        self.length = args.length
        self.transform = transform

    def __len__(self):
        return len(self.video_records)

    @staticmethod
    def _crop_face_from_scene(image, scale):
        scale_input = 1.6  # The multiple of the detected face box
        w = image.shape[1]
        h = image.shape[0]

        y1 = int(0 + (0.5 - scale / scale_input / 2) * h)
        x1 = int(0 + (0.5 - scale / scale_input / 2) * w)
        y2 = int(h - (0.5 - scale / scale_input / 2) * h)
        x2 = int(w - (0.5 - scale / scale_input / 2) * w)

        region = image[y1:y2, x1:x2]
        return region

    def __getitem__(self, idx):
        spoofing_label = self.video_records[idx].label
        path = self.video_records[idx].path

        face_scale = 1.3 if self.test else random.randint(12, 14) / 10.0
        image_id = random.randint(0, self.video_records[idx].video_length - self.length)

        img_faces = []
        for i in range(self.length):
            img = cv2.imread(os.path.join(self.image_dir, path, '%d.jpg' % (image_id + i)))
            img_faces.append(
                cv2.resize(self._crop_face_from_scene(img, face_scale), (self.input_size, self.input_size)))
        img_faces = self.transform(img_faces)

        depth_map = torch.ones(1, self.map_size, self.map_size) if spoofing_label == 1 else torch.zeros(
            1, self.map_size, self.map_size)

        return img_faces, depth_map, spoofing_label, path
