from __future__ import print_function, division
import os
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from skimage.feature import local_binary_pattern


# np.array
class RandomHorizontalFlip(object):
    """Horizontally flip the given Image randomly with a probability of 0.5."""

    def worker(self, img_faces, depth_map):
        p = random.random()
        if p < 0.5:
            img_faces = cv2.flip(img_faces, 1)
            depth_map = cv2.flip(depth_map, 1)
        return img_faces, depth_map

    def __call__(self, args):
        img_faces, depth_map = args
        return self.worker(img_faces, depth_map)


# np.array
class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def worker(self, img_faces, depth_map):
        img = np.array(img_faces, dtype=np.float)
        img = img[:, :, ::-1].transpose((2, 0, 1))
        depth_map = np.array(depth_map, dtype=np.float)
        return torch.from_numpy(img.copy()).float(), torch.from_numpy(depth_map.copy()).float()

    def __call__(self, args):
        img_faces, depth_map = args
        return self.worker(img_faces, depth_map)


# Tensor
class Normaliztion(object):

    def worker(self, img_faces, depth_map):
        return (img_faces - 127.5) / 127.5, depth_map / 255.0  # [-1,1], [0,1]

    def __call__(self, args):
        img_faces, depth_map = args
        return self.worker(img_faces, depth_map)


class Spoofing_train(Dataset):
    def __init__(self, info_list, image_dir, map_dir, args, transform=None):
        class Record(object):
            def __init__(self, row):
                self._data = row.strip().split(',')
                self.path = self._data[1]
                self.label = eval(self._data[0])  # real 1 photo -1 video -2
                self.video_length = int(self._data[2])  # the num of frames in the entire video

        # self.video_records = [Record(x) for x in open(info_list)]
        self.video_records = [Record(x) for x in open(info_list) if not x.startswith('-1')]  # for texture
        self.image_dir = image_dir
        self.map_dir = map_dir
        self.test = args.test
        self.map_size = args.mapsize
        self.input_size = args.inputsize
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
        image_id = random.randint(0, self.video_records[idx].video_length - 1)

        img = cv2.imread(os.path.join(self.image_dir, path, '%d.jpg' % image_id))
        img_face = cv2.resize(self._crop_face_from_scene(img, face_scale), (self.input_size, self.input_size))
        img_lbp = np.zeros_like(img_face)
        for c in range(3):
            img_lbp[:, :, c] = local_binary_pattern(img_face[:, :, c], 8, 1)

        if spoofing_label == 1:
            map = cv2.imread(os.path.join(self.map_dir, path, '%d.jpg' % image_id), 0)
            depth_map = cv2.resize(self._crop_face_from_scene(map, face_scale), (self.map_size, self.map_size))
        else:
            depth_map = np.zeros((self.map_size, self.map_size))

        img_lbp, depth_map = self.transform((img_lbp, depth_map))

        return img_lbp, depth_map, spoofing_label, path
