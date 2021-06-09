from __future__ import print_function, division
import cv2
import numpy as np
import random
import torch
import time
import os
import imgaug.augmenters as iaa
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage.feature import local_binary_pattern


# np.array
class RandomErasing_one(object):
    """
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    """

    def __init__(self, probability=0.5, sl=0.01, sh=0.05, r1=0.5, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def worker(self, img):
        if random.random() < self.probability:
            attempts = random.randint(1, 2)
            for attempt in range(attempts):
                area = img[0].shape[0] * img[0].shape[1]
                target_area = random.uniform(self.sl, self.sh) * area
                aspect_ratio = random.uniform(self.r1, 1 / self.r1)

                h = int(round(np.sqrt(target_area * aspect_ratio)))
                w = int(round(np.sqrt(target_area / aspect_ratio)))

                if w < img[0].shape[1] and h < img[0].shape[0]:
                    x1 = random.randint(0, img[0].shape[0] - h)
                    y1 = random.randint(0, img[0].shape[1] - w)
                    img[0][x1:x1 + h, y1:y1 + w, 0] = self.mean[0]
                    img[0][x1:x1 + h, y1:y1 + w, 1] = self.mean[1]
                    img[0][x1:x1 + h, y1:y1 + w, 2] = self.mean[2]
        return img

    def __call__(self, args):
        img_faces, img_lbp, depth_map = args
        return self.worker(img_faces), img_lbp, depth_map


# np.array
class RandomHorizontalFlip(object):
    """Horizontally flip the given Image randomly with a probability of 0.5."""

    def worker(self, img_faces, img_lbp, depth_map):
        p = random.random()
        if p < 0.5:
            for i in range(len(img_faces)):
                img_faces[i] = cv2.flip(img_faces[i], 1)
            img_lbp = cv2.flip(img_lbp, 1)
            depth_map = cv2.flip(depth_map, 1)
        return img_faces, img_lbp, depth_map

    def __call__(self, args):
        img_faces, img_lbp, depth_map = args
        return self.worker(img_faces, img_lbp, depth_map)


# np.array
class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def worker(self, img_faces, img_lbp, depth_map):
        img = np.array(img_faces, dtype=np.float32)
        img = img[:, :, :, ::-1].transpose((3, 0, 1, 2))
        img_lbp = np.array(img_lbp, dtype=np.float32)
        img_lbp = img_lbp[:, :, ::-1].transpose((2, 0, 1))
        depth_map = np.array(depth_map, dtype=np.float32)
        return torch.from_numpy(img.copy()).float(), torch.from_numpy(img_lbp.copy()).float(), torch.from_numpy(
            depth_map.copy()).float()

    def __call__(self, args):
        img_faces, img_lbp, depth_map = args
        return self.worker(img_faces, img_lbp, depth_map)


# Tensor
class Cutout_one(object):
    def __init__(self, length=50):
        self.length = length

    def worker(self, img):
        h, w = img.shape[2], img.shape[3]  # Tensor [1][2],  nparray [0][1]
        mask = np.ones((h, w), np.float32)
        y = random.randint(0, h)
        x = random.randint(0, w)
        length_new = random.randint(1, self.length)

        y1 = np.clip(y - length_new // 2, 0, h)
        y2 = np.clip(y + length_new // 2, 0, h)
        x1 = np.clip(x - length_new // 2, 0, w)
        x2 = np.clip(x + length_new // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img[:, 0, :, :])
        img[:, 0, :, :] *= mask
        return img

    def __call__(self, args):
        img_faces, img_lbp, depth_map = args
        return self.worker(img_faces), img_lbp, depth_map


# Tensor
class Normaliztion(object):

    def worker(self, img_faces, img_lbp, depth_map):
        return (img_faces - 127.5) / 127.5, (img_lbp - 127.5) / 127.5, depth_map / 255.0  # [-1,1], [0,1]

    def __call__(self, args):
        img_faces, img_lbp, depth_map = args
        return self.worker(img_faces, img_lbp, depth_map)


class Spoofing_train(Dataset):

    def __init__(self, info_list, image_dir, map_dir, args, transform=None):

        class Record(object):
            def __init__(self, row):
                self._data = row.strip().split(',')
                self.path = self._data[1]
                self.label = eval(self._data[0])  # real 1 photo -1 video -2
                self.video_length = int(self._data[2])  # the num of frames in the entire video

        self.video_records = [Record(x) for x in open(info_list)]
        self.image_dir = image_dir
        self.map_dir = map_dir
        self.test = args.test
        self.map_size = args.mapsize
        self.input_size = args.inputsize
        self.length = args.length + 1
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
            img_face = cv2.resize(self._crop_face_from_scene(img, face_scale), (self.input_size, self.input_size))
            img_faces.append(img_face)
            if i == 0:  # the first frame is used for 2D conv and the rest for 3D conv
                img_lbp = np.zeros_like(img_face)
                for c in range(3):
                    img_lbp[:, :, c] = local_binary_pattern(img_face[:, :, c], 8, 1)
                if spoofing_label == 1:
                    map = cv2.imread(os.path.join(self.map_dir, path, '%d.jpg' % image_id), 0)
                    depth_map = cv2.resize(self._crop_face_from_scene(map, face_scale), (self.map_size, self.map_size))
                else:
                    depth_map = np.zeros((self.map_size, self.map_size))

        seq = iaa.Sequential([
            iaa.Add(value=(-40, 40), per_channel=True),  # Add color
            iaa.GammaContrast(gamma=(0.5, 1.5))  # GammaContrast with a gamma of 0.5 to 1.5
        ])

        if not self.test:
            img_faces[0] = seq.augment_image(img_faces[0])

        # transform
        img_faces, img_lbp, depth_map = self.transform((img_faces, img_lbp, depth_map))

        return img_faces, depth_map, img_lbp, spoofing_label, path
