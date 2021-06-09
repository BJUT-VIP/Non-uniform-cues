import os
import numpy as np
import torch
import shutil
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.autograd import Variable
import sklearn
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import pdb
import logging
import time


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


# def get_threhold(scores, labels):
#     # Since the classification is based on the image mean value,
#     # the mean value with largest spacing is selected as the threshold
#     scores.sort()
#     spacing = []
#     for i in range(1, len(scores)):
#         spacing.append(scores[i]-scores[i-1])
#     idx = spacing.index(max(spacing))
#     threhold = (scores[idx]+scores[idx+1])/2
#     return threhold

def get_threhold(scores, labels, num_real, num_fake):
    sorts = scores[:]
    sorts.sort()
    thrs = [0]
    for i in range(1, len(sorts)):  # count all possible thrs
        thr = (sorts[i] + sorts[i - 1]) / 2
        if thr != thrs[-1]:
            thrs.append(thr)
    acers = []
    for thr in thrs:  # calculate ACER under all thrs
        num_err_fake = num_err_real = 0
        for i in range(len(scores)):
            if scores[i] <= thr and labels[i] == 1:
                num_err_real += 1
            elif scores[i] > thr and labels[i] != 1:
                num_err_fake += 1
        val_APCER = num_err_fake / num_fake
        val_BPCER = num_err_real / num_real
        acers.append((val_APCER + val_BPCER) / 2.0)
    idx = acers.index(min(acers))
    return thrs[idx]


def performances(map_score_val_filename, map_score_test_filename):
    # val
    with open(map_score_val_filename, 'r') as file:
        lines = file.readlines()
    val_scores = []
    val_labels = []
    data = []
    num_real = 1e-5
    num_fake = 1e-5
    for line in lines:
        token = line.split()
        score = float(token[1])
        val_scores.append(score)
        label = float(token[2])
        val_labels.append(label)
        if label == 1:
            num_real += 1
        else:
            num_fake += 1
        data.append({'map_score': score, 'label': label, 'path': token[0]})

    val_threshold = get_threhold(val_scores, val_labels, num_real, num_fake)

    num_err_fake = num_err_real = 0
    for i, s in enumerate(data):
        if s['map_score'] <= val_threshold and s['label'] == 1:
            lines[i] = lines[i].strip() + ' err 0\n'
            num_err_real += 1
        if s['map_score'] > val_threshold and s['label'] != 1:
            lines[i] = lines[i].strip() + ' err 1\n'
            num_err_fake += 1
    with open(map_score_val_filename, 'w') as file:
        file.writelines(lines)

    val_APCER = num_err_fake / num_fake
    val_BPCER = num_err_real / num_real
    val_ACER = (val_APCER + val_BPCER) / 2.0

    # test
    with open(map_score_test_filename, 'r') as file:
        lines = file.readlines()
    test_scores = []
    test_labels = []
    data = []
    num_real = 1e-5
    num_fake = 1e-5
    for line in lines:
        token = line.split()
        score = float(token[1])
        test_scores.append(score)
        label = float(token[2])
        test_labels.append(label)
        if label == 1:
            num_real += 1
        else:
            num_fake += 1
        data.append({'map_score': score, 'label': label, 'path': token[0]})

    # test based on val_threshold
    num_err_fake = num_err_real = 0
    for i, s in enumerate(data):
        if s['map_score'] <= val_threshold and s['label'] == 1:
            lines[i] = lines[i].strip() + ' err 0\n'
            num_err_real += 1
        if s['map_score'] > val_threshold and s['label'] != 1:
            lines[i] = lines[i].strip() + ' err 1\n'
            num_err_fake += 1
    with open(map_score_test_filename, 'w') as file:
        file.writelines(lines)

    test_APCER = num_err_fake / num_fake
    test_BPCER = num_err_real / num_real
    test_ACER = (test_APCER + test_BPCER) / 2.0

    # test based on test_threshold
    test_threshold = get_threhold(test_scores, test_labels, num_real, num_fake)

    num_err_real = len([s for s in data if s['map_score'] <= test_threshold and s['label'] == 1])
    num_err_fake = len([s for s in data if s['map_score'] > test_threshold and s['label'] != 1])

    test_threshold_APCER = num_err_fake / num_fake
    test_threshold_BPCER = num_err_real / num_real
    test_threshold_ACER = (test_threshold_APCER + test_threshold_BPCER) / 2.0

    return val_APCER, val_BPCER, val_ACER, val_threshold, test_APCER, test_BPCER, test_ACER, test_threshold_ACER


def plot_embedding(data, label, outfile=None, name=''):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    real = [[], []]
    photo = [[], []]
    video = [[], []]
    real_test = [[], []]
    photo_test = [[], []]
    video_test = [[], []]
    for n, l in enumerate(label):
        if l == 1:
            real[0].append(data[n][0])
            real[1].append(data[n][1])
        elif l == -1:
            photo[0].append(data[n][0])
            photo[1].append(data[n][1])
        elif l == -2:
            video[0].append(data[n][0])
            video[1].append(data[n][1])
        elif l == 5:
            real_test[0].append(data[n][0])
            real_test[1].append(data[n][1])
        elif l == 3:
            photo_test[0].append(data[n][0])
            photo_test[1].append(data[n][1])
        elif l == 2:
            video_test[0].append(data[n][0])
            video_test[1].append(data[n][1])
    plt.scatter(real[0], real[1], color='g', marker='2', s=20, linewidth=1, label='Living (val)')
    plt.scatter(photo[0], photo[1], color='b', marker='3', s=15, linewidth=1, label='Photo (val)')
    plt.scatter(video[0], video[1], color='r', marker='4', s=15, linewidth=1, label='Video (val)')
    plt.scatter(real_test[0], real_test[1], color='g', marker='^', s=35, linewidth=0, label='Living (test)')
    plt.scatter(photo_test[0], photo_test[1], color='b', marker='<', s=30, linewidth=0, label='Photo (test)')
    plt.scatter(video_test[0], video_test[1], color='r', marker='>', s=30, linewidth=0, label='Video (test)')

    for i, n in enumerate(name):
        plt.text(data[i, 0], data[i, 1], n, fontsize=1, color='black')
    # plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.legend()
    plt.title('t-SNE %s' % os.path.basename(outfile))
    if outfile:
        plt.savefig(outfile, dpi=600, format='pdf')
    else:
        plt.show()


def init_logging(log_file=None, file_mode='w', overwrite_flag=False, log_level=logging.INFO):
    # basically, the basic log offers console output
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s[%(levelname)s]: %(message)s')
    console_handler.setFormatter(formatter)

    logging.getLogger().setLevel(log_level)
    logging.getLogger().addHandler(console_handler)

    if not log_file or log_file == '':
        print('----------------------------------------------------------')
        print('No log file is specified. The log information is only displayed in command line.')
        print('----------------------------------------------------------')
        return

    # check that the log_file is already existed or not
    if not os.path.exists(log_file):
        location_dir = os.path.dirname(log_file)
        if not os.path.exists(location_dir):
            os.makedirs(location_dir)

        file_handler = logging.FileHandler(filename=log_file, mode=file_mode)
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)

        print('The logging is successfully init. The log file is created.')
    else:
        if overwrite_flag:
            print('The file [%s] is existed. And it is to be handled according to the arg [file_mode]' % log_file)
            file_handler = logging.FileHandler(filename=log_file, mode=file_mode)
            file_handler.setFormatter(formatter)
            logging.getLogger().addHandler(file_handler)
        else:
            print('The file [%s] is existed. The [overwrite_flag] is False, please change the [log_file].' % log_file)
            quit()


if __name__ == '__main__':
    # print(performances('texture_P3-5/texture_P3-5_map_score_val.txt',
    #                    'texture_P3-5/texture_P3-5_map_score_test.txt'))
    data = np.load('/home/yaowen/Documents/PycharmProjects/CDCN-master/ST-CDCN/T-CDCN/T-CDCN_P4-6/test_tsne.npz', allow_pickle=True)
    Y= data['Y']
    label_all = data['label_all']
    name = data['name']
    plot_embedding(Y, label_all, './sample.pdf', name=name)
