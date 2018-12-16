# -*- coding: utf-8 -*-

import os
from glob import glob
import random
import random
import time
import datetime
import sys

from torch.autograd import Variable
import torch
from visdom import Visdom
import numpy as np
import pandas as pd
import skimage.transform
import skimage.io


from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

def data_generator(path, batch_size):
    file_list = glob(path)
    idx = 0
    list_length = len(file_list)
    while True:
        batch = []
        for _ in range(batch_size):
            if idx >= list_length:
                idx = 0
                random.shuffle(file_list)
            batch.append(file_list[idx])
            idx += 1
        yield batch

def get_image(file_path, input_hw, is_random_clip=True, is_random_flip=True):
    """
    画像の読み込み・サイズ変更・データ拡張を行うメソッドです。
    :param file_path: str
        画像ファイルのパス。
    :param input_hw: int
        出力画像の画像の1辺の大きさ。
    :param is_random_clip: bool
        画像からランダムな領域切り出しを行うかどうか。
    :param is_random_flip: bool
        画像にランダムに水平方向の反転を加えるかどうか。
    :return: ndarray
        出力画像。
    """
    image = imread(file_path)
    h, w, c = image.shape
    if is_random_clip and (h >= input_hw) and (w >= input_hw):
        image = random_clip(image, input_hw)
    else:
        image = center_crop(image)
        image = skimage.transform.resize(image, [input_hw, input_hw])

    is_flip = random.choice([True, False])
    if is_random_flip and is_flip:
        image = image[:, ::-1, :]

    return image


def imread(file_path):
    """
    画像の読み込みを行うメソッドです。
    :param file_path: str
        入力画像のファイルパス。
    :return: ndarray
        読み込んだ画像を [-1, 1]に規格化して返します。
    """
    image = skimage.io.imread(file_path).astype(np.float32)
    return image / 127.5 - 1 #0→255を-1→1に変換

def random_clip(image, input_hw):
    """
    画像からランダムな正方形領域を抽出するメソッドです。
    :param image: ndarray
        入力画像
    :param input_hw: int
        抽出する正方形の1辺の長さ
    :return: ndarray
        出力画像
    """
    h ,w, c = image.shape
    if h == input_hw:
        random_y = 0
    else:
        random_y = random.randint(0, h - input_hw)

    if w == input_hw:
        random_x = 0
    else:
        random_x = random.randint(0, w - input_hw)
    return image[random_y:random_y+input_hw, random_x:random_x+input_hw, :]


def center_crop(image):
    """
    画像から可能な限り大きな正方形領域を抽出するメソッドです。
    入力画像の中央から取得します。
    :param image: ndarray
        入力画像。
    :return: ndarray
        抽出後の出力画像。
    """
    h, w, c = image.shape
    if h >= w:
        crop_wh = w
        sub = int((h - w) // 2)
        trimmed = image[sub:sub+crop_wh, :, :]
    else:
        crop_wh = h
        sub = int((w - h) // 2)
        trimmed = image[:, sub:sub+crop_wh, :]
    return trimmed


def output_sample_image(path, combine_image):
    """
    画像を出力するメソッドです。
    :param path: str
        出力先ファイルパス。
    :param combine_image:  ndarray
        出力対象の画像です。[-1, 1] に規格化されているものを想定しています。
    :return: None
    """
    image = (combine_image+1) * 127.5
    image = np.transpose(image, axes=(1, 2, 0))
    skimage.io.imsave(path, image.astype(np.uint8))


class ImageDataset(Dataset):
    def __init__(self, root, dataset_A, dataset_B, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob(os.path.join(root, dataset_A) + '/*.*'))
        self.files_B = sorted(glob(os.path.join(root, dataset_B) + '/*.*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return min(len(self.files_A), len(self.files_B))



class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def tensor2image(tensor):
    image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)


class Logger():
    def __init__(self, n_epochs, batches_epoch):
        self.viz = Visdom()
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}
        self.loss_df = pd.DataFrame(columns=["epoch", "batch" "loss_D", "loss_G",
                                          "adversarial_loss", "cycle_loss"])


    def log(self, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))
        """
        if self.batch % 10 == 0:
            temp_df = pd.DataFrame({"epoch": self.epoch, "batch":self.batch,
                                    "loss_D": losses["loss_D"].data[0], "loss_G":losses["loss_G"].data[0],
                                    "adversarial_loss":losses["loss_G_GAN"].data[0],
                                    "cycle_loss":losses["loss_G_cycle"].data[0]})
            self.loss_df = pd.concat([self.loss_df, temp_df], axis=0)
        """

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name].data[0]
            else:
                self.losses[loss_name] += losses[loss_name].data[0]

            if (i+1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batch))

        batches_done = self.batches_epoch*(self.epoch - 1) + self.batch
        batches_left = self.batches_epoch*(self.n_epochs - self.epoch) + self.batches_epoch - self.batch
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/batches_done)))

        # Draw images
        for image_name, tensor in images.items():
            if image_name not in self.image_windows:
                self.image_windows[image_name] = self.viz.image(tensor2image(tensor.data), opts={'title':image_name})
            else:
                self.viz.image(tensor2image(tensor.data), win=self.image_windows[image_name], opts={'title':image_name})

        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            # Plot losses
            for loss_name, loss in self.losses.items():
                if loss_name not in self.loss_windows:
                    self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]),
                                                                    opts={'xlabel': 'epochs', 'ylabel': loss_name, 'title': loss_name})
                else:
                    self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), win=self.loss_windows[loss_name], update='append')
                # Reset losses for next epoch
                self.losses[loss_name] = 0.0

            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1

