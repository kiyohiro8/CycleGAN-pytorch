# -*- coding: utf-8 -*-

import os

import torch
from torch.autograd import Variable
from torch.cuda import FloatTensor
import skimage.io
import numpy as np

from net_utils import Generator

model_path = "./trained/0130netG_B2A.pth"
input_base_dir = "./data"
image_dir_name = "chocolate_cake_test"

output_root_dir = "./predicted"

if __name__ == "__main__":
    input_dir = os.path.join(input_base_dir, image_dir_name)
    file_list = os.listdir(input_dir)
    path_list = [os.path.join(input_dir, file) for file in file_list]

    output_dir = os.path.join(output_root_dir, image_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    param = torch.load(model_path)
    model = Generator(num_channels=64).cuda()
    model.load_state_dict(param)
    for path in path_list:
        filename = os.path.basename(path)
        input_image = skimage.io.imread(path) / 127.5 -1
        input_image = np.transpose(input_image, (2, 0, 1))
        input_image = input_image[np.newaxis, :, :, :]

        translated_image = model(torch.from_numpy(input_image).type("torch.cuda.FloatTensor"))
        translated_image = translated_image.cpu().detach().numpy()[0]
        output_path = os.path.join(output_dir, filename)
        translated_image = np.transpose(translated_image, axes=(1, 2, 0))
        translated_image = ((translated_image + 1) * 127.5).astype(np.uint8)
        skimage.io.imsave(output_path, translated_image)



