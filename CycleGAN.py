
import itertools
import os
import datetime

import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler
from torch.cuda import FloatTensor, LongTensor

from net_utils import Generator, Discriminator, weights_init_normal, ReplayBuffer, Generator_BN, Discriminator_BN
from utils import ImageDataset, LambdaLR, Logger, output_sample_image


class CycleGAN():
    def __init__(self, config):
        self.config = config


    def train(self):
        num_channels = self.config.NUM_CHANNELS
        use_cuda = self.config.USE_CUDA
        lr = self.config.LEARNING_RATE

        # Networks
        netG_A2B = Generator(num_channels)
        netG_B2A = Generator(num_channels)
        netD_A = Discriminator(num_channels)
        netD_B = Discriminator(num_channels)

        #netG_A2B = Generator_BN(num_channels)
        #netG_B2A = Generator_BN(num_channels)
        #netD_A = Discriminator_BN(num_channels)
        #netD_B = Discriminator_BN(num_channels)

        if use_cuda:
            netG_A2B.cuda()
            netG_B2A.cuda()
            netD_A.cuda()
            netD_B.cuda()

        netG_A2B.apply(weights_init_normal)
        netG_B2A.apply(weights_init_normal)
        netD_A.apply(weights_init_normal)
        netD_B.apply(weights_init_normal)

        criterion_GAN = torch.nn.BCELoss()
        criterion_cycle = torch.nn.L1Loss()
        criterion_identity = torch.nn.L1Loss()

        optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                       lr=lr, betas=(0.5, 0.999))
        optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=lr, betas=(0.5, 0.999))

        lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(self.config.EPOCH, 0,
                                                                                           self.config.EPOCH//2).step)
        lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(self.config.EPOCH, 0,
                                                                                           self.config.EPOCH//2).step)
        lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(self.config.EPOCH, 0,
                                                                                           self.config.EPOCH//2).step)

        # Inputs & targets memory allocation
        #Tensor = LongTensor if use_cuda else torch.Tensor
        batch_size = self.config.BATCH_SIZE
        height, width, channels = self.config.INPUT_SHAPE

        input_A = FloatTensor(batch_size, channels, height, width)
        input_B = FloatTensor(batch_size, channels, height, width)
        target_real = Variable(FloatTensor(batch_size).fill_(1.0), requires_grad=False)
        target_fake = Variable(FloatTensor(batch_size).fill_(0.0), requires_grad=False)

        fake_A_buffer = ReplayBuffer()
        fake_B_buffer = ReplayBuffer()

        transforms_ = [transforms.RandomCrop((height, width)),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        dataloader = DataLoader(ImageDataset(self.config.DATA_DIR, self.config.DATASET_A, self.config.DATASET_B,
                                             transforms_=transforms_, unaligned=True),
                                             batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
        # Loss plot
        logger = Logger(self.config.EPOCH, len(dataloader))

        now = datetime.datetime.now()
        datetime_sequence = "{0}{1:02d}{2:02d}_{3:02}{4:02d}".format(str(now.year)[-2:], now.month, now.day ,
                                                                    now.hour, now.minute)

        output_name_1 = self.config.DATASET_A + "2" + self.config.DATASET_B
        output_name_2 = self.config.DATASET_B + "2" + self.config.DATASET_A

        experiment_dir = os.path.join(self.config.RESULT_DIR, datetime_sequence)

        sample_output_dir_1 = os.path.join(experiment_dir, "sample", output_name_1)
        sample_output_dir_2 = os.path.join(experiment_dir, "sample", output_name_2)
        weights_output_dir_1 = os.path.join(experiment_dir, "weights", output_name_1)
        weights_output_dir_2 = os.path.join(experiment_dir, "weights", output_name_2)
        weights_output_dir_resume = os.path.join(experiment_dir, "weights", "resume")

        os.makedirs(sample_output_dir_1, exist_ok=True)
        os.makedirs(sample_output_dir_2, exist_ok=True)
        os.makedirs(weights_output_dir_1, exist_ok=True)
        os.makedirs(weights_output_dir_2, exist_ok=True)
        os.makedirs(weights_output_dir_resume, exist_ok=True)

        counter = 0

        for epoch in range(self.config.EPOCH):
            """
            logger.loss_df.to_csv(os.path.join(experiment_dir,
                                 self.config.DATASET_A + "_"
                                 + self.config.DATASET_B + ".csv"),
                    index=False)
            """
            if epoch % 100 == 0:
                torch.save(netG_A2B.state_dict(), os.path.join(weights_output_dir_1, str(epoch).zfill(4) + 'netG_A2B.pth'))
                torch.save(netG_B2A.state_dict(), os.path.join(weights_output_dir_2, str(epoch).zfill(4) + 'netG_B2A.pth'))
                torch.save(netD_A.state_dict(), os.path.join(weights_output_dir_1, str(epoch).zfill(4) + 'netD_A.pth'))
                torch.save(netD_B.state_dict(), os.path.join(weights_output_dir_2, str(epoch).zfill(4) + 'netD_B.pth'))

            for i, batch in enumerate(dataloader):
                # Set model input
                real_A = Variable(input_A.copy_(batch['A']))
                real_B = Variable(input_B.copy_(batch['B']))

                ###### Generators A2B and B2A ######
                optimizer_G.zero_grad()

                # GAN loss
                fake_B = netG_A2B(real_A)
                pred_fake_B = netD_B(fake_B)
                loss_GAN_A2B = criterion_GAN(pred_fake_B, target_real)

                fake_A = netG_B2A(real_B)
                pred_fake_A = netD_A(fake_A)
                loss_GAN_B2A = criterion_GAN(pred_fake_A, target_real)

                # Cycle loss
                recovered_A = netG_B2A(fake_B)
                loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0

                recovered_B = netG_A2B(fake_A)
                loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0

                # Total loss
                loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
                loss_G.backward()

                optimizer_G.step()
                ###################################

                ###### Discriminator A ######
                optimizer_D_A.zero_grad()

                # Real loss
                pred_A = netD_A(real_A)
                loss_D_real = criterion_GAN(pred_A, target_real)

                # Fake loss
                fake_A_ = fake_A_buffer.push_and_pop(fake_A)
                pred_fake = netD_A(fake_A_.detach())
                loss_D_fake = criterion_GAN(pred_fake, target_fake)

                # Total loss
                loss_D_A = (loss_D_real + loss_D_fake) * 0.5
                loss_D_A.backward()

                optimizer_D_A.step()
                ###################################

                ###### Discriminator B ######
                optimizer_D_B.zero_grad()

                # Real loss
                pred_B = netD_B(real_B)
                loss_D_real = criterion_GAN(pred_B, target_real)

                # Fake loss
                fake_B_ = fake_B_buffer.push_and_pop(fake_B)
                pred_fake = netD_B(fake_B_.detach())
                loss_D_fake = criterion_GAN(pred_fake, target_fake)

                # Total loss
                loss_D_B = (loss_D_real + loss_D_fake) * 0.5
                loss_D_B.backward()

                optimizer_D_B.step()

                # Progress report (http://localhost:8097)
                logger.log({'loss_G': loss_G,
                            'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                            'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)},
                           images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})

                if counter % 500 == 0:
                    real_A_sample = real_A.cpu().detach().numpy()[0]
                    pred_A_sample = fake_A.cpu().detach().numpy()[0]
                    real_B_sample = real_B.cpu().detach().numpy()[0]
                    pred_B_sample = fake_B.cpu().detach().numpy()[0]
                    combine_sample_1 = np.concatenate([real_A_sample, pred_B_sample], axis=2)
                    combine_sample_2 = np.concatenate([real_B_sample, pred_A_sample], axis=2)

                    file_1 = "{0}_{1}.jpg".format(epoch, counter)
                    output_sample_image(os.path.join(sample_output_dir_1, file_1), combine_sample_1)
                    file_2 = "{0}_{1}.jpg".format(epoch, counter)
                    output_sample_image(os.path.join(sample_output_dir_2, file_2), combine_sample_2)

                counter += 1


            # Update learning rates
            lr_scheduler_G.step()
            lr_scheduler_D_A.step()
            lr_scheduler_D_B.step()

        torch.save(netG_A2B.state_dict(), os.path.join(weights_output_dir_1, str(self.config.EPOCH).zfill(4) + 'netG_A2B.pth'))
        torch.save(netG_B2A.state_dict(), os.path.join(weights_output_dir_2, str(self.config.EPOCH).zfill(4) + 'netG_B2A.pth'))
        torch.save(netD_A.state_dict(), os.path.join(weights_output_dir_1, str(self.config.EPOCH).zfill(4) + 'netD_A.pth'))
        torch.save(netD_B.state_dict(), os.path.join(weights_output_dir_2, str(self.config.EPOCH).zfill(4) + 'netD_B.pth'))

    def resume_train(self):
        pass