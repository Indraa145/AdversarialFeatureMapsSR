import os
import numpy as np
import cv2
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.data_train import faces_data, High_Data, Low_Data
from data.data_test import get_loader

from models.discriminator_vgg import Discriminator #Discriminator
from models.generator_l2h import Low2High #Low2High Generator
from models.generator_h2l import High2Low #High2Low Generator

from pretrained_nets.modif_vgg16 import modif_vgg16
from utils.csv_utils import data_write_csv

import argparse

#Command line configuration
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--gpu", action="store", dest="gpu", help="separate numbers with commas, eg. 3,4,5", required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    gpus = args.gpu.split(",")
    n_gpu = len(gpus)

    #Seed number (used for randomization)
    seed_num = 2020
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #Setting the hyperparameters
    max_epoch = 5
    learn_rate = 1e-4
    alpha, beta = 1, 0.05

    #Initialize the models and loss function
    G_h2l = High2Low().cuda()
    G_l2h = Low2High().cuda()
    D_h2l = Discriminator(16, 3, True).cuda()
    D_l2h_64 = Discriminator(64, 64, False).cuda()
    D_l2h_32 = Discriminator(32, 128, False).cuda()
    D_l2h_16 = Discriminator(16, 256, False).cuda()
    mse = nn.MSELoss()

    #Setting the optimizers
    # filter is making sure only the model trainable parameters are used (requires grad params)
    optim_D_h2l = optim.Adam(filter(lambda p: p.requires_grad, D_h2l.parameters()), lr=learn_rate, betas=(0.0, 0.9))
    optim_G_h2l = optim.Adam(G_h2l.parameters(), lr=learn_rate, betas=(0.0, 0.9))
    optim_D_l2h_64 = optim.Adam(filter(lambda p: p.requires_grad, D_l2h_64.parameters()), lr=learn_rate, betas=(0.0, 0.9))
    optim_D_l2h_32 = optim.Adam(filter(lambda p: p.requires_grad, D_l2h_32.parameters()), lr=learn_rate, betas=(0.0, 0.9))
    optim_D_l2h_16 = optim.Adam(filter(lambda p: p.requires_grad, D_l2h_16.parameters()), lr=learn_rate, betas=(0.0, 0.9))
    optim_G_l2h = optim.Adam(G_l2h.parameters(), lr=learn_rate, betas=(0.0, 0.9))

    #Load the dataset (Train Data)
    data = faces_data(High_Data, Low_Data)
    loader = DataLoader(dataset=data, batch_size=32, shuffle=True)

    #Load the dataset (Test Data)
    test_loader = get_loader("widerfacetest", bs=1)
    num_test = 12
    test_save = "intermid_results_revised"

    mod_vgg16 = modif_vgg16().cuda()

    loss_history = [[]]
    loss_data = []
    #Training
    for ep in range(1, max_epoch+1):
        G_h2l.train()
        D_h2l.train()
        G_l2h.train()
        D_l2h_64.train()
        D_l2h_32.train()
        D_l2h_16.train()
        for i, batch in enumerate(loader):
            optim_D_h2l.zero_grad()
            optim_G_h2l.zero_grad()
            optim_D_l2h_64.zero_grad()
            optim_D_l2h_32.zero_grad()
            optim_D_l2h_16.zero_grad()
            optim_G_l2h.zero_grad()

            zs = batch["z"].cuda() # Noises
            lrs = batch["lr"].cuda() # LR Dataset
            hrs = batch["hr"].cuda() # HR Dataset
            downs = batch["hr_down"].cuda() # Bicubic downsampled HR dataset

            lr_gen = G_h2l(hrs, zs) # Generated LR
            lr_gen_detach = lr_gen.detach()
            hr_gen = G_l2h(lr_gen_detach) # Generated HR

            # Putting the generated images to the modified vgg16
            hr_gen_vgg16 = mod_vgg16(hr_gen)  # tuple len 3
            hr_gen_vgg16_64 = hr_gen_vgg16[0]
            hr_gen_vgg16_32 = hr_gen_vgg16[1]
            hr_gen_vgg16_16 = hr_gen_vgg16[2]
            hr_gen_vgg16_64_detach = hr_gen_vgg16_64.detach()
            hr_gen_vgg16_32_detach = hr_gen_vgg16_32.detach()
            hr_gen_vgg16_16_detach = hr_gen_vgg16_16.detach()

            # Putting the original images to the modified vgg16
            hrs_vgg16 = mod_vgg16(hrs)
            hrs_vgg16_64 = hrs_vgg16[0]
            hrs_vgg16_32 = hrs_vgg16[1]
            hrs_vgg16_16 = hrs_vgg16[2]
            hrs_vgg16_64_detach = hrs_vgg16_64.detach()
            hrs_vgg16_32_detach = hrs_vgg16_32.detach()
            hrs_vgg16_16_detach = hrs_vgg16_16.detach()

            # update discriminator
            loss_D_h2l = nn.ReLU()(1.0 - D_h2l(lrs)).mean() + nn.ReLU()(1 + D_h2l(lr_gen_detach)).mean()
            loss_D_l2h_64 = nn.ReLU()(1.0 - D_l2h_64(hrs_vgg16_64_detach)).mean() + nn.ReLU()(1 + D_l2h_64(hr_gen_vgg16_64_detach)).mean()
            loss_D_l2h_32 = nn.ReLU()(1.0 - D_l2h_32(hrs_vgg16_32_detach)).mean() + nn.ReLU()(1 + D_l2h_32(hr_gen_vgg16_32_detach)).mean()
            loss_D_l2h_16 = nn.ReLU()(1.0 - D_l2h_16(hrs_vgg16_16_detach)).mean() + nn.ReLU()(1 + D_l2h_16(hr_gen_vgg16_16_detach)).mean()

            loss_D_h2l.backward()
            loss_D_l2h_64.backward()
            loss_D_l2h_32.backward()
            loss_D_l2h_16.backward()

            optim_D_h2l.step()
            optim_D_l2h_64.step()
            optim_D_l2h_32.step()
            optim_D_l2h_16.step()

            # update generator
            optim_D_h2l.zero_grad()
            gan_loss_h2l = -D_h2l(lr_gen).mean()
            mse_loss_h2l = mse(lr_gen, downs)

            loss_G_h2l = alpha * mse_loss_h2l + beta * gan_loss_h2l
            loss_G_h2l.backward()
            optim_G_h2l.step()

            optim_D_l2h_64.zero_grad()
            optim_D_l2h_32.zero_grad()
            optim_D_l2h_16.zero_grad()
            gan_loss_l2h = -D_l2h_64(hr_gen_vgg16_64).mean() + -D_l2h_32(hr_gen_vgg16_32).mean() + -D_l2h_16(hr_gen_vgg16_16).mean()
            mse_loss_l2h = mse(hr_gen, hrs)

            loss_G_l2h = alpha * mse_loss_l2h + beta * gan_loss_l2h
            loss_G_l2h.backward()
            optim_G_l2h.step()

            print(" {}({}) D_l2h_64: {:.3f}, D_l2h_32: {:.3f}, D_l2h_16: {:.3f}, D_h2l: {:.3f}, "
                  "G_l2h: {:.3f}, G_h2l: {:.3f}, mse_l2h: {:.3f}, mse_h2l: {:.3f}, total_l2h: {:.3f}, total_h2l: {:.3f} \r"
                  .format(i+1, ep, loss_D_l2h_64.item(), loss_D_l2h_32.item(), loss_D_l2h_16.item(), loss_D_h2l.item(),
                          gan_loss_l2h.item(), gan_loss_h2l.item(), mse_loss_l2h.item(), mse_loss_h2l.item(), loss_G_l2h.item(), loss_G_h2l.item()), end=" ")
            loss_data = [ep, loss_D_l2h_64.item(), loss_D_l2h_32.item(), loss_D_l2h_16.item(), loss_D_h2l.item(),
                         gan_loss_l2h.item(), gan_loss_h2l.item(), mse_loss_l2h.item(), mse_loss_h2l.item(), loss_G_l2h.item(), loss_G_h2l.item()]
        print("\n Testing and saving...")
        loss_history.append(loss_data)
        data_write_csv("{}/csv/results_{}.csv".format(test_save, ep), loss_history)  # Write the loss to csv file

        # Testing with intermid results
        G_h2l.eval()
        D_h2l.eval()
        G_l2h.eval()
        D_l2h_64.eval()
        D_l2h_32.eval()
        D_l2h_16.eval()

        for i, sample in enumerate(test_loader):
            if i >= num_test:
                break
            low_temp = sample["img16"].numpy()
            low = torch.from_numpy(np.ascontiguousarray(low_temp[:, ::-1, :, :])).cuda()
            with torch.no_grad():
                high_gen = G_l2h(low)
            np_gen = high_gen.detach().cpu().numpy().transpose(0, 2, 3, 1).squeeze(0)
            np_gen = (np_gen - np_gen.min()) / (np_gen.max() - np_gen.min())
            np_gen = (np_gen * 255).astype(np.uint8)
            cv2.imwrite("{}/imgs/{}_{}_sr.png".format(test_save, ep, i+1), np_gen)
        save_file = "{}/models/model_epoch_{:03d}.pth".format(test_save, ep)
        torch.save({"G_h2l": G_h2l.state_dict(), "D_h2l": D_h2l.state_dict(),
                    "G_l2h": G_l2h.state_dict(), "D_l2h_64": D_l2h_64.state_dict(),
                    "D_l2h_32": D_l2h_32.state_dict(), "D_l2h_16": D_l2h_16.state_dict()}, save_file)
        print("saved: ", save_file)
    print("finished.")