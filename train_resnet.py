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

from models.discriminator_resnet import Discriminator #Discriminator
from models.generator_l2h import Low2High #Low2High Generator
from models.generator_h2l import High2Low #High2Low Generator

from pretrained_nets.modif_resnet import modif_resnet
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
    D_l2h = Discriminator(16, False).cuda()
    D_h2l = Discriminator(16, True).cuda()
    mse = nn.MSELoss()

    #Setting the optimizers
    optim_D_h2l = optim.Adam(filter(lambda p: p.requires_grad, D_h2l.parameters()), lr=learn_rate, betas=(0.0, 0.9))
    optim_G_h2l = optim.Adam(G_h2l.parameters(), lr=learn_rate, betas=(0.0, 0.9))
    optim_D_l2h = optim.Adam(filter(lambda p: p.requires_grad, D_l2h.parameters()), lr=learn_rate, betas=(0.0, 0.9))
    optim_G_l2h = optim.Adam(G_l2h.parameters(), lr=learn_rate, betas=(0.0, 0.9))

    #Load the dataset (Train Data)
    data = faces_data(High_Data, Low_Data)
    loader = DataLoader(dataset=data, batch_size=32, shuffle=True)

    #Load the dataset (Test Data)
    test_loader = get_loader("widerfacetest", bs=1)
    num_test = 12
    test_save = "intermid_results_revised"

    # Load Modified PreTrained ResNet18 Model
    mod_resnet = modif_resnet(output_layer='layer1').cuda()

    # Freeze the weights
    for param in mod_resnet.parameters():
        param.require_grad = False

    loss_history = [[]]
    loss_data = []
    #Training
    for ep in range(1, max_epoch+1):
        G_h2l.train()
        D_h2l.train()
        G_l2h.train()
        D_l2h.train()
        for i, batch in enumerate(loader):
            optim_D_h2l.zero_grad()
            optim_G_h2l.zero_grad()
            optim_D_l2h.zero_grad()
            optim_G_l2h.zero_grad()

            zs = batch["z"].cuda() # Noises
            lrs = batch["lr"].cuda() # LR Dataset
            hrs = batch["hr"].cuda() # HR Dataset
            downs = batch["hr_down"].cuda() # Bicubic downsampled HR dataset

            lr_gen = G_h2l(hrs, zs)
            lr_gen_detach = lr_gen.detach()
            hr_gen = G_l2h(lr_gen_detach)

            # Putting the generated images to the modified resnet18
            hr_gen_resnet18 = mod_resnet(hr_gen) #(B, 64, 16, 16)
            hr_gen_resnet18_detach = hr_gen_resnet18.detach()

            # Putting the original images to the modified resnet18
            hrs_resnet18 = mod_resnet(hrs)
            hrs_resnet18_detach = hrs_resnet18.detach()

            # update discriminator
            loss_D_h2l = nn.ReLU()(1.0 - D_h2l(lrs)).mean() + nn.ReLU()(1 + D_h2l(lr_gen_detach)).mean()
            loss_D_l2h = nn.ReLU()(1.0 - D_l2h(hrs_resnet18_detach)).mean() + nn.ReLU()(1 + D_l2h(hr_gen_resnet18_detach)).mean()
            loss_D_h2l.backward()
            loss_D_l2h.backward()
            optim_D_h2l.step()
            optim_D_l2h.step()

            # update generator
            optim_D_h2l.zero_grad()
            gan_loss_h2l = -D_h2l(lr_gen).mean()
            mse_loss_h2l = mse(lr_gen, downs)

            loss_G_h2l = alpha * mse_loss_h2l + beta * gan_loss_h2l
            loss_G_h2l.backward()
            optim_G_h2l.step()

            optim_D_l2h.zero_grad()
            gan_loss_l2h = -D_l2h(hr_gen_resnet18).mean()
            mse_loss_l2h = mse(hr_gen, hrs)

            loss_G_l2h = alpha * mse_loss_l2h + beta * gan_loss_l2h
            loss_G_l2h.backward()
            optim_G_l2h.step()

            print(" {}({}) D_l2h: {:.3f}, D_h2l: {:.3f}, G_l2h: {:.3f}, G_h2l: {:.3f}, mse_l2h: {:.3f}, mse_h2l: {:.3f}, total_l2h: {:.3f}, total_h2l: {:.3f} \r"
                  .format(i+1, ep, loss_D_l2h.item(), loss_D_h2l.item(), gan_loss_l2h.item(), gan_loss_h2l.item(),
                          mse_loss_l2h.item(), mse_loss_h2l.item(), loss_G_l2h.item(), loss_G_h2l.item()), end=" ")
            loss_data = [ep, loss_D_l2h.item(), loss_D_h2l.item(), gan_loss_l2h.item(), gan_loss_h2l.item(),
                         mse_loss_l2h.item(), mse_loss_h2l.item(), loss_G_l2h.item(), loss_G_h2l.item()]
        print("\n Testing and saving...")
        loss_history.append(loss_data)
        data_write_csv("{}/csv/results_{}.csv".format(test_save, ep), loss_history)  # Write the loss to csv file

        # Testing with intermid results
        G_h2l.eval()
        D_h2l.eval()
        G_l2h.eval()
        D_l2h.eval()

        for i, sample in enumerate(test_loader):
            if i >= num_test:
                break
            low_temp = sample["img16"].numpy()
            low = torch.from_numpy(np.ascontiguousarray(low_temp[:, ::-1, :, :])).cuda()
            with torch.no_grad():
                hign_gen = G_l2h(low)
            np_gen = hign_gen.detach().cpu().numpy().transpose(0, 2, 3, 1).squeeze(0)
            np_gen = (np_gen - np_gen.min()) / (np_gen.max() - np_gen.min())
            np_gen = (np_gen * 255).astype(np.uint8)
            cv2.imwrite("{}/imgs/{}_{}_sr.png".format(test_save, ep, i+1), np_gen)
        save_file = "{}/models/model_epoch_{:03d}.pth".format(test_save, ep)
        torch.save({"G_h2l": G_h2l.state_dict(), "D_h2l": D_h2l.state_dict(),
                    "G_l2h": G_l2h.state_dict(), "D_l2h": D_l2h.state_dict()}, save_file)
        print("saved: ", save_file)
    print("finished.")