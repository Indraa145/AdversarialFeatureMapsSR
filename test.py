# New Test Code
import os
os.sys.path.append(os.getcwd())
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from easydict import EasyDict as edict
from data.data_test import get_loader
from torch.autograd import Variable
from models.generator_l2h import Low2High
import cv2

def to_var(data):
    real_cpu = data
    batchsize = real_cpu.size(0)
    input = Variable(real_cpu.cuda())
    return input, batchsize

def main():
    torch.manual_seed(1)
    np.random.seed(0)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    opt = edict()
    opt.nGPU = 1
    opt.batchsize = 1
    opt.cuda = True
    cudnn.benchmark = True
    print('========================LOAD DATA============================')
    data_name = 'widerfacetest'
    test_loader = get_loader(data_name, opt.batchsize)
    net_G_low2high = Low2High().cuda()

    #pth format
    a = torch.load('./intermid_results_revised/resnet/models/model_epoch_005.pth')
    net_G_low2high.load_state_dict(a['G_l2h'])

    net_G_low2high = net_G_low2high.eval()
    index = 0
    test_file = 'test_res'
    if not os.path.exists(test_file):
        os.makedirs(test_file)
    for idx, data_dict in enumerate(test_loader):
        print(idx)
        index = index + 1
        data_low = data_dict['img16'].numpy()
        low = torch.from_numpy(np.ascontiguousarray(data_low[:, ::-1, :, :])).cuda()
        img_name = data_dict['imgpath'][0].split('/')[-1]
        img_name = os.path.basename(img_name)
        with torch.no_grad():
            data_high_output = net_G_low2high(low)

        np_gen = data_high_output.detach().cpu().numpy().transpose(0, 2, 3, 1).squeeze(0)
        np_gen = (np_gen - np_gen.min()) / (np_gen.max() - np_gen.min())
        np_gen = (np_gen * 255).astype(np.uint8)
        cv2.imwrite("{}/{}.jpg".format(test_file, img_name.split('.')[0]), np_gen)

if __name__ == '__main__':
    main()
