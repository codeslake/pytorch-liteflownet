import torch
import torch.optim as optim
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

import time
import getopt
import math
import numpy
import os
import PIL
import PIL.Image
import sys
import collections
import numpy as np
from data_loader import Data_Loader
import matplotlib.pyplot as plt

from model import Network
from utils import *
from torch_utils import *
from ckpt_manager import CKPT_Manager
   

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def train(config):
    summary = SummaryWriter(config.LOG_DIR.log_scalar_train_itr)

    ## inputs 
    inputs = {'b_t_1':None, 'b_t':None, 's_t_1':None, 's_t':None}
    inputs = collections.OrderedDict(sorted(inputs.items(), key=lambda t:t[0]))

    ## model
    print(toGreen('Loading Model...'))
    moduleNetwork = Network().to(device)
    moduleNetwork.apply(weights_init)
    moduleNetwork_gt = Network().to(device)
    print(moduleNetwork)

    ## checkpoint manager
    ckpt_manager = CKPT_Manager(config.LOG_DIR.ckpt, config.mode, config.max_ckpt_num)
    moduleNetwork.load_state_dict(torch.load('./network/network-default.pytorch'))
    moduleNetwork_gt.load_state_dict(torch.load('./network/network-default.pytorch'))

    ## data loader
    print(toGreen('Loading Data Loader...'))
    data_loader = Data_Loader(config, is_train = True, name = 'train', thread_num = config.thread_num)
    data_loader_test = Data_Loader(config, is_train = False, name = "test", thread_num = config.thread_num)

    data_loader.init_data_loader(inputs)
    data_loader_test.init_data_loader(inputs)


    ## loss, optim
    print(toGreen('Building Loss & Optim...'))
    MSE_sum = torch.nn.MSELoss(reduction = 'sum')
    MSE_mean = torch.nn.MSELoss()
    optimizer = optim.Adam(moduleNetwork.parameters(), lr=config.lr_init, betas=(config.beta1, 0.999))
    errs = collections.OrderedDict()

    print(toYellow('======== TRAINING START ========='))
    max_epoch = 10000
    itr = 0
    for epoch in np.arange(max_epoch):

        # train
        while True:
            itr_time = time.time()

            inputs, is_end = data_loader.get_feed()
            if is_end: break

            if config.loss == 'image':
                flow_bb = torch.nn.functional.interpolate(input=moduleNetwork(inputs['b_t'], inputs['b_t_1']), size=(config.height, config.width), mode='bilinear', align_corners=False)
                flow_bs = torch.nn.functional.interpolate(input=moduleNetwork(inputs['b_t'], inputs['s_t_1']), size=(config.height, config.width), mode='bilinear', align_corners=False)
                flow_sb = torch.nn.functional.interpolate(input=moduleNetwork(inputs['s_t'], inputs['b_t_1']), size=(config.height, config.width), mode='bilinear', align_corners=False)
                flow_ss = torch.nn.functional.interpolate(input=moduleNetwork(inputs['s_t'], inputs['s_t_1']), size=(config.height, config.width), mode='bilinear', align_corners=False)

                with torch.no_grad():
                    flow_ss_gt = torch.nn.functional.interpolate(input=moduleNetwork_gt(inputs['s_t'], inputs['s_t_1']), size=(config.height, config.width), mode='bilinear', align_corners=False)
                    s_t_warped_ss_mask_gt = warp(tensorInput=torch.ones_like(inputs['s_t_1'], device = device), tensorFlow=flow_ss_gt)

                s_t_warped_bb = warp(tensorInput=inputs['s_t_1'], tensorFlow=flow_bb)
                s_t_warped_bs = warp(tensorInput=inputs['s_t_1'], tensorFlow=flow_bs)
                s_t_warped_sb = warp(tensorInput=inputs['s_t_1'], tensorFlow=flow_sb)
                s_t_warped_ss = warp(tensorInput=inputs['s_t_1'], tensorFlow=flow_ss)

                s_t_warped_bb_mask = warp(tensorInput=torch.ones_like(inputs['s_t_1'], device = device), tensorFlow=flow_bb)
                s_t_warped_bs_mask = warp(tensorInput=torch.ones_like(inputs['s_t_1'], device = device), tensorFlow=flow_bs)
                s_t_warped_sb_mask = warp(tensorInput=torch.ones_like(inputs['s_t_1'], device = device), tensorFlow=flow_sb)
                s_t_warped_ss_mask = warp(tensorInput=torch.ones_like(inputs['s_t_1'], device = device), tensorFlow=flow_ss)

                optimizer.zero_grad()

                errs['MSE_bb'] = MSE_sum(s_t_warped_bb * s_t_warped_bb_mask, inputs['s_t']) / s_t_warped_bb_mask.sum()
                errs['MSE_bs'] = MSE_sum(s_t_warped_bs * s_t_warped_bs_mask, inputs['s_t']) / s_t_warped_bs_mask.sum()
                errs['MSE_sb'] = MSE_sum(s_t_warped_sb * s_t_warped_sb_mask, inputs['s_t']) / s_t_warped_sb_mask.sum()
                errs['MSE_ss'] = MSE_sum(s_t_warped_ss * s_t_warped_ss_mask, inputs['s_t']) / s_t_warped_ss_mask.sum()

                errs['MSE_bb_mask_shape'] = MSE_mean(s_t_warped_bb_mask, s_t_warped_ss_mask_gt)
                errs['MSE_bs_mask_shape'] = MSE_mean(s_t_warped_bs_mask, s_t_warped_ss_mask_gt)
                errs['MSE_sb_mask_shape'] = MSE_mean(s_t_warped_sb_mask, s_t_warped_ss_mask_gt)
                errs['MSE_ss_mask_shape'] = MSE_mean(s_t_warped_ss_mask, s_t_warped_ss_mask_gt)

                errs['total'] = errs['MSE_bb'] + errs['MSE_bs'] + errs['MSE_sb'] + errs['MSE_ss'] \
                              + errs['MSE_bb_mask_shape'] + errs['MSE_bs_mask_shape'] + errs['MSE_sb_mask_shape'] + errs['MSE_ss_mask_shape']

            if config.loss == 'image_ss':
                flow_ss = torch.nn.functional.interpolate(input=moduleNetwork(inputs['s_t'], inputs['s_t_1']), size=(config.height, config.width), mode='bilinear', align_corners=False)
                with torch.no_grad():
                    flow_ss_gt = torch.nn.functional.interpolate(input=moduleNetwork_gt(inputs['s_t'], inputs['s_t_1']), size=(config.height, config.width), mode='bilinear', align_corners=False)
                    s_t_warped_ss_mask_gt = warp(tensorInput=torch.ones_like(inputs['s_t_1'], device = device), tensorFlow=flow_ss_gt)

                s_t_warped_ss = warp(tensorInput=inputs['s_t_1'], tensorFlow=flow_ss)
                s_t_warped_ss_mask = warp(tensorInput=torch.ones_like(inputs['s_t_1'], device = device), tensorFlow=flow_ss)

                optimizer.zero_grad()

                errs['MSE_ss'] = MSE_sum(s_t_warped_ss * s_t_warped_ss_mask, inputs['s_t']) / s_t_warped_ss_mask.sum()
                errs['MSE_ss_mask_shape'] = MSE_mean(s_t_warped_ss_mask, s_t_warped_ss_mask_gt)
                errs['total'] = errs['MSE_ss'] + errs['MSE_ss_mask_shape']

            if config.loss == 'flow_only':
                flow_bb = torch.nn.functional.interpolate(input=moduleNetwork(inputs['b_t'], inputs['b_t_1']), size=(config.height, config.width), mode='bilinear', align_corners=False)
                flow_bs = torch.nn.functional.interpolate(input=moduleNetwork(inputs['b_t'], inputs['s_t_1']), size=(config.height, config.width), mode='bilinear', align_corners=False)
                flow_sb = torch.nn.functional.interpolate(input=moduleNetwork(inputs['s_t'], inputs['b_t_1']), size=(config.height, config.width), mode='bilinear', align_corners=False)
                flow_ss = torch.nn.functional.interpolate(input=moduleNetwork(inputs['s_t'], inputs['s_t_1']), size=(config.height, config.width), mode='bilinear', align_corners=False)

                s_t_warped_ss = warp(tensorInput=inputs['s_t_1'], tensorFlow=flow_ss)

                with torch.no_grad():
                    flow_ss_gt = torch.nn.functional.interpolate(input=moduleNetwork_gt(inputs['s_t'], inputs['s_t_1']), size=(config.height, config.width), mode='bilinear', align_corners=False)

                optimizer.zero_grad()
     
                # liteflow_flow_only
                errs['MSE_bb_ss'] = MSE_mean(flow_bb, flow_ss_gt)
                errs['MSE_bs_ss'] = MSE_mean(flow_bs, flow_ss_gt)
                errs['MSE_sb_ss'] = MSE_mean(flow_sb, flow_ss_gt)
                errs['MSE_ss_ss'] = MSE_mean(flow_ss, flow_ss_gt)
                errs['total'] = errs['MSE_bb_ss'] + errs['MSE_bs_ss'] + errs['MSE_sb_ss'] + errs['MSE_ss_ss']


            errs['total'].backward()
            optimizer.step()

            lr = adjust_learning_rate(optimizer, epoch, config.decay_rate, config.decay_every, config.lr_init)

            if itr % config.write_log_every_itr == 0:
                summary.add_scalar('loss/loss_mse', errs['total'].item(), itr)
                vutils.save_image(inputs['s_t_1'].detach().cpu(), '{}/{}_1_input.png'.format(config.LOG_DIR.sample, itr), nrow=3, padding = 0, normalize = False)
                vutils.save_image(s_t_warped_ss.detach().cpu(), '{}/{}_2_warped_ss.png'.format(config.LOG_DIR.sample, itr), nrow=3, padding = 0, normalize = False)
                vutils.save_image(inputs['s_t'].detach().cpu(), '{}/{}_3_gt.png'.format(config.LOG_DIR.sample, itr), nrow=3, padding = 0, normalize = False)

                if config.loss == 'image_ss':
                    vutils.save_image(s_t_warped_ss_mask.detach().cpu(), '{}/{}_4_s_t_wapred_ss_mask.png'.format(config.LOG_DIR.sample, itr), nrow=3, padding = 0, normalize = False)
                elif config.loss != 'flow_only':
                    vutils.save_image(s_t_warped_bb_mask.detach().cpu(), '{}/{}_4_s_t_wapred_bb_mask.png'.format(config.LOG_DIR.sample, itr), nrow=3, padding = 0, normalize = False)


            if itr % config.refresh_image_log_every_itr == 0:
                remove_file_end_with(config.LOG_DIR.sample, '*.png')

            print_logs('TRAIN', config.mode, epoch, itr_time, itr, data_loader.num_itr, errs = errs, lr = lr)
            itr += 1

        if epoch % config.write_ckpt_every_epoch == 0:
            ckpt_manager.save_ckpt(moduleNetwork, epoch, score = errs['total'].item())

##########################################################

def estimate():
    tensorFirst = torch.FloatTensor(numpy.array(PIL.Image.open(arguments_strFirst))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))
    tensorSecond = torch.FloatTensor(numpy.array(PIL.Image.open(arguments_strSecond))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))

    assert(tensorFirst.size(1) == tensorSecond.size(1))
    assert(tensorFirst.size(2) == tensorSecond.size(2))

    intWidth = tensorFirst.size(2)
    intHeight = tensorFirst.size(1)

    assert(intWidth == 1024) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
    assert(intHeight == 436) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

    tensorPreprocessedFirst = tensorFirst.cuda().view(1, 3, intHeight, intWidth)
    tensorPreprocessedSecond = tensorSecond.cuda().view(1, 3, intHeight, intWidth)

    intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 32.0) * 32.0))
    intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 32.0) * 32.0))

    tensorPreprocessedFirst = torch.nn.functional.interpolate(input=tensorPreprocessedFirst, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
    tensorPreprocessedSecond = torch.nn.functional.interpolate(input=tensorPreprocessedSecond, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

    tensorFlow = torch.nn.functional.interpolate(input=moduleNetwork(tensorPreprocessedFirst, tensorPreprocessedSecond), size=(intHeight, intWidth), mode='bilinear', align_corners=False)

    tensorFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
    tensorFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

    return tensorFlow[0, :, :, :].cpu()

##########################################################

if __name__ == '__main__':
    import argparse
    from config import get_config, log_config, print_config

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type = str, default = 'liteFlowNet', help = 'model name')
    parser.add_argument('-t', '--is_train', type = str , default = 'true', help = 'whether to train or not')
    parser.add_argument('-dl', '--delete_log', type = str , default = 'false', help = 'whether to train or not')
    parser.add_argument('-l', '--loss', type = str , default = 'image', help = 'loss mode')
    args = parser.parse_args()

    config = get_config(args.mode)
    config.is_train = args.is_train
    config.delete_log = args.delete_log
    config.loss = args.loss

    print(toGreen('Laoding Config...'))
    print_config(config)

    is_train = to_bool(args.is_train)
    handle_directory(config, to_bool(args.delete_log))

    if is_train:
        train(config)
    else:
        tensorOutput = estimate(tensorFirst, tensorSecond)
