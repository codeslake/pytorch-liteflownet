from easydict import EasyDict as edict
import json
import os
import collections

def get_config(mode = ''):
    ## GLOBAL
    project = 'liteFlow'

    config = edict()
    config.mode = mode
    config.is_train = True
    config.height = 256 
    config.width = 256
    config.delete_log = False
    config.thread_num = 4
    config.loss = None

    ##################################### TRAIN #####################################
    config.is_pretrain = False
    config.pretrain_only = False
    config.batch_size = 10
    config.n_epoch = 10000
    # learning rate
    config.lr_init = 1e-5
    config.decay_rate = 0.8
    config.decay_every = 10
    # data
    config.is_augment = True
    config.is_reverse = True
    # adam
    config.beta1 = 0.9
    # gradient norm
    config.grad_norm_clip_val = 1.0
    # data dir
    offset = '/data1/junyonglee/deblur'
    config.frame_offset = os.path.join(offset, 'train_DVD')
    config.blur_path = 'input'
    config.sharp_path = 'GT'
    config.of_path = os.path.join(offset, 'train_DVD_of')
    config.of_reverse_path = os.path.join(offset, 'train_DVD_of_reverse')
    # data options
    config.sample_num = 2
    config.skip_length = [0, 3]
    config.skip_length_reverse = [3, 0]
    # logs
    config.max_ckpt_num = 10
    config.write_ckpt_every_epoch = 1
    config.refresh_image_log_every_itr = 10000
    config.refresh_image_log_every_epoch = 2
    config.write_log_every_itr = 20
    config.write_ckpt_every_itr = 1000
    # log dirs
    config.LOG_DIR = edict()
    offset = '/Jarvis/logs/junyonglee'
    offset = os.path.join(offset, project)
    offset = os.path.join(offset, '{}'.format(mode))
    config.LOG_DIR.ckpt = os.path.join(offset, 'checkpoint', 'train', 'epoch')
    config.LOG_DIR.ckpt_itr = os.path.join(offset, 'checkpoint', 'train', 'itr')
    config.LOG_DIR.log_scalar_train_epoch = os.path.join(offset, 'log', 'train', 'scalar', 'train', 'epoch')
    config.LOG_DIR.log_scalar_train_itr = os.path.join(offset, 'log', 'train', 'scalar', 'train', 'itr')
    config.LOG_DIR.log_scalar_valid = os.path.join(offset, 'log', 'train', 'scalar', 'valid')
    config.LOG_DIR.log_image = os.path.join(offset, 'log', 'train', 'image', 'train')
    config.LOG_DIR.sample = os.path.join(offset, 'sample', 'train')
    config.LOG_DIR.config = os.path.join(offset, 'config')

    ##################################### TEST ######################################
    # data path
    offset = '/data1/junyonglee/deblur'
    config.frame_offset_test = os.path.join('/data1/junyonglee/deblur', 'test_DVD')
    config.blur_path_test = 'input'
    config.sharp_path_test = 'GT'
    config.of_path_test = os.path.join(offset, 'test_DVD_of')
    config.of_reverse_path_test = os.path.join(offset, 'test_DVD_of_reverse')

    return config

def log_config(path, cfg):
    with open(path + '/config.txt', 'w') as f:
        f.write('================================================\n')
        f.write(json.dumps(cfg, indent=4))
        f.write('\n================================================\n')

def print_config(cfg):
    print(json.dumps(cfg, indent=4))

