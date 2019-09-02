import torch

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)


Backward_tensorGrid = {}
def warp(tensorInput, tensorFlow):
    if str(tensorFlow.size()) not in Backward_tensorGrid:
        tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
        tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))

        Backward_tensorGrid[str(tensorFlow.size())] = torch.cat([ tensorHorizontal, tensorVertical ], 1).cuda()

    tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0) ], 1)
    # tensorFlow = torch.cat([ 2.0 * (tensorFlow[:, 0:1, :, :] / (tensorInput.size(3) - 1.0)) - 1.0 , 2.0 * (tensorFlow[:, 1:2, :, :] / (tensorInput.size(2) - 1.0)) - 1.0  ], 1)

    return torch.nn.functional.grid_sample(input=tensorInput, grid=(Backward_tensorGrid[str(tensorFlow.size())] + tensorFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')

def adjust_learning_rate(optimizer, epoch, decay_rate, decay_every, lr_init):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr_init * (decay_rate ** (epoch // decay_every))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr
