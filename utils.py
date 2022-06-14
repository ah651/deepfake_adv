import torch
import torch.nn as nn 
import torch.nn.init as init 
import torch.nn.functional as F 

##############################################
# Center Loss for Attention Regularization
##############################################
class CenterLoss(nn.Module):
    def __init__(self):
        super(CenterLoss, self).__init__()
        self.l2_loss = nn.MSELoss(reduction='sum')

    def forward(self, outputs, targets):
        return self.l2_loss(outputs, targets) / outputs.size(0)


class AverageMeter(object):
    """
    Computes and stores the average and current value
    avg: avg of all losses/acc_rates 
    val: current values of loss/acc_rate
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else -1


def adjust_learning_rate(optimizers, args, epoch, lr_factor, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = lr_factor ** (epoch // lr_steps)
    lr = args.lr * decay
    weight_decay = args.weight_decay * decay
    for optimizer in optimizers:
        if optimizer is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr  #* param_group['lr_mult']
                param_group['weight_decay'] = weight_decay #* param_group['decay_mult']

def adjust_learning_rate_gen(optimizers, args, epoch, lr_factor, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = lr_factor ** (epoch // lr_steps)
    lr = args.gen_lr * decay
    weight_decay = args.weight_decay * decay
    for optimizer in optimizers:
        if optimizer is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr  #* param_group['lr_mult']
                param_group['weight_decay'] = weight_decay #* param_group['decay_mult']

def adjust_learning_rate_dis(optimizers, args, epoch, lr_factor, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = lr_factor ** (epoch // lr_steps)
    lr = args.lr * decay
    weight_decay = args.weight_decay * decay
    for optimizer in optimizers:
        if optimizer is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr  #* param_group['lr_mult']
                param_group['weight_decay'] = weight_decay #* param_group['decay_mult']
    


def calculate_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    #tensor.topk return topk values and positions
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    # criterion = torch.nn.CrossEntropyLoss().cuda()
    # validsample = criterion(output, target)

    real = target.eq(torch.ones_like(target))
    fake = target.eq(torch.zeros_like(target))
    num_real, num_fake = real.sum(), fake.sum()
    correct_real = (correct[:1].view(-1)*real).float().sum()
    correct_fake = (correct[:1].view(-1)*fake).float().sum()

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    # top k correct rate, num of real correct, num of fake correct, num of all real, num of all fake
    return res, correct_real, correct_fake, num_real, num_fake


def calculate_binary_accuracy(output, target):
    batch_size = target.size(0)

    # output = torch.sigmoid(output)
    pred = output.ge(0.5).byte().view(-1)
    correct = pred.eq(target.byte())

    real = target.eq(torch.ones_like(target))
    fake = target.eq(torch.zeros_like(target))
    num_real, num_fake = real.sum(), fake.sum()
    correct_real = (correct*real).float().sum()
    correct_fake = (correct*fake).float().sum()

    correct_sum = correct.float().sum(0)
    res = correct_sum.mul_(100.0 / batch_size)
    return [res], correct_real, correct_fake, num_real, num_fake

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
        return torch.sum(-true_dist * pred, dim=self.dim)

class SmoothCE(nn.Module):
    def __init__(self, dim=-1):
        super(SmoothCE, self).__init__()
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist[:, 0] = target
            true_dist[:, 1] = 1.0 - target
        # return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
        return torch.sum(-true_dist * pred, dim=self.dim)

class FocalLoss(nn.Module):
    def __init__(self, weight=None, reduction='none', gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction='none')
        self.reduction = reduction
    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "none":
            return loss 
        else:
            raise NotImplementedError

def init_weights(net, init_type='kaiming', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize knetwork with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>
    return net 


def warm_up_lr(i, num_iters=300):
    return i/num_iters