'''
adv train by input grad

'''
import os
import sys
import math 
import time
import argparse
import torch
import torchvision
import torch.nn as nn 
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import tensorboardX 

import radam

from dfdc_dataloader import *
from efficientnet import efficientnet
import kernels
import utils 
from utils import *
from transforms import *
from DiffAugment_pytorch import DiffAugment2 

try:
    import apex
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
    from apex.parallel import DistributedDataParallel as DDP
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

writer = None
global_step = 0
val_acc = 0
EPS = 0.03 
new_norm = 100

def main(args):
    global writer, global_step, val_acc 

    torch.backends.cudnn.benchmark = True
    
    user_home = os.environ['HOME']

    if args.multi:
        args.cls_num = 5
    else:
        args.cls_num = 2 
    # process path
    model_path = 'ckts_exp_grad_{}'.format(args.exp_name)
    os.popen('hdfs dfs -mkdir -p hdfs://deepfake_train/{}'.format(model_path))
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    log_path = 'summary_exp_grad/{}'.format(args.exp_name)
    os.popen('hdfs dfs -mkdir -p hdfs://deepfake_train_summary/{}'.format(log_path))
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)
    writer = tensorboardX.SummaryWriter(log_path)

    
    model = efficientnet(args.cls_num, args.segment_length, args.image_size, model_type=args.model_type, dropout_rate=args.dropout_rate, pretrain=args.pretrain, pretrain_name=args.pretrain_name).cuda()
    model = apex.parallel.convert_syncbn_model(model)
    model = model.cuda()  
    crop_size = model.input_size

    criterion = torch.nn.CrossEntropyLoss(reduce=False, reduction='none').cuda()
    
    data_roots = "ff++"
    if args.dataset_type == 'video':
        annotations = "benchmark/ff_train_all.lst"
    else:
        annotations = "benchmark/ff_train_c23.lst"
            
        
    transform=torchvision.transforms.Compose([
            torchvision.transforms.Scale(crop_size),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5])
        ])  
    if args.dataset_type == 'video':
        train_data_real = FF_video_dataset(data_roots, 
                                annotations,
                                phase='TRAIN',
                                multi=args.multi,
                                transform=transform,
                                label='REAL',
                                img_size=256, args=args)
        train_data_fake = FF_video_dataset(data_roots, 
                                annotations,
                                phase='TRAIN',
                                multi=args.multi,
                                transform=transform,
                                label='FAKE',
                                img_size=256, args=args)
    else:
        train_data_real = FF_lst_dataset(data_roots, 
                                annotations,
                                phase='TRAIN',
                                multi=args.multi,
                                transform=transform,
                                label='REAL',
                                img_size=256, args=args)
        train_data_fake = FF_lst_dataset(data_roots, 
                                annotations,
                                phase='TRAIN',
                                multi=args.multi,
                                transform=transform,
                                label='FAKE',
                                img_size=256, args=args)


    dataloader_real = torch.utils.data.DataLoader(dataset=train_data_real, 
                                            batch_size=args.batch_size//2, 
                                            shuffle=True,
                                            num_workers=args.workers,
                                            drop_last=True) 
    dataloader_fake = torch.utils.data.DataLoader(dataset=train_data_fake, 
                                            batch_size=args.batch_size//2, 
                                            shuffle=True,
                                            num_workers=args.workers,
                                            drop_last=True) 
    val_annotations = "benchmark/ff_val_c23.lst"

    val_data_roots = user_home+"/benchmark"
    val_transform=torchvision.transforms.Compose([
            torchvision.transforms.Scale(crop_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5])
        ]) 
    
    val_data = FF_lst_dataset(val_data_roots, 
                            val_annotations,
                            phase='TEST',
                            multi=args.multi,
                            transform=val_transform,
                            img_size=256, args=args)
    val_dataloader = torch.utils.data.DataLoader(dataset=val_data, 
                            batch_size=args.batch_size, 
                            shuffle=True,
                            num_workers=args.workers,
                            drop_last=True)
    
    
    # Mixed precision training and RAdam
    optimizer = radam.RAdam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)

    [model], [optimizer] = amp.initialize(
        [model], [optimizer], num_losses = 1,
                                    opt_level="O0",
                                    keep_batchnorm_fp32=None,
                                    loss_scale="1.0"
    )


    model = nn.DataParallel(model)
    start_epoch = 0
    args.crop_size = crop_size
    print(args)
    cmd_path = os.path.join(model_path, 'cmd.txt')
    with open(cmd_path, 'w') as f:
        print(args, file=f) 
    os.popen('hdfs dfs -put -f {} hdfs://deepfake_train/{}'.format(cmd_path, cmd_path))

    for epoch in range(start_epoch, args.max_epoch):
        utils.adjust_learning_rate_dis([optimizer], args, epoch, args.lr_decay_factor, args.lr_decay_epochs)
        train([dataloader_real, dataloader_fake], criterion, [model], [optimizer], epoch, args.log_interval, args.segment_length, val_dataloader=val_dataloader, args=args)
        if (epoch + 1) % args.snapshot_interval == 0:
            print('===> Saving models of epoch {}...'.format(epoch+1))
            
            state = {'state': model.state_dict(),'epoch': epoch+1}
            torch.save(state, os.path.join(model_path, 'dfdc_{}.pt'.format(epoch+1)))
            a = os.path.join(model_path, 'dfdc_{}.pt'.format(epoch+1))
            os.popen('hdfs dfs -put -f {} hdfs://deepfake_train/{}'.format(a, a))

        os.popen('hdfs dfs -put -f {} hdfs://deepfake_train_summary/{}'.format(log_path, log_path.split('/')[0]))

    print('===> Saving last models...')
    state = {'state': model.state_dict(),'epoch': epoch}
    torch.save(state, os.path.join(model_path, 'dfdc_{}.pt'.format('last')))
    a = os.path.join(model_path, 'dfdc_{}.pt'.format('last'))
    os.popen('hdfs dfs -put -f {} hdfs://deepfake_train/{}'.format(a, a))
    time.sleep(10)

def train(train_loaders, criterion, models, optimizers, epoch, log_interval, segment_length, val_dataloader=None, args=None):
    global writer, global_step, val_acc, EPS
    dislosses = AverageMeter()
    dislosses_input_real = AverageMeter()
    dislosses_input_fake = AverageMeter()
    dislosses_gen_real = AverageMeter()
    dislosses_gen_fake = AverageMeter()
    gen_real_losses = AverageMeter()
    gen_fake_losses = AverageMeter()

    if args.gen_train_type == 'pert':
        gen_real_att_losses = AverageMeter()
        gen_fake_att_losses = AverageMeter()

    accuracies = AverageMeter()
    if args.multi:
        accuracies1 = AverageMeter()
        accuracies2 = AverageMeter()
        accuracies3 = AverageMeter()
        accuracies4 = AverageMeter()
    sft = nn.Softmax(-1)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    model_dis,  = models
    optimizer_dis,  = optimizers
    train_loader_real, train_loaders_fake = train_loaders
    model_dis.train()

    real_all, real_right = 0, 0
    fake_all, fake_right = 0, 0
    img_all, img_right = 0, 0


    gaussian = kernels.PixelGaussian(radius=args.radius, image_size=args.crop_size).cuda()
 
    for i, ((inputs_real, targets_real), (inputs_fake, targets_fake)) in enumerate(zip(train_loader_real, train_loaders_fake)):
        global_step += 1 
        b, c, w, h = list(inputs_real.size())
        w_out, h_out = math.ceil(w/32/args.stride), math.ceil(h/32/args.stride)
        targets_real = targets_real.cuda()
        targets_fake = targets_fake.cuda()
        if args.fc_model:
            targets_aug_real = targets_real.clone() 
            targets_aug_fake = targets_fake.clone() 
        else:
            targets_aug_real = targets_real.unsqueeze(-1)
            targets_aug_real = targets_aug_real.repeat(1,int(w_out*h_out)).view(-1)
            targets_aug_fake = targets_fake.unsqueeze(-1)
            targets_aug_fake = targets_aug_fake.repeat(1,int(w_out*h_out)).view(-1)
        inputs_real = inputs_real.view(b, c, w, h).cuda()
        inputs_fake = inputs_fake.view(b, c, w, h).cuda()

        # train gen model 
        model_dis.eval()
        for param in model_dis.parameters():
            param.requires_grad = False 
        if args.gen_train_type == 'pert':
            EPS = 1.0 / 255
            # adv train with perturbations
            inputs_real_tmp = torch.tensor(inputs_real, device=inputs_real.device) 
            inputs_real_tmp.requires_grad = True 
            inputs_fake_tmp = torch.tensor(inputs_fake, device=inputs_fake.device) 
            inputs_fake_tmp.requires_grad = True 

            for g_num in range(args.g_per_d): 
                if args.bn2:
                    pred_real = model_dis(inputs_real_tmp, bn_idx=1)
                else:   
                    pred_real = model_dis(inputs_real_tmp)
                # genloss_att_real = -criterion(pred_real, targets_aug_real).mean()
                genloss_att_real = criterion(pred_real, targets_aug_real).mean()
                genloss_real = genloss_att_real
                inputs_real_grad = torch.autograd.grad(genloss_real, inputs_real_tmp)[0] 
                pert_real = EPS * torch.sign(inputs_real_grad)
                pert_real = torch.clamp((inputs_real_tmp.data + pert_real - inputs_real), min=-EPS, max=EPS) 
                inputs_real_tmp.data = inputs_real_tmp.data + pert_real
                # inputs_real_tmp.grad.data.zero_()
                
                    
                # fake model
                if args.bn2:
                    pred_fake = model_dis(inputs_fake_tmp, bn_idx=1)
                else:   
                    pred_fake = model_dis(inputs_fake_tmp)
                # genloss_att_real = -criterion(pred_real, targets_aug_real).mean()
                genloss_att_fake = criterion(pred_fake, targets_aug_fake).mean()
                genloss_fake = genloss_att_fake
                inputs_fake_grad = torch.autograd.grad(genloss_fake, inputs_fake_tmp)[0] 
                pert_fake = EPS * torch.sign(inputs_fake_grad)
                pert_fake = torch.clamp((inputs_real_tmp.data + pert_fake - inputs_real), min=-EPS, max=EPS) 
                inputs_fake_tmp.data = inputs_fake_tmp.data + pert_fake
                # inputs_fake_tmp.grad.data.zero_()

            generations_real = inputs_real_tmp.data
            generations_fake = inputs_fake_tmp.data

        else: 
            # adv train with blur
            new_norm = 100.0 
            EPS = 10.0
            # blur-based
            for g_num in range(args.g_per_d):
                # real model
                sigmas_real = torch.full_like(inputs_real, 1) 
                sigmas_real.requires_grad = True 
                norm_value = torch.norm(sigmas_real, p=2., dim=(1,2,3), keepdim=True)
                sigmas_real = sigmas_real / norm_value * new_norm
                generations_real = gaussian(inputs_real, sigmas_real)
                # pred_real = model_dis(generations_real)
                if args.bn2:
                    pred_real = model_dis(generations_real, bn_idx=1)
                else:   
                    pred_real = model_dis(generations_real)
                genloss_att_real = criterion(pred_real, targets_aug_real).mean()
                genloss_real = genloss_att_real
                sigmas_real_grad = torch.autograd.grad(genloss_real, sigmas_real)[0] 
                sigmas_real = sigmas_real + EPS * sigmas_real_grad 
                norm_value = torch.norm(sigmas_real, p=2., dim=(1,2,3), keepdim=True)
                sigmas_real = sigmas_real / norm_value * new_norm
                generations_real = gaussian(inputs_real, sigmas_real)
                if (i+1) % args.hist_interval ==0:
                    writer.add_histogram('sigmas_real', sigmas_real.detach(), global_step=global_step)

                # fake model
                sigmas_fake = torch.full_like(inputs_fake, 1) 
                sigmas_fake.requires_grad = True 
                norm_value = torch.norm(sigmas_fake, p=2., dim=(1,2,3), keepdim=True)
                sigmas_fake = sigmas_fake / norm_value * new_norm
                generations_fake = gaussian(inputs_fake, sigmas_fake)
                # pred_fake = model_dis(generations_fake)
                if args.bn2:
                    pred_fake = model_dis(generations_fake, bn_idx=1)
                else:   
                    pred_fake = model_dis(generations_fake)
                genloss_att_fake = criterion(pred_fake, targets_aug_fake).mean()
                genloss_fake = genloss_att_fake
                sigmas_fake_grad = torch.autograd.grad(genloss_fake, sigmas_fake)[0] 
                sigmas_fake = sigmas_fake + EPS * sigmas_fake_grad 
                norm_value = torch.norm(sigmas_fake, p=2., dim=(1,2,3), keepdim=True)
                sigmas_fake = sigmas_fake / norm_value * new_norm
                generations_fake = gaussian(inputs_fake, sigmas_fake)
                if (i+1) % args.hist_interval ==0:
                    writer.add_histogram('sigmas_fake', sigmas_fake.detach(), global_step=global_step)

        # train dis model 
        model_dis.train() 
        for param in model_dis.parameters():
            param.requires_grad = True 
        # # add diffaugmentation to inputs and generations
        if args.aug_after:
            generations_real = DiffAugment2(generations_real) 
            generations_fake = DiffAugment2(generations_fake) 
            inputs_real = DiffAugment2(inputs_real) 
            inputs_fake = DiffAugment2(inputs_fake) 
        generations = torch.cat([generations_real, generations_fake])
        generations = generations.detach() 
        inputs = torch.cat([inputs_real, inputs_fake])
        targets_aug = torch.cat([targets_aug_real, targets_aug_fake]) 
        targets = torch.cat([targets_real, targets_fake]) 
        
        # calculate 4 kinds of loss
        model_dis.eval()
        with torch.no_grad():
            # inputs real loss
            pred_inputs_real = model_dis(inputs_real)
            disloss_inputs_real = criterion(pred_inputs_real, targets_aug_real).mean()
            # inputs fake loss
            pred_inputs_fake = model_dis(inputs_fake)
            disloss_inputs_fake = criterion(pred_inputs_fake, targets_aug_fake).mean()
            # gen real loss
            pred_gen_real = model_dis(generations_real)
            disloss_gen_real = criterion(pred_gen_real, targets_aug_real).mean()
             # gen fake loss
            pred_gen_fake = model_dis(generations_fake)
            disloss_gen_fake = criterion(pred_gen_fake, targets_aug_fake).mean()

        model_dis.train()   
        optimizer_dis.zero_grad()
        both_imgs = torch.cat([inputs, generations], dim=0)
        targets_aug = torch.cat([targets_aug, targets_aug], dim=0)
        targets = torch.cat([targets, targets], dim=0)
        if args.perm:
            perms = torch.randperm(both_imgs.size(0), device=both_imgs.device) 
            both_imgs = both_imgs[perms] 
            targets_aug = targets_aug.view(both_imgs.size(0), -1)
            targets_aug = targets_aug[perms] 
            targets_aug = targets_aug.reshape(-1) 
            targets = targets[perms] 
        inputs_to_dis = both_imgs
        pred = model_dis(inputs_to_dis)
        disloss = criterion(pred, targets_aug)
        disloss = disloss.mean()

        with amp.scale_loss(disloss, optimizer_dis) as scaled_disloss:
            scaled_disloss.backward()
        optimizer_dis.step()
        
        pred = sft(pred)
        pred_final = pred.view(b*4,-1,args.cls_num).sum(1).data

        pred_final = pred_final.argmax(dim=-1) 

        right = float(torch.sum(pred_final==targets))
        if args.multi:
            right_real = float(torch.sum((pred_final == targets) * (targets == 0)))
            right1 = float(torch.sum((pred_final == targets) * (targets == 1)))
            right2 = float(torch.sum((pred_final == targets) * (targets == 2)))
            right3 = float(torch.sum((pred_final == targets) * (targets == 3)))
            right4 = float(torch.sum((pred_final == targets) * (targets == 4)))
            # right_fake = float(torch.sum((pred_final == targets) * (targets != 0)))
        
            num_real = float(torch.sum(targets == 0))
            num1 = float(torch.sum(targets == 1))
            num2 = float(torch.sum(targets == 2))
            num3 = float(torch.sum(targets == 3))
            num4 = float(torch.sum(targets == 4))
            # num_fake = torch.sum(targets != 0)
        else:
            right_real = float(torch.sum((pred_final == targets) * (targets == 1)))
            num_real = float(torch.sum(targets == 1))
            right_fake = float(torch.sum((pred_final == targets) * (targets == 0)))
            num_fake = float(torch.sum(targets == 0))

        
        img_all += targets.shape[0]
        img_right += right
        real_all += num_real
        real_right += right_real
        dislosses.update(disloss.item(), targets.size(0))
        accuracies.update(right/targets.size(0), targets.size(0))

        gen_real_losses.update(genloss_real.item(), targets.size(0)//4)
        gen_fake_losses.update(genloss_fake.item(), targets.size(0)//4)
        if args.gen_train_type == 'pert':
            gen_real_att_losses.update(genloss_att_real.item(), targets.size(0)//4)
            gen_fake_att_losses.update(genloss_att_fake.item(), targets.size(0)//4)


        dislosses_input_real.update(disloss_inputs_real.item(), targets.size(0)//4)
        dislosses_input_fake.update(disloss_inputs_fake.item(), targets.size(0)//4)
        dislosses_gen_real.update(disloss_gen_real.item(), targets.size(0)//4)
        dislosses_gen_fake.update(disloss_gen_fake.item(), targets.size(0)//4)

        if args.multi:
            if num1:
                accuracies1.update(right1/num1, num1)
            if num2:
                accuracies2.update(right2/num2, num2)
            if num3:
                accuracies3.update(right3/num3, num3)
            if num4:
                accuracies4.update(right4/num4, num4)
        else:
            fake_all += num_fake
            fake_right += right_fake


        if (i+1) % log_interval == 0:
            print("Traing Epoch: {}, iter: {}, dis-loss avg: {}, dis-loss val: {}, genloss-real: {}, genloss-fake: {}, accuracy.avg: {}, ".format(epoch, i+1, dislosses.avg, dislosses.val, gen_real_losses.avg, gen_fake_losses.avg, 100*accuracies.avg), end=' ')
            print("dislosses_input_real: {}, dislosses_input_fake: {}, dislosses_gen_real: {}, dislosses_gen_fake: {},".format(dislosses_input_real.avg, dislosses_input_fake.avg, dislosses_gen_real.avg, dislosses_gen_fake.avg), end=' ')
            if args.multi:
                print('real-accuracy.avg: {}, accuracy1.avg: {}, accuracy2.avg: {}, accuracy3.avg: {}, accuracy4.avg: {}.'.format(100*real_right/real_all if real_all else 0, 100*accuracies1.avg, 100*accuracies2.avg, 100*accuracies3.avg, 100*accuracies4.avg))
            else:
                print('real-accuracy.avg: {}, fake-accuracy.avg: {}.'.format(100*real_right/real_all if real_all else 0, 100*fake_right/fake_all if fake_all else 0))
            sys.stdout.flush()


            writer.add_scalar('dislosses.val', disloss.item(), global_step)
            writer.add_scalar('dislosses.avg', dislosses.avg, global_step)

            writer.add_scalar('genloss_real', gen_real_losses.avg, global_step)
            writer.add_scalar('genloss_fake',  gen_fake_losses.avg, global_step)
            writer.add_scalar('dislosses_input_real',  dislosses_input_real.avg, global_step)
            writer.add_scalar('dislosses_input_fake',  dislosses_input_fake.avg, global_step)
            writer.add_scalar('dislosses_gen_real',  dislosses_gen_real.avg, global_step)
            writer.add_scalar('dislosses_gen_fake',  dislosses_gen_fake.avg, global_step)

            writer.add_scalar('accuracies.val', 100*accuracies.val, global_step)
            writer.add_scalar('accuracies.avg', 100*accuracies.avg, global_step)
            writer.add_scalar('real_accuracies.avg', 100*real_right/real_all if real_all else 0, global_step)
            if args.multi:
                writer.add_scalar('accuracies1.avg', 100*accuracies1.avg, global_step)
                writer.add_scalar('accuracies2.avg', 100*accuracies2.avg, global_step)
                writer.add_scalar('accuracies3.avg', 100*accuracies3.avg, global_step)
                writer.add_scalar('accuracies4.avg', 100*accuracies4.avg, global_step)
            else:
                writer.add_scalar('fake_accuracies.avg', 100*fake_right/fake_all if fake_all else 0, global_step)
            
            writer.add_images('generations_real', (generations_real[:4]+1)/2, global_step)
            writer.add_images('generations_fake', (generations_fake[:4]+1)/2, global_step)
            writer.add_images('referrence_real', (inputs_real[:4]+1)/2, global_step)
            writer.add_images('referrence_fake', (inputs_fake[:4]+1)/2, global_step)

            gen_img_real = generations_real[:4].detach().cpu().numpy().mean(axis=1, keepdims=True)
            gen_img_real = ((gen_img_real+1)/2 * 255).astype(np.uint8)
            real_img = inputs_real[:4].detach().cpu().numpy().mean(axis=1, keepdims=True)
            real_img = ((real_img+1)/2 * 255).astype(np.uint8)
            diff_tmp = np.abs(gen_img_real - real_img)
            writer.add_images('difference_real', diff_tmp, global_step)

            gen_img_fake = generations_fake[:4].detach().cpu().numpy().mean(axis=1, keepdims=True)
            gen_img_fake = ((gen_img_fake+1)/2 * 255).astype(np.uint8)
            fake_img = inputs_fake[:4].detach().cpu().numpy().mean(axis=1, keepdims=True)
            fake_img = ((fake_img+1)/2 * 255).astype(np.uint8)
            diff_tmp = np.abs(gen_img_fake - fake_img)
            writer.add_images('difference_fake', diff_tmp, global_step)

            writer.flush()

    # start val 
    model_dis.eval() 
    for param in model_dis.parameters():
        param.requires_grad = False 
    dislosses_val = AverageMeter()
    accuracies = AverageMeter()
    val_real_all, val_real_right = 0, 0
    val_img_all, val_img_right = 0, 0
    if args.multi:
        accuracies1 = AverageMeter()
        accuracies2 = AverageMeter()
        accuracies3 = AverageMeter()
        accuracies4 = AverageMeter()
    else:
        val_fake_all, val_fake_right = 0, 0

    for inputs, targets in val_dataloader: 
        # print('-', end='')
        b, c, w, h = list(inputs.size()) 
        w_out, h_out = math.ceil(w/32/args.stride), math.ceil(h/32/args.stride)
        inputs = inputs.view(b, c, w, h)
        inputs, targets = inputs.cuda(), targets.cuda()
        if args.fc_model:
            targets_aug = targets
        elif args.mean:
            targets_aug = targets.unsqueeze(-1)
            targets_aug = targets_aug.repeat(1,int(w_out*h_out)).view(-1)
        else:
            targets_aug = targets 
        outputs = model_dis(inputs) 

        loss = criterion(outputs, targets_aug).mean() 
        dislosses_val.update(loss.item(), targets.size(0))
        outputs = outputs.softmax(dim=-1)

        outputs = outputs.view(b,-1,args.cls_num).mean(1).data

        pred_final = outputs.argmax(dim=-1) 

        right = float(torch.sum(pred_final==targets))
        if args.multi:
            right_real = float(torch.sum((pred_final == targets) * (targets == 0)))
            right1 = float(torch.sum((pred_final == targets) * (targets == 1)))
            right2 = float(torch.sum((pred_final == targets) * (targets == 2)))
            right3 = float(torch.sum((pred_final == targets) * (targets == 3)))
            right4 = float(torch.sum((pred_final == targets) * (targets == 4)))
            # right_fake = float(torch.sum((pred_final == targets) * (targets != 0)))
        
            num_real = float(torch.sum(targets == 0))
            num1 = float(torch.sum(targets == 1))
            num2 = float(torch.sum(targets == 2))
            num3 = float(torch.sum(targets == 3))
            num4 = float(torch.sum(targets == 4))
            # num_fake = torch.sum(targets != 0)
        else:
            right_real = float(torch.sum((pred_final == targets) * (targets == 1)))
            num_real = float(torch.sum(targets == 1))
            right_fake = float(torch.sum((pred_final == targets) * (targets == 0)))
            num_fake = float(torch.sum(targets == 0))

        val_img_all += targets.shape[0]
        val_img_right += right
        val_real_all += num_real
        val_real_right += right_real

        accuracies.update(right/targets.size(0), targets.size(0))
        if args.multi:
            if num1:
                accuracies1.update(right1/num1, num1)
            if num2:
                accuracies2.update(right2/num2, num2)
            if num3:
                accuracies3.update(right3/num3, num3)
            if num4:
                accuracies4.update(right4/num4, num4)
        else:
            val_fake_all += num_fake
            val_fake_right += right_fake


    print('----------------------------------------------------------------')
    if epoch == 0:
        val_imgs, _ = next(iter(val_dataloader))
        b ,c, w, h = list(val_imgs.size()) 
        val_imgs = val_imgs.view(b, c, w, h)
        writer.add_images('val-img', (val_imgs[:8]+1)/2, global_step)
    
    val_acc_tmp = 100*val_img_right/val_img_all
    if val_acc_tmp > val_acc:
        val_acc = val_acc_tmp
        print('Save best model at Epoch : {}, best acc: {}'.format(epoch+1, val_acc))
        state = {'state': model_dis.state_dict(),'epoch': epoch+1}
        model_path = 'ckts_exp_grad_{}'.format(args.exp_name)
        torch.save(state, os.path.join(model_path, 'dfdc_{}.pt'.format('best')))
        
        a = os.path.join(model_path, 'dfdc_{}.pt'.format('best'))
        os.popen('hdfs dfs -put -f {} hdfs://deepfake_train/{}'.format(a, a))
    # print val message 
    print('Val on val-set, Epoch: {}, val_loss: {}, acc: {}, real-acc: {}, '.format(epoch+1, dislosses_val.avg, 100*val_img_right/val_img_all, 100*val_real_right/val_real_all), end=' ')
    if args.multi:
        print('accuracy1.avg: {}, accuracy2.avg: {}, accuracy3.avg: {}, accuracy4.avg: {}.'.format(100*accuracies1.avg, 100*accuracies2.avg, 100*accuracies3.avg, 100*accuracies4.avg))
    else:
        print('fake-acc: {}'.format(100*val_fake_right/val_fake_all)) 
    sys.stdout.flush()

    writer.add_scalar('val:avg-loss', dislosses_val.avg, global_step)
    writer.add_scalar('val:acc', 100*val_img_right/val_img_all, global_step)
    writer.add_scalar('val:real-acc', 100*val_real_right/val_real_all, global_step)
    if args.multi:
        writer.add_scalar('val:accuracies1', 100*accuracies1.avg, global_step)
        writer.add_scalar('val:accuracies2', 100*accuracies2.avg, global_step)
        writer.add_scalar('val:accuracies3', 100*accuracies3.avg, global_step)
        writer.add_scalar('val:accuracies4', 100*accuracies4.avg, global_step)
    else:
        writer.add_scalar('val:fake-acc', 100*val_fake_right/val_fake_all, global_step)
    writer.flush()

    model_dis.train() 
    for param in model_dis.parameters():
        param.requires_grad = True 

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mean', action='store_true',
        help='Train single-frame model with 5 frame mean')
    parser.add_argument('--exp_name', type=str,
        help='name of experiment', default='debug')

    parser.add_argument('--model_type', type=str,
        help='model type for backbone', default='b5')
    parser.add_argument('--pretrain', action='store_true',
        help='Use our pretrained model')
    parser.add_argument('--pretrain_name', type=str,
        help='model type for backbone', default='efficientnet_b5_ns-6f26d0cf.pth')
    parser.add_argument('--dataset', type=str,
        help='Dataset type', default='default')
    parser.add_argument('--dropout_rate', type=float,
        help='Dropout rate for cls model', default=0)
    parser.add_argument('--dataset_type', type=str,
        help='Dataset type', default='video')
    parser.add_argument('--val_dataset_type', type=str,
        help='Dataset type', default='small')
    parser.add_argument('--stride', type=int,
        help='stride for b5 model.', default=2)
    
    parser.add_argument('--fc_model', action='store_true',
        help='Use fc model to classify')
    parser.add_argument('--multi', action='store_true',
        help='Use multi types of fake')
    parser.add_argument('--fake_type', type=str, default='NeuralTextures', choices=['FaceSwap', 'NeuralTextures', 'Face2Face', 'Deepfakes'],
        help='Fake type')
    parser.add_argument('--cmp_level', type=str, default='c23',
        help='Cmp level')

    parser.add_argument('--norm', action='store_true',
        help='Norm sigma')
    parser.add_argument('--norm_value', type=float,
        help='Norm value for stadv.', default=2.0)

    parser.add_argument('--aug_before', action='store_true',
        help='Use transform before gen')
    parser.add_argument('--aug_before_full', action='store_true',
        help='Use full transform before gen')
    parser.add_argument('--aug_after', action='store_true',
        help='Use transform before gen')
    parser.add_argument('--perm', action='store_true',
        help='Perm inputs when training dis')
    parser.add_argument('--aug_blur', action='store_true',
        help='Aug blur image for a fair compair')
    parser.add_argument('--check_blur', action='store_true',
        help='Use sigmoid other than abs as sigma')
    parser.add_argument('--bn2', action='store_true',
        help='Use multi bn for real and fake images')

    parser.add_argument('--radius', type=int,
        help='radius of gaussian kernel.', default=3)
    parser.add_argument('--gen_train_type', type=str,
        help='Use lsgan-like loss for g or wgan-like loss', default='default')
    parser.add_argument('--g_per_d', type=int, default=1,
        help='# of train g every d')

    parser.add_argument('--max_epoch', type=int,
        help='Number of training epoches.', default=160)
    parser.add_argument('--batch_size', type=int,
        help='Number of samples per minibatch for training.', default=32)  #256
    parser.add_argument('--image_size', type=int,
        help='Width and hight of trainng images.', default=256)
    parser.add_argument('--lr', type=float,
        help='Initial learning rate.', default=5e-4)
    parser.add_argument('--lr_decay_factor', type=float,
        help='The factor to decay learning rate.', default=1e-1)    
    parser.add_argument('--lr_decay_epochs', type=float,
        help='Every 200 epochs to decay the learning rate.', default=2) 
    parser.add_argument('--weight_decay', type=float,
        help='Parameter for weigth decay.', default=5e-4)
    parser.add_argument('--momentum', type=float, 
        help='Solver momentum', default=0.9)
    parser.add_argument('--snapshot_interval', type=int, 
        help='Snapshot per X epochs during training.', default=1)
    parser.add_argument('--log_interval', type=int, 
        help='Log per X epochs during training.', default=200)
    parser.add_argument('--hist_interval', type=int, 
        help='Log per X epochs during training.', default=200)
    parser.add_argument('--segment_length', type=int,
        help='Length of video segment to be used.', default=1)
    parser.add_argument('--workers', type=int, metavar='N',
        help='Number of data loading workers (default: 4)', default=4, )

    return parser.parse_args(argv)

                                                                                            
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
