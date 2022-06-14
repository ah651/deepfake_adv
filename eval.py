import torch 
import torch.nn as nn 

import os 
import os.path 
import sys
import math 
import argparse

from dfdc_dataloader import *
from efficientnet import *
from transforms import *
from utils import AverageMeter, calculate_accuracy

from tqdm import tqdm 

use_cuda = False 
def main(args):
    global use_cuda 
    if torch.cuda.is_available():
        use_cuda = True 
        torch.backends.cudnn.benchmark = True

    if args.multi:
        args.cls_num = 5
    else:
        args.cls_num = 2 

    if args.gen:
        test_log_root = 'test_log/gen/'
    else:
        test_log_root = 'test_log/grad/'
    if not os.path.exists(test_log_root):
        os.mkdir(test_log_root)
    test_log = open(os.path.join(test_log_root, '{}.log').format(args.exp_name), 'a', encoding='utf-8')
    test_log.write('{} : {}\n'.format(args.exp_name, args.epoch))
    # prepare model 
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    
    if args.gen:
        model_path = '../ckts/ckts_exp_gen_{}/dfdc_{}.pt'.format(args.exp_name, args.epoch)
    else:
        model_path = '../ckts/ckts_exp_grad_{}/dfdc_{}.pt'.format(args.exp_name, args.epoch)
    model_dict = torch.load(model_path, map_location='cpu')['state']
    if list(model_dict.keys())[0].startswith('module'):
        tmp = {}
        for k, v in model_dict.items():
            # drop module. at start of key name
            tmp[k[7:]] = v 
        model_dict = tmp


    model_dis = efficientnet(args.cls_num, 1, args.image_size, model_type=args.model_type, dropout_rate=0).cuda()
    
    model_dis.load_state_dict(model_dict)

    if use_cuda:
        model_dis = model_dis.cuda()
        model_dis = torch.nn.DataParallel(model_dis)
    model_dis.eval()
    for params in model_dis.parameters():
        params.requires_grad = False 

    scale_size = args.image_size
    
    user_home = os.environ['HOME']
    
    if args.multi:
        annotations = "benchmark/ff_test_c23.lst"
    else:
        annotations = "benchmark/ff_test_{}_c23.lst".format(args.fake_type)
    data_roots = user_home+"/benchmark"
    transform=torchvision.transforms.Compose([
            torchvision.transforms.Scale(scale_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5])
        ]) 
    
    args.multi = args.multi and not args.multi2two
    if args.multi2two:
        args.cls_num = 2

    data_set = FF_lst_dataset(data_roots, 
                            annotations,
                            phase='TEST',
                            multi=args.multi and not args.multi2two,
                            transform=transform,
                            img_size=256, args=args)
    dataloader = torch.utils.data.DataLoader(dataset=data_set, 
                            batch_size=args.batch_size, 
                            shuffle=True,
                            num_workers=args.workers,
                            drop_last=False)
    
    print('Eval On {}'.format(annotations))
    test_log.write('Eval On {}.\n'.format(annotations))
    img_all, img_right = 0., 0.
    real_all, real_right = 0., 0. 
    if args.multi:
        fake1_all, fake1_right = 0., 0.
        fake2_all, fake2_right = 0., 0.
        fake3_all, fake3_right = 0., 0.
        fake4_all, fake4_right = 0., 0.

    else:
        fake_all, fake_right = 0., 0.
    
    dislosses = AverageMeter()
    sft = nn.Softmax(-1)
    fake_label = [] 
    fake_pred_score = [] 

    criterion = torch.nn.CrossEntropyLoss(reduce=False, reduction='none').cuda()
    for inputs, targets in tqdm(dataloader): 
        b, c, w, h = list(inputs.size()) 
        w_out, h_out = math.ceil(w/32/args.stride), math.ceil(h/32/args.stride)
        inputs, targets = inputs.cuda(), targets.cuda()
        if args.fc_model:
            targets_aug = targets
        elif args.mean:
            targets_aug = targets.unsqueeze(-1)
            targets_aug = targets_aug.repeat(1,int(w_out*h_out)).view(-1)
        else:
            targets_aug = targets 
        outputs = model_dis(inputs) 
        if args.multi2two:
            outputs = outputs.softmax(dim=-1)
            outputs_new = torch.empty((outputs.size(0), 2), device=outputs.device, dtype=outputs.dtype)
            outputs_new[:, 1] = outputs[:, 0] 
            outputs_new[:, 0] = 1 - outputs[:, 0]
            outputs = outputs_new 


        loss = criterion(outputs, targets_aug).mean() 
        dislosses.update(loss.item(), targets.size(0))
        outputs = outputs.softmax(dim=-1)

        if args.mean:
            outputs = outputs.view(b,-1,args.cls_num).mean(1).data
        else:
            outputs = outputs.view(b,-1,args.cls_num).mean(1).data

        # for roc curve and auc score
        outputs = sft(outputs)
        for output, target in zip(outputs, targets):
            fake_pred_score.append(output[0].data.cpu().item()) 
            fake_label.append(1-target.item()) 

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

        img_all += targets.shape[0]
        img_right += right
        real_all += num_real
        real_right += right_real

        if args.multi:
            fake1_all += num1
            fake1_right += right1
            fake2_all += num2
            fake2_right += right2
            fake3_all += num3
            fake3_right += right3
            fake4_all += num4
            fake4_right += right4

        else:
            fake_all += num_fake
            fake_right += right_fake


        
    print('avg-loss: {}\n'.format(dislosses.avg))
    test_log.write('avg-loss: {}\n'.format(dislosses.avg))

    # acc
    if args.multi:
        print('accuracy: {}, Real_accu: {}, Deepfakes_accu: {}, Face2Face_accu: {}, FaceSwap_accu: {}, NeuralTextures_accu: {}\n'.format(100*img_right/img_all if img_all else 0, 100*real_right/real_all if real_all else 0, 100*fake1_right/fake1_all if fake1_all else 0, 100*fake2_right/fake2_all if fake2_all else 0, 100*fake3_right/fake3_all if fake3_all else 0, 100*fake4_right/fake4_all if fake4_all else 0))
        test_log.write('accuracy: {}, Real_accu: {}, Deepfakes_accu: {}, Face2Face_accu: {}, FaceSwap_accu: {}, NeuralTextures_accu: {}\n'.format(100*img_right/img_all if img_all else 0, 100*real_right/real_all if real_all else 0, 100*fake1_right/fake1_all if fake1_all else 0, 100*fake2_right/fake2_all if fake2_all else 0, 100*fake3_right/fake3_all if fake3_all else 0, 100*fake4_right/fake4_all if fake4_all else 0))
    else:
        print('accuracy: {}, Real_accu: {}, Fake_accu: {}\n'.format(100*img_right/img_all if img_all else 0, 100*real_right/real_all if real_all else 0, 100*fake_right/fake_all if fake_all else 0))
        test_log.write('accuracy: {}, Real_accu: {}, Fake_accu: {}\n'.format(100*img_right/img_all if img_all else 0, 100*real_right/real_all if real_all else 0, 100*fake_right/fake_all if fake_all else 0))

    if not args.multi:
        from sklearn.metrics import roc_curve, roc_auc_score
        import matplotlib 
        from matplotlib import pyplot as plt 
        # roc
        fpr, tpr, thresholds = roc_curve(fake_label, fake_pred_score, pos_label=1)
        auc_value = roc_auc_score(fake_label, fake_pred_score)
        plt.plot(fpr,tpr,marker = 'o')
        plt.plot(fpr,fpr, linestyle='--')
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("exp_name:%s fake-type:%s, AUC:%s"%(args.exp_name, args.fake_type, str(round(auc_value, 5))))
        # plt.savefig(os.path.join(test_log_root, '{}_{}.png'.format(args.exp_name, args.fake_type)), dpi=100)
        # plt.savefig(os.path.join(test_log_root, '{}.png'.format(args.exp_name)), dpi=100)
        print('auc score: {}\n'.format(auc_value))
        test_log.write('auc score: {}\n'.format(auc_value))

        # ap
        from sklearn.metrics import average_precision_score
        ap_score = average_precision_score(fake_label, fake_pred_score)
        # print(ap_score) 
        # print(type(ap_score))
        print('ap score: {}\n'.format(ap_score))
        test_log.write('ap score: {}\n'.format(ap_score))

        # eer
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        print('eer score: {}\n'.format(eer))
        test_log.write('eer score: {}\n'.format(eer))

    print('\n')
    test_log.write('\n')
    test_log.close()

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--batch_size', type=int,
        help='Number of samples per minibatch for training.', default=32)  #256
    parser.add_argument('--model_type', type=str, 
        help='name of experiment', default='b5')
    parser.add_argument('--image_size', type=int,
        help='Width and hight of trainng images.', default=256)
    parser.add_argument('--mean', action='store_true',
        help='Train single-frame model with 5 frame mean')
    parser.add_argument('--stride', type=int,
        help='stride for b5 model.', default=2)
    parser.add_argument('--segment_length', type=int,
        help='Length of video segment to be used. default 5', default=1)
    parser.add_argument('--workers', type=int, metavar='N',
        help='Number of data loading workers (default: 10)', default=4, )
    parser.add_argument('--aug_before', action='store_true',
        help='Use transform before gen')
    parser.add_argument('--bn2', action='store_true',
        help='Use multi bn for real and fake images')

    parser.add_argument('--exp_name', type=str, 
        help='name of experiment', default='test')
    parser.add_argument('--epoch', type=int,
        help='Epoche of model for test.', default=55)

    # model struct 
    parser.add_argument('--fc_model', action='store_true',
        help='Use fc model to classify')
    parser.add_argument('--cmp_level', type=str, default='c23',
        help='Cmp level')
    parser.add_argument('--multi', action='store_true',
        help='Use multi types of fake')
    parser.add_argument('--fake_type', type=str, default='FaceSwap', choices=['FaceSwap', 'NeuralTextures', 'Face2Face', 'Deepfakes'],
        help='Fake type')
    parser.add_argument('--dataset', type=str, default='ff',
        help='Test dataset')
    
    return parser.parse_args(argv)

                                                                                            
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))