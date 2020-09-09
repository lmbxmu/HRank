
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import argparse

from data import imagenet
from models import *
from mask import *
import utils


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Evaluate')
parser.add_argument(
    '--data_dir',
    type=str,
    default='./data',
    help='dataset path')
parser.add_argument(
    '--dataset',
    type=str,
    default='cifar10',
    choices=('cifar10','imagenet'),
    help='dataset')
parser.add_argument(
    '--arch',
    type=str,
    default='vgg_16_bn',
    choices=('resnet_50','vgg_16_bn','resnet_56','resnet_110','densenet_40','googlenet'),
    help='The architecture to prune')
parser.add_argument(
    '--test_model_dir',
    type=str,
    default='./result/tmp/',
    help='The directory where the summaries will be stored.')
parser.add_argument(
    '--eval_batch_size',
    type=int,
    default=100,
    help='Batch size for validation.')
parser.add_argument(
    '--gpu',
    type=str,
    default='0',
    help='Select gpu to use')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
cudnn.benchmark = True
cudnn.enabled = True

# Data
print('==> Preparing data..')
if args.dataset=='cifar10':
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.eval_batch_size, shuffle=False, num_workers=2)
    print_freq = 3000 // args.eval_batch_size
elif args.dataset=='imagenet':
    data_tmp = imagenet.Data(args, is_evaluate=True)
    testloader = data_tmp.loader_test
    print_freq = 10000 // args.eval_batch_size
else:
    raise NotImplementedError

# Model
print('==> Building model..')
net = eval(args.arch)(compress_rate=[0.]*200)
net = net.cuda()
print(net)

if len(args.gpu)>1 and torch.cuda.is_available():
    device_id = []
    for i in range((len(args.gpu) + 1) // 2):
        device_id.append(i)
    net = torch.nn.DataParallel(net, device_ids=device_id)

def test():
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    net.eval()
    num_iterations = len(testloader)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)

            prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))

            if batch_idx%print_freq==0:
                print(
                    '({0}/{1}): '
                    'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}'.format(
                        batch_idx, num_iterations, top1=top1, top5=top5))

        print("Final Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}".format(top1=top1, top5=top5))


if len(args.gpu)>1:
    convcfg = net.module.covcfg
else:
    convcfg = net.covcfg

param_per_cov_dic={
    'vgg_16_bn': 4,
    'densenet_40': 3,
    'googlenet': 28,
    'resnet_50':3,
    'resnet_56':3,
    'resnet_110':3
}

cov_id=len(convcfg)
new_state_dict = OrderedDict()
if len(args.gpu) == 1:
    pruned_checkpoint = torch.load(args.test_model_dir + "/pruned_checkpoint/" + args.arch + "_cov" + str(cov_id) + '.pt', map_location='cuda:' + args.gpu)
    tmp_ckpt = pruned_checkpoint['state_dict']
    for k, v in tmp_ckpt.items():
        new_state_dict[k.replace('module.', '')] = v
else:
    pruned_checkpoint = torch.load(args.test_model_dir + "/pruned_checkpoint/" + args.arch + "_cov" + str(cov_id) + '.pt')
    tmp_ckpt = pruned_checkpoint['state_dict']
    for k, v in tmp_ckpt.items():
        new_state_dict['module.' + k.replace('module.', '')] = v
net.load_state_dict(new_state_dict)

test()

