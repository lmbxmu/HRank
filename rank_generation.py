
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import data.imagenet as imagenet
from models import *
from utils import progress_bar
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

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
    '--job_dir',
    type=str,
    default='result/tmp',
    help='The directory where the summaries will be stored.')
parser.add_argument(
    '--arch',
    type=str,
    default='vgg_16_bn',
    choices=('resnet_50','vgg_16_bn','resnet_56','resnet_110','densenet_40','googlenet'),
    help='The architecture to prune')
parser.add_argument(
    '--resume',
    type=str,
    default=None,
    help='load the model from the specified checkpoint')
parser.add_argument(
    '--epochs',
    type=int,
    default=30,
    help='The num of epochs to train.')
parser.add_argument(
    '--lr',
    default=0.1,
    type=float,
    help='learning rate')
parser.add_argument(
    '--limit',
    type=int,
    default=5,
    help='The num of batch to get rank.')
parser.add_argument(
    '--train_batch_size',
    type=int,
    default=128,
    help='Batch size for training.')
parser.add_argument(
    '--eval_batch_size',
    type=int,
    default=100,
    help='Batch size for validation.')

parser.add_argument(
    '--start_idx',
    type=int,
    default=0,
    help='The num of epochs to train.')
parser.add_argument(
    '--gpu',
    type=str,
    default='0',
    help='Select gpu to use')
parser.add_argument(
    '--adjust_ckpt',
    action='store_true',
    help='adjust ckpt from pruned checkpoint')
parser.add_argument(
    '--compress_rate',
    type=str,
    default=None,
    help='The num of cov to start prune')


args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
print('==> Preparing data..')
if args.dataset=='cifar10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
elif args.dataset=='imagenet':
    data_tmp = imagenet.Data(args)
    trainloader = data_tmp.loader_train
    testloader = data_tmp.loader_test

if args.compress_rate:
    import re
    cprate_str=args.compress_rate
    cprate_str_list=cprate_str.split('+')
    pat_cprate = re.compile(r'\d+\.\d*')
    pat_num = re.compile(r'\*\d+')
    cprate=[]
    for x in cprate_str_list:
        num=1
        find_num=re.findall(pat_num,x)
        if find_num:
            assert len(find_num) == 1
            num=int(find_num[0].replace('*',''))
        find_cprate = re.findall(pat_cprate, x)
        assert len(find_cprate)==1
        cprate+=[float(find_cprate[0])]*num

    compress_rate=cprate

# Model
print('==> Building model..')
print(compress_rate)
net = eval(args.arch)(compress_rate=compress_rate)
net = net.to(device)

if len(args.gpu)>1 and torch.cuda.is_available():
    device_id = []
    for i in range((len(args.gpu) + 1) // 2):
        device_id.append(i)
    net = torch.nn.DataParallel(net, device_ids=device_id)


if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(args.resume, map_location='cuda:'+args.gpu)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    if args.adjust_ckpt:
        for k, v in checkpoint.items():
            new_state_dict[k.replace('module.', '')] = v
    else:
        for k, v in checkpoint['state_dict'].items():
            new_state_dict[k.replace('module.', '')] = v
    net.load_state_dict(new_state_dict)


criterion = nn.CrossEntropyLoss()
feature_result = torch.tensor(0.)
total = torch.tensor(0.)
def get_feature_hook(self, input, output):
    global feature_result
    global entropy
    global total
    a = output.shape[0]
    b = output.shape[1]
    c = torch.tensor([torch.matrix_rank(output[i,j,:,:]).item() for i in range(a) for j in range(b)])

    c = c.view(a, -1).float()
    c = c.sum(0)
    feature_result = feature_result * total + c
    total = total + a
    feature_result = feature_result / total

def get_feature_hook_densenet(self, input, output):
    global feature_result
    global total
    a = output.shape[0]
    b = output.shape[1]
    c = torch.tensor([torch.matrix_rank(output[i,j,:,:]).item() for i in range(a) for j in range(b-12,b)])

    c = c.view(a, -1).float()
    c = c.sum(0)
    feature_result = feature_result * total + c
    total = total + a
    feature_result = feature_result / total

def get_feature_hook_googlenet(self, input, output):
    global feature_result
    global total
    a = output.shape[0]
    b = output.shape[1]
    c = torch.tensor([torch.matrix_rank(output[i,j,:,:]).item() for i in range(a) for j in range(b-12,b)])

    c = c.view(a, -1).float()
    c = c.sum(0)
    feature_result = feature_result * total + c
    total = total + a
    feature_result = feature_result / total


def test():
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    limit = args.limit

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            if batch_idx >= limit:  # use the first 6 batches to estimate the rank.
               break
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, limit, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))#'''


if len(args.gpu)>1:
    convcfg = net.module.covcfg
else:
    convcfg = net.covcfg


if args.arch=='vgg_16_bn':
    for i, cov_id in enumerate(convcfg):
        cov_layer = net.features[cov_id]
        handler = cov_layer.register_forward_hook(get_feature_hook)
        test()
        handler.remove()

        if not os.path.isdir('rank_conv/'+args.arch+'_limit%d'%(args.limit)):
            os.mkdir('rank_conv/'+args.arch+'_limit%d'%(args.limit))
        np.save('rank_conv/'+args.arch+'_limit%d'%(args.limit)+'/rank_conv' + str(i + 1) + '.npy', feature_result.numpy())
        feature_result = torch.tensor(0.)
        total = torch.tensor(0.)

elif args.arch=='resnet_56':

    cov_layer = eval('net.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    test()
    handler.remove()

    if not os.path.isdir('rank_conv/' + args.arch+'_limit%d'%(args.limit)):
        os.mkdir('rank_conv/' + args.arch+'_limit%d'%(args.limit))
    np.save('rank_conv/' + args.arch+'_limit%d'%(args.limit)+ '/rank_conv%d' % (1) + '.npy', feature_result.numpy())
    feature_result = torch.tensor(0.)
    total = torch.tensor(0.)

    # ResNet56 per block
    cnt=1
    for i in range(3):
        block = eval('net.layer%d' % (i + 1))
        for j in range(9):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            test()
            handler.remove()
            np.save('rank_conv/' + args.arch+'_convwise' +'_limit%d'%(args.limit)+ '/rank_conv%d'%(cnt + 1)+'.npy', feature_result.numpy())
            cnt+=1
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            test()
            handler.remove()
            np.save('rank_conv/' + args.arch+'_convwise' +'_limit%d'%(args.limit)+ '/rank_conv%d'%(cnt + 1)+'.npy', feature_result.numpy())
            cnt += 1
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)

elif args.arch=='densenet_40':

    if not os.path.isdir('rank_conv/' + args.arch+'_limit%d'%(args.limit)):
        os.mkdir('rank_conv/' + args.arch+'_limit%d'%(args.limit))

    feature_result = torch.tensor(0.)
    total = torch.tensor(0.)

    # Densenet per block & transition
    for i in range(3):
        dense = eval('net.dense%d' % (i + 1))
        for j in range(12):
            cov_layer = dense[j].relu
            if j==0:
                handler = cov_layer.register_forward_hook(get_feature_hook)
            else:
                handler = cov_layer.register_forward_hook(get_feature_hook_densenet)
            test()
            handler.remove()

            np.save('rank_conv/' + args.arch +'_limit%d'%(args.limit) + '/rank_conv%d'%(13*i+j+1)+'.npy', feature_result.numpy())
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)

        if i<2:
            trans=eval('net.trans%d' % (i + 1))
            cov_layer = trans.relu
    
            handler = cov_layer.register_forward_hook(get_feature_hook_densenet)
            test()
            handler.remove()
    
            np.save('rank_conv/' + args.arch +'_limit%d'%(args.limit) + '/rank_conv%d' % (13 * (i+1)) + '.npy', feature_result.numpy())
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)#'''

    cov_layer = net.relu
    handler = cov_layer.register_forward_hook(get_feature_hook_densenet)
    test()
    handler.remove()
    np.save('rank_conv/' + args.arch +'_limit%d'%(args.limit) + '/rank_conv%d' % (39) + '.npy', feature_result.numpy())
    feature_result = torch.tensor(0.)
    total = torch.tensor(0.)

elif args.arch=='googlenet':

    if not os.path.isdir('rank_conv/' + args.arch+'_limit%d'%(args.limit)):
        os.mkdir('rank_conv/' + args.arch+'_limit%d'%(args.limit))
    feature_result = torch.tensor(0.)
    total = torch.tensor(0.)

    cov_list=['pre_layers',
              'inception_a3',
              'maxpool1',
              'inception_a4',
              'inception_b4',
              'inception_c4',
              'inception_d4',
              'maxpool2',
              'inception_a5',
              'inception_b5',
              ]

    # branch type
    tp_list=['n1x1','n3x3','n5x5','pool_planes']
    for idx, cov in enumerate(cov_list):

        if idx<args.start_idx:
            continue
        cov_layer=eval('net.'+cov)

        handler = cov_layer.register_forward_hook(get_feature_hook)
        test()
        handler.remove()

        if idx>0:
            for idx1,tp in enumerate(tp_list):
                if idx1==3:
                    np.save('rank_conv/' + args.arch+'_limit%d'%(args.limit) + '/rank_conv%d_'%(idx+1)+tp+'.npy',
                            feature_result[sum(net.filters[idx-1][:-1]) : sum(net.filters[idx-1][:])].numpy())
                #elif idx1==0:
                #    np.save('rank_conv1/' + args.arch + '/rank_conv%d_'%(idx+1)+tp+'.npy',
                #            feature_result[0 : sum(net.filters[idx-1][:1])].numpy())
                else:
                    np.save('rank_conv/' + args.arch+'_limit%d'%(args.limit) + '/rank_conv%d_' % (idx + 1) + tp + '.npy',
                            feature_result[sum(net.filters[idx-1][:idx1]) : sum(net.filters[idx-1][:idx1+1])].numpy())
        else:
            np.save('rank_conv/' + args.arch+'_limit%d'%(args.limit) + '/rank_conv%d' % (idx + 1) + '.npy',feature_result.numpy())
        feature_result = torch.tensor(0.)
        total = torch.tensor(0.)

elif args.arch=='resnet_110':

    cov_layer = eval('net.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    test()
    handler.remove()

    if not os.path.isdir('rank_conv/' + args.arch+'_limit%d'%(args.limit)):
        os.mkdir('rank_conv/' + args.arch+'_limit%d'%(args.limit))
    np.save('rank_conv/' + args.arch+'_limit%d'%(args.limit) + '/rank_conv%d' % (1) + '.npy', feature_result.numpy())
    feature_result = torch.tensor(0.)
    total = torch.tensor(0.)

    cnt = 1
    # ResNet110 per block
    for i in range(3):
        block = eval('net.layer%d' % (i + 1))
        for j in range(18):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            test()
            handler.remove()
            np.save('rank_conv/' + args.arch  + '_limit%d' % (args.limit) + '/rank_conv%d' % (
            cnt + 1) + '.npy', feature_result.numpy())
            cnt += 1
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            test()
            handler.remove()
            np.save('rank_conv/' + args.arch  + '_limit%d' % (args.limit) + '/rank_conv%d' % (
                cnt + 1) + '.npy', feature_result.numpy())
            cnt += 1
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)

elif args.arch=='resnet_50':

    cov_layer = eval('net.maxpool')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    test()
    handler.remove()

    if not os.path.isdir('rank_conv/' + args.arch+'_limit%d'%(args.limit)):
        os.mkdir('rank_conv/' + args.arch+'_limit%d'%(args.limit))
    np.save('rank_conv/' + args.arch+'_limit%d'%(args.limit) + '/rank_conv%d' % (1) + '.npy', feature_result.numpy())
    feature_result = torch.tensor(0.)
    total = torch.tensor(0.)

    # ResNet50 per bottleneck
    cnt=1
    for i in range(4):
        block = eval('net.layer%d' % (i + 1))
        for j in range(net.num_blocks[i]):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            test()
            handler.remove()
            np.save('rank_conv/' + args.arch+'_limit%d'%(args.limit) + '/rank_conv%d'%(cnt+1)+'.npy', feature_result.numpy())
            cnt+=1
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            test()
            handler.remove()
            np.save('rank_conv/' + args.arch + '_limit%d' % (args.limit) + '/rank_conv%d' % (cnt + 1) + '.npy',
                    feature_result.numpy())
            cnt += 1
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)

            cov_layer = block[j].relu3
            handler = cov_layer.register_forward_hook(get_feature_hook)
            test()
            handler.remove()
            if j==0:
                np.save('rank_conv/' + args.arch + '_limit%d' % (args.limit) + '/rank_conv%d' % (cnt + 1) + '.npy',
                        feature_result.numpy())#shortcut conv
                cnt += 1
            np.save('rank_conv/' + args.arch + '_limit%d' % (args.limit) + '/rank_conv%d' % (cnt + 1) + '.npy',
                    feature_result.numpy())#conv3
            cnt += 1
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)