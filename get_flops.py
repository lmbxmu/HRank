from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import torch
from torch.autograd import Variable
from functools import reduce
import operator

count_ops = 0
count_params = 0


def get_num_gen(gen):
    return sum(1 for x in gen)

def is_pruned(layer):
    try:
        layer.mask
        return True
    except AttributeError:
        return False

def is_leaf(model):
    return get_num_gen(model.children()) == 0

def get_layer_info(layer):
    layer_str = str(layer)
    # print(layer_str)
    type_name = layer_str[:layer_str.find('(')].strip()
    return type_name

def get_layer_param(model, is_conv=True):
    if is_conv:
        total=0.
        for idx, param in enumerate(model.parameters()):
            assert idx<2
            f = param.size()[0]
            pruned_num = int(model.cp_rate * f)
            if len(param.size())>1:
                c=param.size()[1]
                if hasattr(model,'last_prune_num'):
                    last_prune_num=model.last_prune_num
                    total += (f - pruned_num) * (c-last_prune_num) * param.numel() / f / c
                else:
                    total += (f - pruned_num) * param.numel() / f
            else:
                total += (f - pruned_num) * param.numel() / f
        return total
    else:
        return sum([reduce(operator.mul, i.size(), 1) for i in model.parameters()])

### The input batch size should be 1 to call this function
def measure_layer(layer, x, print_name):
    global count_ops, count_params
    delta_ops = 0
    delta_params = 0
    multi_add = 1
    type_name = get_layer_info(layer)

    ### ops_conv
    if type_name in ['Conv2d']:
        out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) /
                    layer.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) /
                    layer.stride[1] + 1)
        pruned_num = int(layer.cp_rate * layer.out_channels)

        if hasattr(layer,'tmp_name') and 'trans' in layer.tmp_name:
            delta_ops = (layer.in_channels-layer.last_prune_num) * (layer.out_channels - pruned_num) * layer.kernel_size[0] * \
                        layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add
        else:
            delta_ops = layer.in_channels * (layer.out_channels-pruned_num) * layer.kernel_size[0] *  \
                    layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add

        delta_ops_ori = layer.in_channels * layer.out_channels * layer.kernel_size[0] * \
                    layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add

        delta_params = get_layer_param(layer)

        if print_name:
            print(layer.tmp_name, layer.cp_rate, '| input:',x.size(),'| weight:',[layer.out_channels, layer.in_channels, layer.kernel_size[0], layer.kernel_size[1]],
                  '| params:', delta_params, '| flops:', delta_ops_ori)
        else:
            print(layer.cp_rate, [layer.out_channels,layer.in_channels,layer.kernel_size[0],layer.kernel_size[1]],
                  'params:',delta_params, ' flops:',delta_ops_ori)

    ### ops_linear
    elif type_name in ['Linear']:
        weight_ops = layer.weight.numel() * multi_add
        bias_ops = layer.bias.numel()
        delta_ops = x.size()[0] * (weight_ops + bias_ops)
        delta_params = get_layer_param(layer, is_conv=False)

        print('linear:',layer, delta_ops, delta_params)

    elif type_name in ['DenseBasicBlock', 'ResBasicBlock']:
        measure_layer(layer.conv1, x)

    elif type_name in ['Inception']:
        measure_layer(layer.conv1, x)

    elif type_name in ['DenseBottleneck', 'SparseDenseBottleneck']:
        measure_layer(layer.conv1, x)

    elif type_name in ['Transition', 'SparseTransition']:
        measure_layer(layer.conv1, x)

    elif type_name in ['ReLU', 'BatchNorm1d','BatchNorm2d', 'Dropout2d', 'DropChannel', 'Dropout', 'AdaptiveAvgPool2d', 'AvgPool2d', 'MaxPool2d', 'Mask', 'channel_selection', 'LambdaLayer', 'Sequential']:
        return 
    ### unknown layer type
    else:
        raise TypeError('unknown layer type: %s' % type_name)

    count_ops += delta_ops
    count_params += delta_params
    return

def measure_model(model, device, C, H, W, print_name=False):
    global count_ops, count_params
    count_ops = 0
    count_params = 0
    data = Variable(torch.zeros(1, C, H, W)).to(device)
    model = model.to(device)
    model.eval()

    def should_measure(x):
        return is_leaf(x)

    def modify_forward(model, print_name):
        for child in model.children():
            if should_measure(child):
                def new_forward(m):
                    def lambda_forward(x):
                        measure_layer(m, x, print_name)
                        return m.old_forward(x)
                    return lambda_forward
                child.old_forward = child.forward
                child.forward = new_forward(child)
            else:
                modify_forward(child, print_name)

    def restore_forward(model):
        for child in model.children():
            # leaf node
            if is_leaf(child) and hasattr(child, 'old_forward'):
                child.forward = child.old_forward
                child.old_forward = None
            else:
                restore_forward(child)

    modify_forward(model, print_name)
    model.forward(data)
    restore_forward(model)

    return count_ops, count_params
