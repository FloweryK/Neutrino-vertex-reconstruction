from utils import load, path_list
import nets

import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def draw_activation(root_dir):
    # model selection
    configuration = load(root_dir + '/configuration.json')
    # configuration = load('configuration.json')
    mode = configuration['mode']
    net_type = configuration['net_type']
    if net_type == 'Net':
        net = nets.Net()
    elif net_type == 'Net2c':
        net = nets.Net2c()
    elif net_type == 'CNN1c':
        net = nets.CNN1c()
    elif net_type == 'CNN2c':
        net = nets.CNN2c()
    else:
        print('invalide net type')
        raise ValueError

    # get the latest model for neural network
    epoch_path = path_list(root_dir + '/models/')[-1]
    model_path = path_list(epoch_path, filter='pt')[-1]
    # model_path = '00300.pt'
    net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    for module in net.modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.detach().numpy()
            plt.plot(np.sum(np.abs(weight), axis=0), linestyle='None', marker='.')
            plt.title('%s with %s' % (mode, net_type))
            plt.xlabel('input node #')
            plt.ylabel('absolute sum of node weight')
            plt.xlim([0, 708])
            plt.ylim([0, 15])
            plt.grid()
            plt.tight_layout()
            plt.show()
            plt.savefig('MC_vis_activation_' + root_dir + '.png')
            plt.close()
            break


def main():
    print('# of targets:')
    num_targets = int(input())

    print('target roots (%i):' % num_targets)
    target = [str(input()) for _ in range(num_targets)]

    for root in target:
        if root[-1] == '/':
            root = root[:-1]
        draw_activation(root)


if __name__ == '__main__':
    main()
