from utils import load, save, path_list, strftime
from utils import DEAD_PMTS

import time
import json
import pprint
import random
import argparse
import datetime
import numpy as np
import pandas as pd
from itertools import repeat
from multiprocessing import Pool

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# for RuntimeError: see https://discuss.pytorch.org/t/runtimeerror-received-0-items-of-ancdata/4999
torch.multiprocessing.set_sharing_strategy('file_system')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(354, 128)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(64, 3)

        nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))


    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class JsonDataset(Dataset):
    def __init__(self, paths, input_type='prompt', output_type='prompt', dead=True):
        self.paths = paths
        self.len = len(self.paths)
        self.input_type = input_type
        self.output_type = output_type
        self.is_dead_PMT = dead

    def __getitem__(self, index):
        return self.get_x(self.paths[index]), self.get_y(self.paths[index])

    def __len__(self):
        return self.len

    def get_x(self, path):
        with open(path, encoding='UTF8') as j:
            f = json.load(j)

            capture_time = f['capture_time']    # scalar value
            hits = int(f['photon_hits'])        # scalar value
            hit_counts = f['hit_count']         # vector value
            hit_pmts = f['hit_pmt']             # vector value
            hit_time = f['hit_time']            # vector value

            # fill a single data
            x = np.zeros(354)

            for i in range(hits):
                count = hit_counts[i]
                pmt = hit_pmts[i]
                t = hit_time[i]

                if self.is_dead_PMT and (pmt in DEAD_PMTS):
                    continue

                # get prompted signal (before capture) or delayed signal (after capture)
                if self.input_type == 'prompt':
                    if t < capture_time:
                        x[pmt] += count
                elif self.input_type == 'delayed':
                    if t > capture_time:
                        x[pmt] += count
                else:
                    x[pmt] += count

            # normalizing
            if max(x) > 1:
                x = x / max(x)

            x = np.array([x.tolist()])

        return x

    def get_y(self, path):
        with open(path, encoding='UTF8') as j:
            f = json.load(j)

            # vertices for label
            if self.output_type == 'prompt':
                vertex_x0 = f['positron_x']
                vertex_y0 = f['positron_y']
                vertex_z0 = f['positron_z']
            elif self.output_type == 'delayed':
                vertex_x0 = f['cap_neu_x']
                vertex_y0 = f['cap_neu_y']
                vertex_z0 = f['cap_neu_z']
            else:
                vertex_x0 = f['vertex_x0']
                vertex_y0 = f['vertex_y0']
                vertex_z0 = f['vertex_z0']

            vertex = [vertex_x0, vertex_y0, vertex_z0]

            y = np.array(vertex) / 1000  # mm -> m transform

        return y


def load_all(dataloader):
    total = len(dataloader)
    start = time.time()

    inputs = []
    labels = []

    for i, data in enumerate(dataloader):
        inputs.append(data[0])
        labels.append(data[1])

        print('data loading [%2i%%] (%i/%i), %.2fs' % (int(100 * i / total), i, total, time.time() - start),
              end='\n' if i == (total - 1) else '\r')

    print('input, labels cat...', end='')
    inputs = torch.cat(inputs, 0)
    labels = torch.cat(labels, 0)
    print('complete.')
    print('inputs size:', inputs.size())
    print('labels size:', labels.size())

    print('data loading took %.2f secs' % (time.time() - start))

    return inputs, labels


def __job_filter_zero_counts(path, input_type):
    f = load(path)

    capture_time = f['capture_time']  # scalar value
    hits = int(f['photon_hits'])  # scalar value
    hit_counts = f['hit_count']  # vector value
    hit_time = f['hit_time']  # vector value

    valid_counts = 0

    for i in range(hits):
        count = hit_counts[i]
        t = hit_time[i]

        # get prompted signal (before capture) or delayed signal (after capture)
        if input_type == 'prompt':
            if t < capture_time:
                valid_counts += count
        elif input_type == 'delayed':
            if t >= capture_time:
                valid_counts += count
        else:
            valid_counts += count

    if (valid_counts > 0) and (valid_counts < sum(hit_counts)):
        return True
    else:
        return False


def filter_zero_counts(paths, input_type):
    p = Pool(processes=40)

    is_not_empty = []

    start = time.time()
    for i in range(100):
        print('data filtering %2i%%, %.2fs' % (i, time.time() - start),
              end='\n' if i == 99 else '\r')
        paths_batch = paths[int(0.01*i*len(paths)):int(0.01*(i+1)*len(paths))]

        is_not_empty += p.starmap(__job_filter_zero_counts, zip(paths_batch, repeat(input_type)))

    filtered_paths = [paths[i] for i in range(len(is_not_empty)) if is_not_empty[i]]

    print('after filtering: %i->%i (%.1f%%)'
          % (len(paths), len(filtered_paths), 100 * len(filtered_paths) / len(paths)))

    return filtered_paths


def main():
    # argument configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', type=str, default='MC', help='MC root directory')
    parser.add_argument('-i', type=str, default='prompt', help='input type from prompt, delayed or all')
    parser.add_argument('-o', type=str, default='prompt', help='output type from prompt, delayed or IBD vertex')
    parser.add_argument('-l', type=float, default=1e-4, help='learning rate')
    parser.add_argument('-b', type=int, default=32, help='batch size, multiplied by cuda device number')
    parser.add_argument('-w', type=int, default=128, help='num_worker of dataloader')
    parser.add_argument('-t', type=str, default='', help='additional text to test save directory')
    parser.add_argument('-e', type=int, default=40, help='number of epochs')
    parser.add_argument('-d', type=int, default=0, help='number of dataset, if 0, use all')
    parser.add_argument('-n', type=str, default='Net', help='network type.')
    parser.add_argument('-p', type=int, default=1, help='is dead PMT on or not.')

    args = parser.parse_args()
    root_directory = args.r
    input_type = args.i
    output_type = args.o
    lr = args.l
    batch_size = args.b * (torch.cuda.device_count() if torch.cuda.is_available() else 1)
    num_worker = args.w
    num_epochs = args.e
    num_dataset = args.d
    net_type = args.n
    dead_pmt = args.p

    # save directory
    save_directory = strftime('%Y%m%d-%H%M') + '_' + input_type + '-' + output_type
    if args.t:
        save_directory += '_' + args.t

    # network, criterion, optimizer
    net = Net()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # configuration summary
    config = {
        'root_directory': root_directory,
        'save directory': save_directory,
        'input_type': input_type,
        'output_type': output_type,
        'lr': lr,
        'batch_size': batch_size,
        'num_worker': num_worker,
        'num_epochs': num_epochs,
        'number of data using:': num_dataset if num_dataset else 'full load',
        'model': {
            l[0]: str(l[1]) for l in net.named_children()
        },
        'net_type': net_type,
        'using_dead_pmt': dead_pmt,
    }

    # show and save config summary
    pprint.pprint(config)
    save(config, save_directory + '/configuration.json')

    # load dataset paths, only valid ones.
    paths = path_list(root_directory, filter='.json')
    paths = filter_zero_counts(paths, input_type)

    # shuffle the dataset
    random.shuffle(paths)

    # if num_dataset exists, cut the dataset.
    if num_dataset:
        paths = paths[:num_dataset]

    # define paths
    trainpaths = paths[:int(len(paths) * 0.8)]
    valipaths = paths[int(len(paths) * 0.8):int(len(paths) * 0.9)]
    testpaths = paths[int(len(paths) * 0.9):]

    # define datasets
    trainset = JsonDataset(paths=trainpaths, input_type=input_type, output_type=output_type, dead=dead_pmt)
    valiset = JsonDataset(paths=valipaths, input_type=input_type, output_type=output_type, dead=dead_pmt)
    testset = JsonDataset(paths=testpaths, input_type=input_type, output_type=output_type, dead=dead_pmt)

    # define dataloaders
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_worker)
    valiloader = DataLoader(valiset, batch_size=batch_size, shuffle=True, num_workers=num_worker)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_worker)

    # load all from sample, vali, test
    vali_inputs, vali_labels = load_all(valiloader)
    test_inputs, test_labels = load_all(testloader)

    # save all paths
    save(trainpaths, save_directory + '/trainpaths.list')
    save(valipaths, save_directory + '/valipaths.list')
    save(testpaths, save_directory + '/testpaths.list')

    # save all inputs, labels of sample, vali, test
    save(vali_inputs, save_directory + '/vali_inputs.tensor')
    save(vali_labels, save_directory + '/vali_labels.tensor')
    save(test_inputs, save_directory + '/test_inputs.tensor')
    save(test_labels, save_directory + '/test_labels.tensor')

    # Use data parallelism for GPU usage.
    if torch.cuda.device_count() > 1:
        print('currently using ' + str(torch.cuda.device_count()) + ' cuda devices.')
        net = nn.DataParallel(net)

    # Runtime error handling for float type to use Data parallelism.
    net = net.float()
    vali_inputs = vali_inputs.float()
    vali_labels = vali_labels.float()

    # move data to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    vali_inputs = vali_inputs.to(device)
    vali_labels = vali_labels.to(device)

    # optional: check start time
    start_time = datetime.datetime.now()

    loss_history = {}

    for epoch in range(num_epochs):
        for i, data in enumerate(trainloader):
            train_inputs, train_labels = data

            # Runtime error handling for float type
            train_inputs = train_inputs.float()
            train_labels = train_labels.float()

            # move data to GPU
            train_inputs = train_inputs.to(device)
            train_labels = train_labels.to(device)

            optimizer.zero_grad()

            train_outputs = net(train_inputs)
            loss = criterion(train_outputs, train_labels)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                # test
                vali_outputs = net(vali_inputs)
                vali_outputs = vali_outputs.detach().cpu().clone().numpy()
                try:
                    vali_labels = vali_labels.detach().cpu().clone().numpy()
                except AttributeError:
                    pass

                vali_dis = (vali_outputs - vali_labels) * 1000
                vali_sigma = np.std(vali_dis, axis=0)
                vali_mu = np.mean(vali_dis, axis=0)
                vali_loss = np.mean(vali_dis**2)

                dframe = {
                    'axis': ['x', 'y', 'z'],
                    'vali_sigma': vali_sigma,
                    'vali_mu': vali_mu
                }

                dframe = pd.DataFrame(dframe).T

                print('===========================================')
                print(datetime.datetime.now(), '(started at:', start_time, ')')
                print('epoch: %02i [%04i/%04i], train loss(mm2)=%.1f, vali loss(mm2)=%.1f'
                      % (epoch, i, len(trainset) / batch_size, loss.item() * 1000000, vali_loss))
                print(dframe)

                if epoch not in loss_history:
                    loss_history[epoch] = {}
                loss_history[epoch][i] = loss.item()

                save(loss_history, save_directory + '/loss_history.json')
                torch.save(net.state_dict(), save_directory + '/models/epoch_%05i/%05i.pt' % (epoch, i))


if __name__ == '__main__':
    main()

