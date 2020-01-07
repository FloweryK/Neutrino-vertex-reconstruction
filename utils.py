import os
import json
import time
import pickle
import datetime

import torch
import torch.nn as nn


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save(data, path):
    if path[-4:] == 'json':
        with open(path, 'w', encoding='UTF8') as f:
            json.dump(data, f, ensure_ascii=False)
    else:
        with open(path, 'wb') as f:
            pickle.dump(data, f)


def load(path):
    if path[-4:] == 'json':
        with open(path, encoding='UTF8') as f:
            result = json.load(f)
    else:
        with open(path, 'rb') as f:
            result = pickle.load(f)

    return result


def file_list(path):
    try:
        return sorted(os.listdir(path))
    except FileNotFoundError:
        print(path + 'not found')
        return False


def path_list(dir, filter=False, reverse=False):
    try:
        if filter:
            return [dir + '/' + p for p in sorted(os.listdir(dir), reverse=reverse) if filter in p]
        else:
            return [dir + '/' + p for p in sorted(os.listdir(dir), reverse=reverse)]
    except FileNotFoundError:
        print(dir + 'not found')
        return False


def strftime(form='%Y%m%d-%H:%M:%S'):
    return datetime.datetime.now().strftime(form)


def get_labels(root_directory, target, form='tensor'):
    labels = load(root_directory + '/' + target + '_labels.tensor')
    labels = labels.float()

    if form is 'numpy':
        labels = labels.numpy()

    return labels


def get_inputs(root_directory, target, form='tensor'):
    inputs = load(root_directory + '/' + target + '_inputs.tensor')
    inputs = inputs.float()

    if form is 'numpy':
        inputs = inputs.numpy()

    return inputs


def get_epoch_outputs(root_directory, inputs, net, form='tensor'):
    # for GPU usage
    if torch.cuda.device_count() > 1:
        print('currently using ' + str(torch.cuda.device_count()) + ' cuda devices.')
        net = nn.DataParallel(net)

    # move network and data to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    inputs = inputs.to(device)

    epoch_outputs = []

    epoch_paths = path_list(root_directory + '/models', filter='epoch')
    total = len(epoch_paths)
    start_time = time.time()

    for i, epoch_path in enumerate(epoch_paths):
        print('calculating outputs... [%2i%%] (%i/%i), %.2fs' % (int(100 * i / total), i, total, time.time() - start_time),
              end='\n' if i == (total - 1) else '\r')

        # pick the last one of each epoch
        last_idx_path = path_list(epoch_path, filter='pt')[-1]

        # load model
        net.load_state_dict(torch.load(last_idx_path, map_location=device))

        # get outputs
        outputs = net(inputs).detach().cpu().clone()

        if form is 'numpy':
            outputs = outputs.numpy()

        epoch_outputs.append(outputs)

    return epoch_outputs
