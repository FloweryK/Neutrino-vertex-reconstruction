import os
import json
import random
import pickle
import datetime

DEAD_PMTS = [12, 31, 103, 111, 148, 157, 192, 219, 228, 240, 251, 270, 278, 342]


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save(data, path):
    # make proper directories if needed.
    path_split = list(filter(None, path.split('/')))
    if len(path_split) > 1:
        dir = '/'.join(path_split[:-1])
        path = '/'.join(path_split)

        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)

    # if saving json, use json dump.
    if path[-4:] == 'json':
        with open(path, 'w', encoding='UTF8') as f:
            json.dump(data, f, ensure_ascii=False)
    # if not, use pickle dump.
    else:
        with open(path, 'wb') as f:
            pickle.dump(data, f)


def load(path):
    # if loading json, use json load
    if path[-4:] == 'json':
        with open(path, encoding='UTF8') as f:
            result = json.load(f)
    # if not, use pickle load.
    else:
        with open(path, 'rb') as f:
            result = pickle.load(f)

    return result


def file_list(path, filter=False):
    if not os.path.exists(path):
        print(path + 'not found')
        raise FileNotFoundError

    files = sorted(os.listdir(path))
    if filter:
        files = [f for f in files if filter in f]

    return files


def path_list(dir, shuffle=False, filter=False, reverse=False):
    if not os.path.exists(dir):
        print(dir + 'not found.')
        raise FileNotFoundError

    paths = [dir + '/' + p for p in sorted(os.listdir(dir), reverse=reverse)]
    if shuffle:
        random.shuffle(paths)
    if filter:
        paths = [p for p in paths if filter in p]

    return paths
