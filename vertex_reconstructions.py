from utils import save, load, path_list
from training import Net
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import scipy.interpolate as interpolate


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


# TODO: check with cpu usage
def get_nn_outputs(model_directory, inputs, epoch, gpu=True):
    def __get_labels(root_directory, target, form='tensor'):
        labels = load(root_directory + '/' + target + '_labels.tensor')
        labels = labels.float()

        if form is 'numpy':
            labels = labels.numpy()

        return labels

    def __get_inputs(root_directory, target, form='tensor'):
        inputs = load(root_directory + '/' + target + '_inputs.tensor')
        inputs = inputs.float()

        if form is 'numpy':
            inputs = inputs.numpy()

        return inputs

    net = Net()

    if gpu and torch.cuda.is_available():
        device = torch.device('cuda')

        # data parallelism
        if torch.cuda.device_count() > 1:
            print('currently using ' + str(torch.cuda.device_count()) + ' cuda devices.')
            net = nn.DataParallel(net)
    else:
        device = torch.device('cpu')
    net.to(device)

    # load state
    model = torch.load(path_list(model_directory + '/models/epoch_%05i/' % epoch, filter='pt')[-1], map_location=device)
    net.load_state_dict(model)

    # convert inputs to tensor
    inputs = torch.FloatTensor(inputs)
    inputs.to(device)

    outputs = net(inputs).detach().cpu().clone().numpy().T
    outputs *= 1000

    # add correction
    '''
    vali_inputs = __get_inputs(model_directory, target='vali')
    vali_labels = __get_labels(model_directory, target='vali', form='numpy')
    vali_outputs = net(vali_inputs).detach().cpu().clone().numpy()
    vali_residual = (vali_outputs - vali_labels) * 1000
    vali_residual = vali_residual.T
    vali_mean = np.mean(vali_residual, axis=1)
    '''

    return outputs


def get_cwm_outputs(inputs, interpol_kind='linear'):
    def __perp(vertex):
        return np.linalg.norm(vertex[:2])

    try:
        interp_r = load('src/interp_r')
        interp_z = load('src/interp_z')
    except FileNotFoundError:
        f = open('src/WeightingCorrection_att.dat', 'r')

        df = []
        while True:
            line = f.readline().split(' ')
            line = list(filter(lambda a: a != '', line))
            try:
                line[3] = line[3][:-1]
            except IndexError:
                break
            df.append(line)
        df = pd.DataFrame(df, dtype=float)

        # calculate interpolation
        R = df[0]
        Z = df[1]
        weight_R = df[2]
        weight_Z = df[3]

        interp_r = interpolate.interp2d(R, Z, weight_R, kind=interpol_kind)
        interp_z = interpolate.interp2d(R, Z, weight_Z, kind=interpol_kind)

        save(interp_r, 'src/interp_r')
        save(interp_z, 'src/interp_z')

    pmt_positions = load('src/pmtcoordinates_ID.json')
    outputs = []
    for x in inputs:
        reco_vertex = np.array([.0, .0, .0])

        for pmt_id, hits in enumerate(x):
            pmt_pos = pmt_positions[str(pmt_id)]
            reco_vertex += hits * np.array([pmt_pos['x'], pmt_pos['y'], pmt_pos['z']], dtype=float)
        reco_vertex = reco_vertex / sum(x)

        weight1r = interp_r(__perp(reco_vertex), abs(reco_vertex[2]))
        weight1z = interp_z(__perp(reco_vertex), abs(reco_vertex[2]))
        weight2 = 0.8784552 - 0.0000242758 * __perp(reco_vertex)
        reco_vertex[:2] *= weight1r
        reco_vertex[2] *= weight1z
        reco_vertex *= weight2

        outputs.append(reco_vertex)     # N * [x, y, z]

    outputs = np.array(outputs).T       # X, Y, Z

    return outputs