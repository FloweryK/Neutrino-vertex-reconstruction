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

    # correction related to attenuation length from cwm (2018 -> 2013)
    # r' = [(a-b)/R * r + b] * r                    -> [c * r + d] * r
    # z' = [(a-b)/R * r + b] * z                    -> [c * r + d] * z
    # r = [R/2(a-b)] * [-b + sqrt(b^2 + 4(a-b)/R * r')]  -> (1/2c) * (-d + sqrt(d^2 + 4cr'))
    # z = 1/(c * r + d) * z'

    # 2018 -> 2010 transform
    # weight2 = 0.8784552 - 0.0000242758 * __perp(reco_vertex)  # base point -> 2018
    c = -0.0000242758
    d = 0.8784552
    outputs_r = np.sqrt(outputs[0]**2 + outputs[1]**2)
    outputs_r0 = 1/(2*c) * (-d + np.sqrt(np.square(d) + 4*c*outputs_r))
    outputs *= 1 / (c * outputs_r0 + d)

    # 2010 -> 2013 transform
    # weight2 = 1.09723 + (1.04556 - 1.09723) / 1685 * __perp(reco_vertex)    # base point -> 2013
    c = (1.04556 - 1.09723) / 1685
    d = 1.09723
    outputs_r0 = np.sqrt(outputs[0]**2 + outputs[1]**2)
    outputs *= (c * outputs_r0 + d)

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
        weight2 = 1.09723 + (1.04556 - 1.09723) / 1685 * __perp(reco_vertex)    # base point -> 2013
        # weight2 = 0.8784552 - 0.0000242758 * __perp(reco_vertex)  # base point -> 2018
        reco_vertex[:2] *= weight1r
        reco_vertex[2] *= weight1z
        reco_vertex *= weight2

        outputs.append(reco_vertex)     # N * [x, y, z]

    outputs = np.array(outputs).T       # X, Y, Z

    return outputs