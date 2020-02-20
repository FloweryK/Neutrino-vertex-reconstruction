from utils import save, load, path_list
import nets
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
def get_nn_outputs(model_directory, net_type, inputs, epoch, gpu=True):
    if net_type == 'Net':
        net = nets.Net()
    elif net_type == 'Net2c':
        net = nets.Net2c()
    elif net_type == 'CNN1c':
        net = nets.CNN1c()
    elif net_type == 'CNN2c':
        net = nets.CNN2c()
    else:
        print('invalid net_type: ' + net_type)
        raise ValueError

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

    # get outputs
    outputs = net(inputs).detach().cpu().clone().numpy().T
    outputs *= 1000

    # MC data -> raw data transfrom
    # correction related to attenuation length from cwm (2018 -> 2013)
    # r' = [(a-b)/R * r + b] * r                    -> [c * r + d] * r
    # z' = [(a-b)/R * r + b] * z                    -> [c * r + d] * z
    # r = [R/2(a-b)] * [-b + sqrt(b^2 + 4(a-b)/R * r')]  -> (1/2c) * (-d + sqrt(d^2 + 4cr'))
    # z = 1/(c * r + d) * z'
    # weight2 = 0.8784552 - 0.0000242758 * __perp(reco_vertex)  # base point -> 2018
    # c = -0.0000242758
    c = (0.8375504 - 0.8784552) / 1685
    d = 0.8784552
    outputs_r = np.sqrt(outputs[0]**2 + outputs[1]**2)
    outputs_r0 = 1/(2*c) * (-d + np.sqrt(np.square(d) + 4*c*outputs_r))
    outputs *= 1 / (c * outputs_r0 + d)

    # raw data -> source data transform
    # weight2 = 1.09723 + (1.04556 - 1.09723) / 1685 * __perp(reco_vertex)    # base point -> 2013
    c = (1.04556 - 1.09723) / 1685
    d = 1.09723
    outputs_r0 = np.sqrt(outputs[0]**2 + outputs[1]**2)
    outputs *= (c * outputs_r0 + d)

    return outputs


def get_cwm_outputs(inputs, interpol_kind='linear'):
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

    # load pmt locations
    pmt_positions = load('src/pmtcoordinates_ID.json')

    # get outputs from inputs
    outputs = []
    for x in inputs:
        reco_vertex = np.array([.0, .0, .0])

        # Original Charge Weighting method
        for pmt_id, hits in enumerate(x):
            pmt_pos = pmt_positions[str(pmt_id)]
            reco_vertex += hits * np.array([pmt_pos['x'], pmt_pos['y'], pmt_pos['z']], dtype=float)
        reco_vertex = reco_vertex / sum(x)

        # Corrections on CWM
        r = np.linalg.norm(reco_vertex[:2])
        z = reco_vertex[2]
        weight1r = interp_r(r, abs(z))
        weight1z = interp_z(r, abs(z))
        weight2 = 1.09723 + (1.04556 - 1.09723) / 1685 * np.linalg.norm(r)    # raw data -> source data
        # weight2 = 0.8784552 - 0.0000242758 * __perp(reco_vertex)  # raw data -> MC data
        reco_vertex[:2] *= weight1r
        reco_vertex[2] *= weight1z
        reco_vertex *= weight2

        outputs.append(reco_vertex)

    # transform N*(x, y, z) to [X, Y, Z]
    outputs = np.array(outputs).T

    return outputs