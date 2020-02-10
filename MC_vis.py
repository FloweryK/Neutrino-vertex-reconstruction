from utils import load, save, path_list, DEAD_PMTS
import nets

import torch
import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from itertools import repeat
from multiprocessing import Pool


def neural_residual(root_dir):
    # model selection
    mode = load(root_dir + '/configuration.json')['mode']
    if mode == 'hit-time':
        net = nets.Cnn2c()
    elif mode == 'time':
        net = nets.Cnn1c()
    elif mode == 'hit':
        net = nets.Net()
    else:
        print('invalid net model type.')
        raise ValueError

    # get the latest model for neural network
    epoch_path = path_list(root_dir + '/models/')[-1]
    model_path = path_list(epoch_path, filter='pt')[-1]
    net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    # get inputs, labels, outputs and residuals
    inputs = load(root_dir + '/test_inputs.tensor').float()
    labels = load(root_dir + '/test_labels.tensor').float().numpy()
    outputs = net(inputs).detach().cpu().clone().numpy()
    residuals = (outputs - labels) * 1000

    return residuals.T


def cwm_residual(root_dir):
    try:
        interp_r = load('src/interp_r')
        interp_z = load('src/interp_z')
    except FileNotFoundError:
        with open('src/WeightingCorrection_att.dat', 'r') as f:
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

            interp_r = interpolate.interp2d(R, Z, weight_R, kind='linear')
            interp_z = interpolate.interp2d(R, Z, weight_Z, kind='linear')
            save(interp_r, 'src/interp_r')
            save(interp_z, 'src/interp_z')

    pmt_positions = load('src/pmtcoordinates_ID.json')
    testpaths = load(root_dir + '/testpaths.list')

    # multiprocessing
    p = Pool(processes=40)
    residuals = []
    total = len(testpaths)

    for i in range(5):
        print('getting cwm residuals... %i' % i)
        paths_batch = testpaths[int(0.2 * i * total):int(0.2 * (i + 1) * total)]
        residuals += p.starmap(__job, zip(paths_batch,
                                          repeat(interp_r),
                                          repeat(interp_z),
                                          repeat(pmt_positions)
                                          )
                               )

    residuals = [r for r in residuals if r]

    return np.array(residuals).T


def __job(path, interp_r, interp_z, pmt_positions):
    f = load(path)

    capture_time = f['capture_time']    # scalar value
    hits = int(f['photon_hits'])        # scalar value
    
    hit_counts = f['hit_count']         # vector value
    hit_pmts = f['hit_pmt']             # vector value
    hit_time = f['hit_time']            # vector value
    true_vertex = [f['positron_x'], f['positron_y'], f['positron_z']]

    x = np.zeros(354)
    for i in range(hits):
        pmt = hit_pmts[i]
        count = hit_counts[i]
        t = hit_time[i]

        if pmt in DEAD_PMTS:
            continue

        if t < capture_time:
            x[pmt] += count

    # if the entry is valid, reconstruct the vertex
    if sum(x) > 0:
        # calculate cwm vertex
        reco_vertex = np.array([.0, .0, .0])
        for pmt_id, hits in enumerate(x):
            pmt_pos = pmt_positions[str(pmt_id)]
            reco_vertex += hits * np.array([pmt_pos['x'], pmt_pos['y'], pmt_pos['z']], dtype=float)

        # normalize
        reco_vertex = reco_vertex / sum(x)

        # correction 1
        weight1r = interp_r(np.linalg.norm(reco_vertex[:2]), abs(reco_vertex[2]))
        weight1z = interp_z(np.linalg.norm(reco_vertex[:2]), abs(reco_vertex[2]))
        reco_vertex[:2] *= weight1r
        reco_vertex[2] *= weight1z

        # correction 2
        weight2 = 0.8784552 - 0.0000242758 * np.linalg.norm(reco_vertex[:2])
        reco_vertex *= weight2

        return (reco_vertex - true_vertex).tolist()
    else:
        return False


def main():
    # control group
    print('control group root: ')
    control_root = str(input())

    print('control group name:')
    control_name = str(input())

    # experimental group
    print('# of experimental groups:')
    ex_number = int(input())

    print('experimental group roots (%i):' % ex_number)
    ex_root = []
    for _ in range(ex_number):
        ex_root.append(str(input()))

    print('experimental group names')
    ex_names = []
    for i in range(ex_number):
        print('name for ' + ex_root[i])
        ex_names.append(str(input()))

    # get residuals
    print('calculating residuals')
    control_residual = cwm_residual(root_dir=control_root)
    ex_residuals = [neural_residual(root_dir=ex_root[i]) for i in range(ex_number)]

    # draw histograms
    print('drawing histograms')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis in range(3):
        axes[axis].hist(control_residual[axis],
                        bins=200,
                        density=True,
                        histtype='step',
                        linestyle=':',
                        color='black',
                        label=control_name)

        for i in range(ex_number):
            axes[axis].hist(ex_residuals[i][axis],
                            bins=200,
                            density=True,
                            histtype='step',
                            label=ex_names[i])

        # axes properties
        axis_name = ['x', 'y', 'z'][axis]
        axes[axis].set_xlabel(r'$%s_{rec} - %s_{real} $ (mm)' % (axis_name, axis_name))
        axes[axis].set_ylabel('portion')
        axes[axis].yaxis.set_major_formatter(PercentFormatter(1))
        axes[axis].set_xlim([-1000, 1000])
        axes[axis].set_ylim([0, 1.0/100])
        axes[axis].grid()
        axes[axis].legend(fontsize=8, loc='upper left')

    plt.tight_layout()
    plt.savefig('MC_vis.png')
    plt.close()


if __name__ == '__main__':
    main()
