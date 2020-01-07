from utils import load, save, get_labels, get_inputs, get_epoch_outputs
import training

import argparse
import numpy as np
import pandas as pd
from itertools import repeat
from multiprocessing import Pool

import torch.nn as nn
from scipy import interpolate
import matplotlib.pyplot as plt


# NN functions
def get_nn_best_residuals(root_directory):
    # load vali labels, inputs, outputs by epoch.
    vali_labels = get_labels(root_directory, target='vali', form='tensor')
    vali_inputs = get_inputs(root_directory, target='vali', form='tensor')
    vali_epoch_outputs = get_epoch_outputs(root_directory,
                                           inputs=vali_inputs, net=training.Net().float(),
                                           form='tensor')

    min_vali_loss = 1e10
    min_epoch = 0
    criterion = nn.MSELoss()

    for epoch, vali_outputs in enumerate(vali_epoch_outputs):
        vali_loss = criterion(vali_outputs, vali_labels).item() * 1000000

        if vali_loss < min_vali_loss:
            min_epoch = epoch
            min_vali_loss = vali_loss

    test_labels = get_labels(root_directory, target='test', form='numpy')
    test_inputs = get_inputs(root_directory, target='test', form='tensor')
    test_epoch_outputs = get_epoch_outputs(root_directory,
                                           inputs=test_inputs, net=training.Net().float(),
                                           form='numpy')

    residuals = (test_epoch_outputs[min_epoch] - test_labels) * 1000

    return residuals


# CWM functions
def perp(vertex):
    return np.linalg.norm(vertex[:2])


def interpolate_weights(kind, path='WeightingCorrection_att.dat'):
    # load the correction data
    f = open(path, 'r')

    # prepare a empty dataframe
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

    interp_r = interpolate.interp2d(R, Z, weight_R, kind=kind)
    interp_z = interpolate.interp2d(R, Z, weight_Z, kind=kind)

    return interp_r, interp_z


def job(path, interp_r, interp_z, pmt_positions):
    # load json entry file
    f = load(path)

    # get attributes
    capture_time = f['capture_time']  # scalar value
    hits = int(f['photon_hits'])  # scalar value
    hit_counts = f['hit_count']  # vector value
    hit_pmts = f['hit_pmt']  # vector value
    hit_time = f['hit_time']  # vector value

    true_vertex = [f['positron_x'], f['positron_y'], f['positron_z']]

    # get photon hit data (input)
    x = np.zeros(354)

    for i in range(hits):
        count = hit_counts[i]
        pmt = hit_pmts[i]
        t = hit_time[i]

        if t < capture_time:
            x[pmt] += count

    # if the entry is valid, reconstruct the vertex
    reco_vertex = np.array([.0, .0, .0])
    valid_hits = x.sum()

    if (valid_hits > 0) and (valid_hits < sum(hit_counts)):
        for i, n in enumerate(x):
            pmt_pos = pmt_positions[str(i)]
            reco_vertex += n * np.array([pmt_pos['x'], pmt_pos['y'], pmt_pos['z']], dtype=float)

        reco_vertex = reco_vertex / x.sum()

        # correction 1
        weight1r = interp_r(perp(reco_vertex), abs(reco_vertex[2]))
        weight1z = interp_z(perp(reco_vertex), abs(reco_vertex[2]))
        reco_vertex[:2] *= weight1r
        reco_vertex[2] *= weight1z

        # correction 2
        weight2 = 0.8784552 - 0.0000242758 * perp(reco_vertex)
        reco_vertex *= weight2

        return (reco_vertex - true_vertex).tolist()
    else:
        return False


def get_cwm_residuals(test_paths, interpol_kind='linear'):
    # load pmt positions
    pmt_positions = load('pmtcoordinates_ID.json')

    # load Weighting Corrections
    try:
        interp_r = load('interp_r_%s.interp' % interpol_kind)
        interp_z = load('interp_z_%s.interp' % interpol_kind)
    except FileNotFoundError:
        interp_r, interp_z = interpolate_weights('WeightingCorrection_att.dat')
        save(interp_r, 'interp_r_%s.interp' % interpol_kind)
        save(interp_z, 'interp_z_%s.interp' % interpol_kind)

    # multiprocessing
    p = Pool(processes=20)
    residuals = []
    paths = test_paths
    total = len(paths)

    for i in range(10):
        print('getting cwm residuals... %i' % i)
        paths_batch = paths[int(0.1 * i * total):int(0.1 * (i + 1) * total)]
        residuals += p.starmap(job, zip(paths_batch, repeat(interp_r), repeat(interp_z), repeat(pmt_positions)))

    residuals = [r for r in residuals if r]

    return np.array(residuals)


def main():
    # argument configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', type=str, required=True, help='NN root directory')

    args = parser.parse_args()

    nn_directory = args.r

    test_paths = load(nn_directory + '/testpaths.list')

    # get residuals
    try:
        nn_residuals = load(nn_directory + '/nn_residuals.np')
        cwm_residuals = load(nn_directory + '/cwm_residuals.np')
    except FileNotFoundError:
        nn_residuals = get_nn_best_residuals(nn_directory)
        cwm_residuals = get_cwm_residuals(test_paths)

        save(nn_residuals, nn_directory + '/nn_residuals.np')
        save(cwm_residuals, nn_directory + '/cwm_residuals.np')

    nn_residuals = nn_residuals.T
    cwm_residuals = cwm_residuals.T

    nn_sigma = np.std(nn_residuals, axis=1)
    cwm_sigma = np.std(cwm_residuals, axis=1)

    # draw histogram
    for i in range(3):
        nn_hist = plt.hist(nn_residuals[i],
                           bins=400,
                           density=True,
                           histtype='step',
                           color='black',
                           label='neural network')
        cwm_hist = plt.hist(cwm_residuals[i],
                            bins=400,
                            density=True,
                            histtype='step',
                            color='black',
                            linestyle=':',
                            label='cwm + correction')

        text = '$\\sigma_{nn}=%.1fmm$\n$\\sigma_{cwm}=%.1fmm$' % (nn_sigma[i], cwm_sigma[i])
        props = dict(boxstyle='square', fc='w')
        plt.text(180, 0.004, text,
                 horizontalalignment='left', verticalalignment='top', fontsize=12, bbox=props)

        # plt properties
        plt.xlabel('Δ%s (mm)' % ['x', 'y', 'z'][i], fontsize=15)
        plt.ylabel('# of events (portion)', fontsize=15)

        plt.minorticks_on()

        plt.xlim([-500, 500])
        
        plt.legend()

        plt.savefig('NN-CWM_hist_%s.jpg' % ['x', 'y', 'z'][i], bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    main()

