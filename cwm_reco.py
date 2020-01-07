from utils import load, save, path_list
import argparse
import numpy as np
import pandas as pd
from itertools import repeat
from multiprocessing import Pool
from scipy import interpolate
import matplotlib.pyplot as plt


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


def filter_range(d, left, right):
    return [x for x in d if (x > left) and (x < right)]


def gaussian(x, a, mu, std):
    return a * np.exp(-((x - mu) / 4 / std)**2)


def main():
    # argument configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', type=str, default='MC', help='MC root directory')
    parser.add_argument('-k', type=str, default='linear', help='interpolation kind')

    args = parser.parse_args()
    root_directory = args.r
    interpol_kind = args.k
    
    print(interpol_kind)

    try:
        residuals = load('CWM_reco_residuals.list')
    except FileNotFoundError:
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
        paths = path_list(root_directory)
        total = len(paths)

        for i in range(10):
            print(i)
            paths_batch = paths[int(0.1*i*total):int(0.1*(i+1)*total)]
            residuals += p.starmap(job, zip(paths_batch, repeat(interp_r), repeat(interp_z), repeat(pmt_positions)))

            residuals = [r for r in residuals if r]
            print(np.std(residuals, axis=0))
        save(residuals, 'CWM_reco_residuals.list')

    # draw histogram
    residuals = np.array(residuals).T

    dx = residuals[0]
    dy = residuals[1]
    dz = residuals[2]

    print(np.std(residuals, axis=1))

    plt.hist(dz,
             bins=400,
             density=True,
             histtype='step',
             color='black',
             label='cwm + correction')

    plt.xlabel('Δz (mm)', fontsize=15)
    plt.ylabel('# of events (portion)', fontsize=15)

    plt.minorticks_on()

    plt.xlim([-500, 500])

    plt.savefig('CWM_reco-test.jpg')


if __name__ == '__main__':
    main()

