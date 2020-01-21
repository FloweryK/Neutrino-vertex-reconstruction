from utils import DEAD_PMTS
from utils import path_list, save
from vertex_reconstructions import get_cwm_outputs, get_nn_outputs

import argparse
import numpy as np
import ROOT as root


def zpos_dict(path='src/runList_20140821.txt'):
    with open(path, 'r') as f:
        zpos_dict = {}

        while True:
            line = f.readline()
            if not line:
                break

            if 'cm' in line:
                line = line.split(' ')

                if '-' in line[2]:
                    run = line[0]

                    if 'cm' in line[4]:
                        z_position = line[4][0]
                    elif 'cm' in line[3]:
                        z_position = line[3][:2]
                    elif 'cm' in line[2]:
                        z_position = line[2][5:-3]

                    try:
                        run = int(run)
                        z_position = 1600 - int(z_position) * 10
                        zpos_dict[run] = z_position
                    except ValueError:
                        pass

    return zpos_dict


def source_root2inputs(path, event_cut=5000):
    f = root.TFile.Open(path)
    ntp = f.Get('ntp')

    inputs = {}
    for entry in ntp:
        event_number = int(entry._eventnumber)
        event_type = int(entry._eventtype)
        # hit_time = entry._pmt_hittime
        # qsum_buffer = entry._qsumbuffer
        pmt = int(entry._pmt_id)
        charge = entry._pmt_charge

        if (pmt > 353) or (pmt in DEAD_PMTS):
            continue
        if charge < 0:
            continue
        if event_type == 0:
            continue

        if event_number not in inputs:
            if len(inputs) > event_cut:
                break
            else:
                inputs[event_number] = [0]*354

        inputs[event_number][pmt] += charge

    inputs = list(inputs.values())
    inputs = [x for x in inputs if sum(x) > 0]

    return inputs


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=str, required=True, help='source root directory')
    parser.add_argument('-m', type=str, required=True, help='model directory for nn')
    parser.add_argument('-c', type=int, default=5000, help='event number cut for every root file.')
    args = parser.parse_args()

    source_directory = args.s
    model_directory = args.m
    event_cut = args.c

    try:
        zpos = zpos_dict('src/runList_20140821.txt')
    except FileNotFoundError:
        zpos = zpos_dict(path='src/runList_20140821.txt')
        save(zpos, 'src/run_zpos.json')

    paths = path_list(dir=source_directory, filter='.root')
    for i, path in enumerate(paths):
        print(i, len(paths), path)
        run = int(path.split('far_charge_00')[1][:4])
        subrun = int(path.split('far_charge_00')[1][5:11])

        if run not in zpos:
            continue

        if subrun != 1:
            continue

        inputs = source_root2inputs(path, event_cut)
        outputs_cwm = get_cwm_outputs(inputs)
        outputs_nn = get_nn_outputs(model_directory=model_directory,
                                    inputs=[(np.array(x)/max(x)).tolist() for x in inputs],
                                    epoch=49)

        result = {
            'run': run,
            'subrun': subrun,
            'zpos': zpos[run],
            'outputs_cwm': outputs_cwm.tolist(),
            'outputs_nn': outputs_nn.tolist(),
            'sigma_cwm': np.std(outputs_cwm, axis=1).tolist(),
            'sigma_nn': np.std(outputs_nn, axis=1).tolist(),
            'mu_cwm': np.mean(outputs_cwm, axis=1).tolist(),
            'mu_nn': np.mean(outputs_nn, axis=1).tolist(),

        }

        save(result, 'prd2outputs/%05i_%05i.json' % (run, subrun))


if __name__ == '__main__':
    run()