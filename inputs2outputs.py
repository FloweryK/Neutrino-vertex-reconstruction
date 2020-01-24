from utils import DEAD_PMTS
from utils import path_list, save, load
from vertex_reconstructions import get_cwm_outputs, get_nn_outputs

import argparse
import numpy as np


def zpos(path='src/runList_20140821.txt'):
    with open(path, 'r') as f:
        result = {}

        while True:
            line = f.readline()
            if not line:
                break

            if 'cm' in line:
                line = line.split(' ')

                if '-' in line[2]:
                    run = line[0]

                    if 'cm' in line[4]:
                        z = line[4][0]
                    elif 'cm' in line[3]:
                        z = line[3][:2]
                    elif 'cm' in line[2]:
                        z = line[2][5:-3]

                    try:
                        run = int(run)
                        z = 1600 - int(z) * 10
                        result[run] = z
                    except ValueError:
                        pass

    return result


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=str, required=True, help='model directory for nn')
    args = parser.parse_args()

    model_directory = args.m

    # load sources
    z = zpos('src/runList_20140821.txt')

    paths = path_list(dir='prd2inputs/', filter='.json')
    for i, path in enumerate(paths):
        print(i, len(paths))

        # get basic info
        filename = path.split('/')[-1]
        run = int(filename.split('_')[0])
        subrun = int(filename.split('_')[1][:5])

        # breakpoints
        if run not in z:
            return
        if subrun != 1:
            return

        # get inputs
        file = load(path)
        inputs = file['inputs']

        # get outputs
        outputs_cwm = get_cwm_outputs(inputs)
        outputs_nn = get_nn_outputs(model_directory=model_directory, inputs=[(np.array(x) / max(x)).tolist() for x in inputs], epoch=49)

        # get statistics
        sigma_cwm = np.std(outputs_cwm, axis=1).tolist()
        sigma_nn = np.std(outputs_nn, axis=1).tolist()
        mu_cwm = np.mean(outputs_cwm, axis=1).tolist()
        mu_nn = np.mean(outputs_nn, axis=1).tolist()

        result = {
            'run': run,
            'subrun': subrun,
            'z': z[run],
            'outputs_cwm': outputs_cwm.tolist(),
            'outputs_nn': outputs_nn.tolist(),
            'sigma_cwm': sigma_cwm,
            'sigma_nn': sigma_nn,
            'mu_cwm': mu_cwm,
            'mu_nn': mu_nn,
        }
        save(result, 'inputs2outputs/%05i_%05i.json' % (run, subrun))

        # print
        for axis in range(3):
            improve_from_cwm_to_nn = (sigma_nn[axis] - sigma_cwm[axis]) / sigma_cwm[axis] * 100
            print('run=%i, axis=%i, z=%i, mu(cwm, nn): (%.1f, %.1f), sigma(cwm, nn): (%.1f, %.1f) (%.1f %%)'
                  % (run, axis, z[run], mu_cwm[axis], mu_nn[axis], sigma_cwm[axis], sigma_nn[axis], improve_from_cwm_to_nn))


if __name__ == '__main__':
    run()

