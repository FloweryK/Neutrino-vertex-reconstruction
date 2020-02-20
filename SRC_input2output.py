import nets
from utils import DEAD_PMTS
from utils import path_list, save, load, mkdir
from vertex_reconstructions import get_cwm_outputs, get_nn_outputs

import argparse
from multiprocessing import Pool
from itertools import repeat
import numpy as np


def job(path, model_root, input_type, net_type):
    print('working on ' + path)
    f = load(path)
    input_charge = f['input_charge']
    input_time = f['input_time']

    if input_type == 'hit':
        batch_nn = [[(np.array(charge) / max(charge)).tolist()] for charge in input_charge]
    elif input_type == 'time':
        batch_nn = [[(np.array(timing) / max(timing)).tolist()] for timing in input_charge]
    elif input_type == 'hit-time':
        batch_nn = [[(np.array(charge) / max(charge)).tolist() + (np.array(timing) / max(timing)).tolist()]
                    for charge, timing in zip(input_charge, input_time)]
    elif input_type == 'hit-time-2c':
        batch_nn = [[(np.array(charge) / max(charge)).tolist(), (np.array(timing) / max(timing)).tolist()]
                    for charge, timing in zip(input_charge, input_time)]
    else:
        print('invalid input_type: ' + input_type)
        raise ValueError

    # get outputs
    output_cwm = get_cwm_outputs(input_charge)
    output_nn = get_nn_outputs(model_directory=model_root,
                               net_type=net_type,
                               inputs=batch_nn,
                               epoch=49,
                               gpu=False)

    # get statistics
    sigma_cwm = np.std(output_cwm, axis=1).tolist()
    sigma_nn = np.std(output_nn, axis=1).tolist()
    mu_cwm = np.mean(output_cwm, axis=1).tolist()
    mu_nn = np.mean(output_nn, axis=1).tolist()

    result = {
        'info': f['info'],
        'outputs_cwm': output_cwm.tolist(),
        'outputs_nn': output_nn.tolist(),
        'sigma_cwm': sigma_cwm,
        'sigma_nn': sigma_nn,
        'mu_cwm': mu_cwm,
        'mu_nn': mu_nn,
    }
    save(result, f"SRC_input2output/{result['info']['detector_type']}_{result['info']['run_number']:06}_{result['info']['subrun_number']:06}.json")


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_root', type=str, required=True, help='model directory for nn')
    parser.add_argument('--input_type', type=str, required=True, help='input type among "hit", "time", "hit-time", "hit-time-2c".')
    parser.add_argument('--net_type', type=str, required=True, help='net type among "Net", "Net2c", "CNN1c", "CNN2c".')
    args = parser.parse_args()
    model_root = args.model_root
    input_type = args.input_type
    net_type = args.net_type

    mkdir('SRC_input2output/')
    p = Pool(processes=40)
    p.starmap(job, zip(path_list(dir='SRC_json2input/', filter='.json'),
                       repeat(model_root),
                       repeat(input_type),
                       repeat(net_type)
                       )
              )


if __name__ == '__main__':
    run()

