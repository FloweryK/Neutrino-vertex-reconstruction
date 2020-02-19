from utils import DEAD_PMTS
from utils import path_list, save, load, mkdir
from vertex_reconstructions import get_cwm_outputs, get_nn_outputs

import argparse
from multiprocessing import Pool
from itertools import repeat
import numpy as np


def job(path, model_root):
    print('working on ' + path)
    f = load(path)

    input_charge = f['input_charge']
    input_time = f['input_time']

    # get outputs
    outputs_cwm = get_cwm_outputs(input_charge)
    outputs_nn = get_nn_outputs(model_directory=model_root,
                                net_type='Net',
                                inputs=[[(np.array(charge) / max(charge)).tolist()]
                                        for charge, timing in zip(input_charge, input_time)],
                                epoch=49,
                                gpu=False)
    '''
    outputs_nn = get_nn_outputs(model_directory=model_root,
                                inputs=[[(np.array(charge) / max(charge)).tolist() + (np.array(timing) / max(timing)).tolist()]
                                        for charge, timing in zip(input_charge, input_time)],
                                epoch=49,
                                gpu=False)
    '''

    # get statistics
    sigma_cwm = np.std(outputs_cwm, axis=1).tolist()
    sigma_nn = np.std(outputs_nn, axis=1).tolist()
    mu_cwm = np.mean(outputs_cwm, axis=1).tolist()
    mu_nn = np.mean(outputs_nn, axis=1).tolist()

    result = {
        'info': f['info'],
        'outputs_cwm': outputs_cwm.tolist(),
        'outputs_nn': outputs_nn.tolist(),
        'sigma_cwm': sigma_cwm,
        'sigma_nn': sigma_nn,
        'mu_cwm': mu_cwm,
        'mu_nn': mu_nn,
    }
    save(result, f"SRC_input2output/{result['info']['detector_type']}_{result['info']['run_number']:06}_{result['info']['subrun_number']:06}.json")


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_root', type=str, required=True, help='model directory for nn')
    args = parser.parse_args()
    model_root = args.model_root

    mkdir('SRC_input2output/')
    p = Pool(processes=40)
    p.starmap(job, zip(path_list(dir='SRC_json2input/', filter='.json'),
                       repeat(model_root)))


if __name__ == '__main__':
    run()

