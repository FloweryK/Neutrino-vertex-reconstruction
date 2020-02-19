from utils import DEAD_PMTS
from utils import path_list, save, load, mkdir

from multiprocessing import Pool
import numpy as np


def job(path):
    print('working on ' + path)
    f = load(path)

    # result format
    result = {
        'detector_type': f['detector_type'],
        'z': f['z'],
        'run_number': f['run_number'],
        'subrun_number': f['subrun_number'],
        'input_charge': [],
        'input_time': [],
        'valid_count': 0
    }

    # write every valid event
    for event_number, content in f['event'].items():
        input_charge = [0] * 354
        input_time = [1e25] * 354
        for pmt_id, charge, hit_time in zip(content['pmt_id'], content['pmt_charge'], content['pmt_hittime']):
            if (pmt_id > 353) or (pmt_id in DEAD_PMTS):
                continue
            if charge > 0:
                input_charge[pmt_id] += charge
            if hit_time < input_time[pmt_id]:
                input_time[pmt_id] = hit_time

        if (content['event_type'] != 0) and (sum(input_charge) > 0):
            result['valid_count'] += 1
            result['input_charge'].append(input_charge)
            result['input_time'].append(input_time)

    # save the result
    save(result, f"SRC_json2input/{f['detector_type']}_{f['run_number']:06}_{f['subrun_number']:06}.json")


def run():
    mkdir('json2input/')

    # multiprocessing
    p = Pool(processes=40)
    p.map(job, path_list(dir='SRC_prd2json/', filter='.json'))


if __name__ == '__main__':
    run()

