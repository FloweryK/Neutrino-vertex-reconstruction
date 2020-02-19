from utils import path_list, save, mkdir

import re
import argparse
from itertools import repeat
from multiprocessing import Pool
import ROOT as root


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
                    run_number = line[0]
                    if 'cm' in line[4]:
                        z = line[4][0]
                    elif 'cm' in line[3]:
                        z = line[3][:2]
                    elif 'cm' in line[2]:
                        z = line[2][5:-3]
                    try:
                        run_number = int(run_number)
                        z = 1600 - int(z) * 10
                        result[run_number] = z
                    except ValueError:
                        pass
    return result


def job(path, z, event_cut):
    print('working on ' + path)
    parsed_path = re.split('_|\.', path.split('/')[-1])
    detector_type = parsed_path[0]
    run_number = int(parsed_path[2])
    subrun_number = int(parsed_path[3])

    if run_number not in z:
        return
    if subrun_number != 1:
        return

    # result format
    result = {
        'info': {
            'detector_type': detector_type,
            'z': z[run_number],
            'run_number': run_number,
            'subrun_number': subrun_number
        },
        'event': {}
    }

    # convert every root entry into json format
    f = root.TFile.Open(path)
    ntp = f.Get('ntp')
    for entry in ntp:
        if len(result['event']) >= event_cut:
            break

        event_number = int(entry._eventnumber)
        event_type = int(entry._eventtype)
        event_time = entry._eventtime
        qsum_buffer = float(entry._qsumbuffer)
        qmax_buffer = float(entry._qmaxbuffer)
        nhit_buffer = int(entry._nhitbuffer)
        nhit_veto = int(entry._nhitveto)
        pmt_id = int(entry._pmt_id)
        pmt_charge = float(entry._pmt_charge)
        pmt_hittime = float(entry._pmt_hittime)

        if event_number not in result['event']:
            result['event'][event_number] = {
                'event_number': event_number,
                'event_type': event_type,
                'event_time': event_time,
                'qsum_buffer': qsum_buffer,
                'qmax_buffer': qmax_buffer,
                'nhit_buffer': nhit_buffer,
                'nhit_veto': nhit_veto,
                'pmt_id': [],
                'pmt_charge': [],
                'pmt_hittime': []
            }

        if run_number != int(entry._run_number):
            print(f'run number changed at {path}')
            raise ValueError
        if subrun_number != int(entry._subrun_number):
            print(f'subrun number changed at {path}')
            raise ValueError
        if event_type != result['event'][event_number]['event_type']:
            print(f'event type changed at {path}')
            raise ValueError
        if event_time != result['event'][event_number]['event_time']:
            print(f'event time changed at {path}')
            raise ValueError
        if qsum_buffer != result['event'][event_number]['qsum_buffer']:
            print(f'qsum buffer changed at {path}')
            raise ValueError
        if qmax_buffer != result['event'][event_number]['qmax_buffer']:
            print(f'qmax buffer changed at {path}')
            raise ValueError
        if nhit_buffer != result['event'][event_number]['nhit_buffer']:
            print(f'nhit buffer changed at {path}')
            raise ValueError
        if nhit_veto != result['event'][event_number]['nhit_veto']:
            print(f'nhit veto changed at {path}')
            raise ValueError

        result['event'][event_number]['pmt_id'].append(pmt_id)
        result['event'][event_number]['pmt_charge'].append(pmt_charge)
        result['event'][event_number]['pmt_hittime'].append(pmt_hittime)

    # save the result
    save(result, f'SRC_prd2json/{detector_type}_{run_number:06}_{subrun_number:06}.json')


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_root', type=str, required=True, help='source root directory')
    parser.add_argument('--event_limit', type=int, default=5000, help='event number limit every root file.')
    args = parser.parse_args()
    source_root = args.source_root
    event_limit = args.event_limit

    # multiprocessing
    mkdir('SRC_prd2json/')
    p = Pool(processes=40)
    p.starmap(job, zip(path_list(dir=source_root, filter='.root'),
                       repeat(zpos('src/runList_20140821.txt')),
                       repeat(event_limit)))


if __name__ == '__main__':
    run()

