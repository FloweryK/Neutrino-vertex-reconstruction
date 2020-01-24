from utils import DEAD_PMTS
from utils import path_list, save

import argparse
from itertools import repeat
from multiprocessing import Pool
import numpy as np
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


def pmt_performace(path='src/pmtPerformance_far_20110903.dat'):
    result = {}
    with open(path, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break

            line = line.split('\t')
            line = [l for l in line if l]

            pmt = int(line[0])
            qe = float(line[6])
            re = float(line[7])

            result[pmt] = {
                'qe': qe,
                're': re
            }

    return result


def job(path, z, pmtperformance, event_cut):
    def __source2inputs(path, pmtperformance, event_cut=5000):
        # load source root file and get ntp
        f = root.TFile.Open(path)
        ntp = f.Get('ntp')

        # get inputs
        inputs = {}
        for entry in ntp:
            # get values
            event_number = int(entry._eventnumber)
            event_type = int(entry._eventtype)
            pmt = int(entry._pmt_id)
            charge = entry._pmt_charge

            # breakpoints
            if (pmt > 353) or (pmt in DEAD_PMTS):
                continue
            if charge < 0:
                continue
            if event_type == 0:
                continue
            if len(inputs) > event_cut:
                break

            # make input value for each pmt
            if event_number not in inputs:
                inputs[event_number] = [0] * 354
            re = pmtperformance[pmt]['re']
            qe = pmtperformance[pmt]['qe']
            nphoton = charge / (re * qe)
            inputs[event_number][pmt] += nphoton

        # dict to value list, and discard empty inputs
        inputs = list(inputs.values())
        inputs = [x for x in inputs if sum(x) > 0]

        return inputs

    # get info
    print(path)
    run = int(path.split('far_charge_00')[1][:4])
    subrun = int(path.split('far_charge_00')[1][5:11])

    # breakpoints
    if run not in z:
        return
    if subrun != 1:
        return

    # get inputs
    inputs = __source2inputs(path, pmtperformance, event_cut)

    result = {
        'run': run,
        'subrun': subrun,
        'z': z[run],
        'inputs': inputs
    }
    save(result, 'prd2inputs/%05i_%05i.json' % (run, subrun))


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=str, required=True, help='source root directory')
    parser.add_argument('-c', type=int, default=5000, help='event number cut for every root file.')
    args = parser.parse_args()

    source_directory = args.s
    event_cut = args.c

    # load source
    z = zpos('src/runList_20140821.txt')
    pmtperformance = pmt_performace('src/pmtPerformance_far_20110903.dat')

    # multiprocessing
    paths = path_list(dir=source_directory, filter='.root')
    p = Pool(processes=40)
    p.starmap(job, zip(paths, repeat(z), repeat(pmtperformance), repeat(event_cut)))


if __name__ == '__main__':
    run()

