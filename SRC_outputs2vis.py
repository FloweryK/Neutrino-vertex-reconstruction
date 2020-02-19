from utils import path_list, load
import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

N_BIN = 200


def filter_nsigma(outputs, n):
    ns, bins = np.histogram(outputs, bins=N_BIN)
    peak = bins[np.argmax(ns)]
    std = np.std(outputs)
    return [output for output in outputs if (peak-n*std) < output < (peak+n*std)]


def standard_error(output):
    return np.std(output) / np.sqrt(len(output))


def run():
    X = []
    Y_rec_cwm = []
    Y_rec_nn = []
    Y_width_cwm = []
    Y_width_nn = []
    Y_err_cwm = []
    Y_err_nn = []

    paths = path_list('SRC_input2output/')
    for path in paths:
        f = load(path)
        z = f['info']['z']
        if z in X:
            break
        print(f'working on z={z} in {path}')

        # get 3-axis outputs of cwm, nn
        outputs_cwm = f['outputs_cwm']
        outputs_nn = f['outputs_nn']

        # filter desired ranges
        filtered_cwm = [filter_nsigma(outputs, n=2) for outputs in outputs_cwm]
        filtered_nn = [filter_nsigma(outputs, n=2) for outputs in outputs_nn]

        # get rec
        rec_cwm = [np.mean(output) for output in filtered_cwm]
        rec_nn = [np.mean(output) for output in filtered_nn]

        # get std
        std_cwm = [np.std(output) for output in filtered_cwm]
        std_nn = [np.std(output) for output in filtered_nn]

        # get std error
        se_cwm = [standard_error(output) for output in filtered_cwm]
        se_nn = [standard_error(output) for output in filtered_nn]

        # make x, y axis data
        X.append(z)
        Y_rec_cwm.append(rec_cwm)
        Y_rec_nn.append(rec_nn)
        Y_width_cwm.append(std_cwm)
        Y_width_nn.append(std_nn)
        Y_err_cwm.append(se_cwm)
        Y_err_nn.append(se_nn)

    # make np array, and transpose
    X = np.array(X)
    Y_rec_cwm = np.array(Y_rec_cwm)
    Y_rec_nn = np.array(Y_rec_nn)
    Y_width_cwm = np.array(Y_width_cwm).T
    Y_width_nn = np.array(Y_width_nn).T
    Y_err_cwm = np.array(Y_err_cwm).T
    Y_err_nn = np.array(Y_err_nn).T

    Y_residual_cwm = np.array(Y_rec_cwm).T
    Y_residual_cwm[2] += -X
    Y_residual_nn = np.array(Y_rec_nn).T
    Y_residual_nn[2] += -X

    # sigmoid correction

    # draw plots: true, rec position
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis in range(3):
        # plot cwm, nn
        if axis is 2:
            axes[axis].errorbar(X, Y_residual_cwm[axis] + X, yerr=Y_err_cwm[axis], label='cwm with correction', color='black', markersize=3, linewidth=1, capsize=3, fmt='o')
            axes[axis].errorbar(X, Y_residual_nn[axis] + X, yerr=Y_err_nn[axis], label='neural network', color='r', markersize=3, linewidth=1, capsize=3, fmt='o')
        else:
            axes[axis].errorbar(X, Y_residual_cwm[axis], yerr=Y_err_cwm[axis], label='cwm with correction', color='black', markersize=3, linewidth=1, capsize=3, fmt='o')
            axes[axis].errorbar(X, Y_residual_nn[axis], yerr=Y_err_nn[axis], label='neural network', color='r', markersize=3, linewidth=1, capsize=3, fmt='o')

        # axes properties
        axis_name = ['x', 'y', 'z'][axis]
        axes[axis].set_xlabel(r'$z_{src}$ (mm)')
        axes[axis].set_ylabel(r'$%s_{rec}$ (mm)' % axis_name)
        axes[axis].grid()
        axes[axis].legend(fontsize=8, loc='upper left')
        if axis is 2:
            axes[axis].set_ylim([-1600, 1600])
        else:
            axes[axis].set_ylim([-100, 100])
    plt.tight_layout()
    plt.savefig('1_rec.png')
    plt.close()

    # draw plots: residual
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis in range(3):
        # plot cwm, nn
        axes[axis].errorbar(X, Y_residual_cwm[axis], yerr=Y_err_cwm[axis], label='cwm with correction', color='black', markersize=3, linewidth=1, capsize=3, fmt='o')
        axes[axis].errorbar(X, Y_residual_nn[axis], yerr=Y_err_nn[axis], label='neural network', color='r', markersize=3, linewidth=1, capsize=3, fmt='o')

        # axes properties
        axis_name = ['x', 'y', 'z'][axis]
        axes[axis].set_xlabel(r'$z_{src}$ (mm)')
        axes[axis].set_ylabel(r'$%s_{rec}-%s_{src}$ (mm)' % (axis_name, axis_name))
        axes[axis].grid()
        axes[axis].legend(fontsize=8, loc='upper left')
        if axis is 2:
            axes[axis].set_ylim([-150, 150])
        else:
            axes[axis].set_ylim([-100, 100])
    plt.tight_layout()
    plt.savefig('2_residual.png')
    plt.close()

    # draw plots: width
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis in range(3):
        # plot cwm, nn
        axes[axis].errorbar(X, Y_width_cwm[axis], yerr=Y_err_cwm[axis], label='cwm with correction', color='black', markersize=3, linewidth=1, capsize=3, fmt='o')
        axes[axis].errorbar(X, Y_width_nn[axis], yerr=Y_err_nn[axis], label='neural network', color='r', markersize=3, linewidth=1, capsize=3, fmt='o')

        # axes properties
        axis_name = ['x', 'y', 'z'][axis]
        axes[axis].set_xlabel(r'$z_{src}$ (mm)')
        axes[axis].set_ylabel('%s width (mm)' % axis_name)
        axes[axis].grid()
        axes[axis].set_ylim([150, 330])
        axes[axis].legend(fontsize=8, loc='upper left')
    plt.tight_layout()
    plt.savefig('3_width.png')
    plt.close()

    # draw plots: improvement
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis in range(3):
        # plot cwm, nn
        improvement = (Y_width_nn[axis] - Y_width_cwm[axis]) / Y_width_cwm[axis] * 100
        mean_imp = np.mean(improvement)
        axes[axis].plot(X, improvement, marker='.', color='black')
        axes[axis].axhline(y=mean_imp, color='blue', linestyle=':')

        # axes properties
        axis_name = ['x', 'y', 'z'][axis]
        axes[axis].set_xlabel(r'$z_{src} (mm)$')
        axes[axis].set_ylabel('$(\sigma_{%s, nn}-\sigma_{%s, cwm})/\sigma_{%s, cwm}$ (%%)' % (axis_name, axis_name, axis_name))
        axes[axis].set_ylim([-40, 15])
        axes[axis].grid()

        axes[axis].text(0, mean_imp-2, 'mean=%.1f%%' % mean_imp, ha='left', va='top', color='blue')
    plt.tight_layout()
    plt.savefig('4_improvement.png')
    plt.close()


if __name__ == '__main__':
    run()

