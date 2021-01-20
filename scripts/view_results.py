from collections import defaultdict
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.interpolate import interp1d

import tenkit
from tenkit_tools.utils import load_best_group
from sklearn.metrics import roc_auc_score
import h5py

RUN_FOLDER = Path("201128_noise_05_20_30_40")
LOG_FOLDER = RUN_FOLDER/"results"

NUM_EXPERIMENTS = 50
MAX_ITS = 1000
TIME_NAME = 'time'

EXPERIMENT_NAMES = [
    "double_split",
    "single_split_Dk",
#    "double_split_NOPF2"
    "flexible_coupling"
]


def load_logs(log_folder: Path, experiment_num: int) -> dict:
    logs = {}
    for experiment_name in EXPERIMENT_NAMES:
        with (log_folder/f"{experiment_name}_{experiment_num:03d}.json").open() as f:
            logs[experiment_name] = json.load(f)
    return logs
    #with (log_folder/f"single_split_C_{experiment_num:03d}.json").open() as f:
    #    single_split_C_results = json.load(f)

    #with (log_folder/f"single_split_Dk_{experiment_num:03d}.json").open() as f:
    #    single_split_Dk_results = json.load(f)

    #with (log_folder/f"flexible_coupling_{experiment_num:03d}.json").open() as f:
    #    flexible_coupling_results = json.load(f)

    #return double_split_results, single_split_Dk_results
    #return double_split_results, single_split_C_results, single_split_Dk_results, flexible_coupling_results


def load_double_split_logs(log_folder: Path, experiment_num: int) -> dict:
    with (log_folder/f"double_split_{experiment_num:03d}.json").open() as f:
        double_split_results = json.load(f)

    return double_split_results,


def load_checkpoint(log_folder: Path, experiment_name: str, experiment_num: int) -> tenkit.decomposition.EvolvingTensor:
    checkpoint_folder = log_folder/"checkpoints"
    EvolvingTensor = tenkit.decomposition.EvolvingTensor

    with h5py.File(checkpoint_folder/f"{experiment_name}_{experiment_num:03d}.h5", "r") as h5:
        group = load_best_group(h5)
        estimated = EvolvingTensor.load_from_hdf5_group(group)
    return estimated


def load_checkpoints(log_folder: Path, experiment_num: int) -> list:
    return {experiment_name: load_checkpoint(log_folder, experiment_name, experiment_num)
            for experiment_name in EXPERIMENT_NAMES}


def load_decomposition(log_folder: Path, experiment_num: int) -> tenkit.decomposition.EvolvingTensor:
    checkpoint_folder = log_folder/"decompositions"
    EvolvingTensor = tenkit.decomposition.EvolvingTensor
    with h5py.File(checkpoint_folder/f"{experiment_num:03d}.h5", "r") as h5:
        estimated = EvolvingTensor.load_from_hdf5_group(h5["evolving_tensor"])
    return estimated


def compute_accuracies(log_folder: Path, progress=False) -> dict:
    accuracies = {}
    if progress:
        range_ = trange
    else:
        range_ = range
    for experiment_num in range_(NUM_EXPERIMENTS):
        checkpoints = load_checkpoints(log_folder, experiment_num)
        true = load_decomposition(log_folder, experiment_num)
        for name, decomposition in checkpoints.items():
            if name not in accuracies:
                accuracies[name] = {
                    'Sensitivity': [],
                    'Specificity': [],
                    'Dice': [],
                    'ROC AUC': [],
                }

            
            # Binarize
            B = np.array(decomposition.B)
            B /= np.linalg.norm(B, axis=1, keepdims=True)
            estimated_map = abs(B) > 1e-8
            true_map = np.array(true.B) > 1e-8

            # Compute metrics
            accuracies[name]['Sensitivity'].append(np.sum(estimated_map*true_map) / np.sum(true_map))
            accuracies[name]['Specificity'].append(np.sum((1 - estimated_map)*(1 - true_map)) / np.sum(1 - true_map))
            accuracies[name]['Dice'].append(2*np.sum(estimated_map*true_map) / (np.sum(true_map) + np.sum(estimated_map)))
            accuracies[name]['ROC AUC'].append(roc_auc_score(true_map.ravel().astype(int), B.ravel()))
    return accuracies


def create_summaries(experiment_log: dict) -> dict:
    """Takes a single result dict as input and creates a summary.

    Summary just contains the logs for the final iteration.
    """
    summary = {}
    for key, value in experiment_log.items():
        summary[key] = value[-1]
    return summary


def load_summaries(log_folder, num_experiments: int) -> (dict, dict):
    """Take number of experiments as input and return two dicts, one for logs and one for summaries.

    The keys of these dicts are the experiment types (e.g. double_split) and the values are dictionaries of lists.

    The keys of the inner dictionaries are log-types (e.g. fms) and the values are lists.

    The i-th element of these lists are the logs and summaries for the i-th experiment.
    """
    logs = {
        experiment_name: defaultdict(list) for experiment_name in EXPERIMENT_NAMES
    }
    summaries = {
        experiment_name: defaultdict(list) for experiment_name in EXPERIMENT_NAMES
    }
    for i in range(num_experiments):
        for experiment_name, log in load_logs(log_folder, i).items():
            for key, value in log.items():
                logs[experiment_name][key].append(value)
        for experiment_name, log in load_logs(log_folder, i).items():
            summary = create_summaries(log)
            for key, value in summary.items():
                summaries[experiment_name][key].append(value)
    
    logs = {key: dict(value) for key, value in logs.items()}
    summaries = {key: dict(value) for key, value in summaries.items()}
    return logs, summaries


def load_double_split_summaries(log_folder, num_experiments: int) -> (dict, dict):
    """Take number of experiments as input and return two dicts, one for logs and one for summaries.

    The keys of these dicts are the experiment types (e.g. double_split) and the values are dictionaries of lists.

    The keys of the inner dictionaries are log-types (e.g. fms) and the values are lists.

    The i-th element of these lists are the logs and summaries for the i-th experiment.
    """
    experiment_names = (
        'double_split',
    )
    logs = {
        experiment_name: defaultdict(list) for experiment_name in experiment_names
    }
    summaries = {
        experiment_name: defaultdict(list) for experiment_name in experiment_names
    }
    for i in range(num_experiments):
        for experiment_name, log in zip(experiment_names, load_double_split_logs(log_folder, i)):
            for key, value in log.items():
                logs[experiment_name][key].append(value)
        for experiment_name, log in zip(experiment_names, load_double_split_logs(log_folder, i)):
            summary = create_summaries(log)
            for key, value in summary.items():
                summaries[experiment_name][key].append(value)
        
    return logs, summaries


def make_log_array(log_list: list) -> np.array:
    """Takes uneven list of logs and creates a 2D numpy array where the last element of each list is used as padding.
    """
    log_array = np.zeros((NUM_EXPERIMENTS, MAX_ITS))
    for i, log in enumerate(log_list):
        num_its = len(log)
        log_array[i, :num_its] = log
        log_array[i, num_its:] = log[-1]
    return log_array


def make_summary_df(summaries):
    """Convert nested dictionary of summaries (inner dicts represent summaries for one method) into a single dataframe.
    """
    summary_dfs = {method: pd.DataFrame(summary) for method, summary in summaries.items()}
    for method, summary_df in summary_dfs.items():
        summary_df['Method'] = method
        summary_df["Dataset num"] = summary_df.index
    for method in summary_dfs:
        summary_dfs[method] = summary_dfs[method].set_index(["Method", "Dataset num"])

    return pd.concat(summary_dfs)


def plot_with_bounds(ax, xdata, ydata, label, log_scale=False, fms_log_scale=False):
    assert not (log_scale and fms_log_scale)
    if log_scale:
        ax.plot(xdata, np.median(ydata, axis=0), label=label)
        ax.fill_between(xdata, np.quantile(ydata, 0.25, axis=0), np.quantile(ydata, 0.75, axis=0), alpha=0.3)
        ax.set_yscale('log')
    elif fms_log_scale:
        ydata = 1 - ydata
        ax.plot(xdata, np.median(ydata, axis=0), label=label)
        ax.fill_between(xdata, np.quantile(ydata, 0.25, axis=0), np.quantile(ydata, 0.75, axis=0), alpha=0.3)
        ax.set_yscale('log')
        ax.set_ylim(1, 0.01)
        yticks = [1, 0.1, 0.01]
        yticklabels = [str(1 - ytick) for ytick in yticks]
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
    else:
        ax.plot(xdata, np.median(ydata, axis=0), label=label)
        ax.fill_between(xdata, np.quantile(ydata, 0.25, axis=0), np.quantile(ydata, 0.75, axis=0), alpha=0.3)



def make_log_plot(logs, log_name, ax=None, log_scale=False, fms_log_scale=False, legend=True):
    if ax is None:
        ax = plt.gca()
    for method, log_dict in logs.items():
        log = make_log_array(log_dict[log_name])
        it_num = np.arange(1, MAX_ITS+1)
        plot_with_bounds(ax, it_num, log, label=method, log_scale=log_scale, fms_log_scale=fms_log_scale)
        #ax.plot(it_num, np.median(log, axis=0), label=method)
        #ax.fill_between(it_num, np.quantile(log, 0.25, axis=0), np.quantile(log, 0.75, axis=0), alpha=0.3)
    if legend:
        ax.legend()
    ax.set_xlabel("Iteration")
    return ax


def make_time_per_it_plot(logs, ax=None):
    if ax is None:
        ax = plt.gca()
    for method, log_dict in logs.items():
        time = make_log_array(log_dict[TIME_NAME])
        time_diff = time[:, 1:] - time[:, :-1]
        it_num = np.arange(1, MAX_ITS)
        ax.plot(it_num, np.median(time_diff, axis=0), label=method)
        ax.fill_between(it_num, np.quantile(time_diff, 0.25, axis=0), np.quantile(time_diff, 0.75, axis=0), alpha=0.3)
    ax.legend()
    ax.set_xlabel("Iteration")
    return ax


def make_log_time_plot(logs, log_name, ax=None, log_scale=False, fms_log_scale=False, legend=True):
    if ax is None:
        ax = plt.gca()
    max_time = np.max([np.max(make_log_array(log_dict[TIME_NAME])) for log_dict in logs.values()])
    min_time = np.min([np.min(make_log_array(log_dict[TIME_NAME])) for log_dict in logs.values()])
    uniform_time = np.linspace(min_time, max_time, 100000)
    for method, log_dict in logs.items():
        log = make_log_array(log_dict[log_name])
        time = make_log_array(log_dict[TIME_NAME])
        uniform_log = np.zeros((log.shape[0], 100000))
        for row, _ in enumerate(uniform_log):
            interpolant = interp1d(time[row], log[row], fill_value=(log[row, 0], log[row, -1]), bounds_error=False, kind="linear")
            uniform_log[row] = interpolant(uniform_time)
        
        plot_with_bounds(ax, uniform_time, uniform_log, label=method, log_scale=log_scale, fms_log_scale=fms_log_scale)
        
        #ax.plot(uniform_time, np.median(uniform_log, axis=0), label=method)
        #ax.fill_between(uniform_time, np.quantile(uniform_log, 0.25, axis=0), np.quantile(uniform_log, 0.75, axis=0), alpha=0.3)
    if legend:
        ax.legend()
    ax.set_xlabel("Time [s]")
    return ax

