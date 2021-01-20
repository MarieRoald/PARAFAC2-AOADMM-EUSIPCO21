from tempfile import TemporaryDirectory
import shutil
import sys
from datetime import datetime

import h5py
import json
from pathlib import Path
from shutil import copy


import tenkit
import numpy as np
from bcd_tenkit import bcd_tenkit
from tqdm import trange


NOISE_PATH = "noise"
TENSOR_PATH = "evolving_tensor"
H5_GROUPS = ["dataset"]
SLICES_PATH = "dataset/tensor"

NOISE_LEVEL = float(sys.argv[1])
NUM_DATASETS = 50

INNER_TOL = 1e-3
INNER_SUB_ITS = 5

RELATIVE_TOLERANCE = 1e-10
ABSOLUTE_TOLERANCE = 1e-10
MAX_ITERATIONS = 1000
NUM_INITS = 5

MIN_NODES = 3
MAX_NODES = 20

I = 20
J = 30
K = 20
RANK = 3

date = datetime.date(datetime.now())
OUTPUT_PATH = Path(
    f"{date}_noise_{NOISE_LEVEL}_{I}_{J}_{K}_NN".replace(".", "-")
)
DECOMPOSITION_FOLDER = OUTPUT_PATH/"decompositions"
DECOMPOSITION_FOLDER.mkdir(exist_ok=True, parents=True)
RESULTS_FOLDER = OUTPUT_PATH/"results"
RESULTS_FOLDER.mkdir(exist_ok=True, parents=True)
CHECKPOINTS_FOLDER = OUTPUT_PATH/"checkpoints"
CHECKPOINTS_FOLDER.mkdir(exist_ok=True, parents=True)
ALL_CHECKPOINTS_FOLDER = OUTPUT_PATH/"all_checkpoints"
ALL_CHECKPOINTS_FOLDER.mkdir(exist_ok=True, parents=True)

def truncated_normal(size, rng):
    factor = rng.standard_normal(size)
    factor[factor < 0] = 0
    return factor

def generate_component(I, J, K, rank, rng):
    A = truncated_normal((I, rank), rng)
    blueprint_B = truncated_normal((J, rank), rng)
    B = [np.roll(blueprint_B, i, axis=0) for i in range(K)]
    C = rng.uniform(0.1, 1.1, size=(K, rank))

    return tenkit.decomposition.EvolvingTensor(A, B, C)

def generate_noise(I, J, K, rng):
    return rng.standard_normal(size=(I, J, K))

def get_dataset_filename(dataset_num):
    return DECOMPOSITION_FOLDER/f"{dataset_num:03d}.h5"

def store_data(dataset_num, decomposition, noise):
    filename = get_dataset_filename(dataset_num)
    with h5py.File(filename, "w") as h5:
        for group_name in H5_GROUPS:
            h5.create_group(group_name)
        group = h5.create_group(TENSOR_PATH)
        decomposition.store_in_hdf5_group(group)
        h5[NOISE_PATH] = noise.transpose(2, 0, 1)
        h5[SLICES_PATH] = np.asarray(decomposition.construct_slices())

def add_noise(X, noise, noise_level):
    return X + noise_level*noise*np.linalg.norm(X)/np.linalg.norm(noise)

def create_loggers(dataset_num):
    dataset_filename = get_dataset_filename(dataset_num)
    return {
        "Relative SSE": tenkit.decomposition.logging.RelativeSSELogger(),
        "FMS (avg)": tenkit.decomposition.logging.EvolvingTensorFMSLogger(dataset_filename, "evolving_tensor", fms_reduction="mean"),
        "Time": tenkit.decomposition.logging.Timer(),
    }

def run_double_experiment(dataset_num, X, rank):
    best_pf2 = None

    for init in range(NUM_INITS):
        loggers = create_loggers(dataset_num)
        checkpoint_path = ALL_CHECKPOINTS_FOLDER/f"double_split_init-{init:03d}_dataset-{dataset_num:03d}.h5"
        pf2 = bcd_tenkit.BlockEvolvingTensor(
            rank,
            sub_problems=[
                bcd_tenkit.Mode0ADMM(non_negativity=True, max_its=INNER_SUB_ITS, tol=INNER_TOL),
                bcd_tenkit.DoubleSplittingParafac2ADMM(non_negativity=True, max_it=INNER_SUB_ITS, tol=INNER_TOL),
                bcd_tenkit.Mode2ADMM(non_negativity=True, max_its=INNER_SUB_ITS, tol=INNER_TOL),
            ],
            convergence_tol=RELATIVE_TOLERANCE,
            absolute_tol=ABSOLUTE_TOLERANCE,
            loggers=list(loggers.values()),
            max_its=MAX_ITERATIONS,
            checkpoint_path=checkpoint_path,
            checkpoint_frequency=2000,
        )
        pf2.fit(X)

        if best_pf2 is None:
            best_pf2 = pf2
            best_path = checkpoint_path
        elif pf2.SSE + pf2.regularisation_penalty < best_pf2.SSE + best_pf2.regularisation_penalty:
            best_pf2 = pf2
            best_path = checkpoint_path

    shutil.copyfile(best_path, CHECKPOINTS_FOLDER/f"double_split_{dataset_num:03d}.h5")
    
    return best_pf2


def run_als_experiment(dataset_num, X, rank):
    best_pf2 = None

    for init in range(NUM_INITS):
        checkpoint_path = ALL_CHECKPOINTS_FOLDER/f"als_init-{init:03d}_dataset-{dataset_num:03d}.h5"

        loggers = create_loggers(dataset_num)
        pf2 = tenkit.decomposition.Parafac2_ALS(
            rank,
            non_negativity_constraints=[True, False, True],
            convergence_tol=RELATIVE_TOLERANCE,
            loggers=list(loggers.values()),
            max_its=MAX_ITERATIONS,
            checkpoint_path=checkpoint_path,
            checkpoint_frequency=2000,
            print_frequency=-1
        )
        pf2.fit(X)
        if best_pf2 is None:
            best_pf2 = pf2
            best_path = checkpoint_path
        elif pf2.SSE + pf2.regularisation_penalty < best_pf2.SSE + best_pf2.regularisation_penalty:
            best_pf2 = pf2
            best_path = checkpoint_path

    shutil.copyfile(best_path, CHECKPOINTS_FOLDER/f"als_{dataset_num:03d}.h5")
    return best_pf2
     
    
def run_flexible_experiment(dataset_num, X, rank):
    best_pf2 = None

    for init in range(NUM_INITS):
        checkpoint_path = ALL_CHECKPOINTS_FOLDER/f"flexible_coupling_init-{init:03d}_dataset-{dataset_num:03d}.h5"

        loggers = create_loggers(dataset_num)
        pf2 = bcd_tenkit.BlockEvolvingTensor( rank,
            sub_problems=[
                bcd_tenkit.Mode0RLS(non_negativity=True),
                bcd_tenkit.FlexibleCouplingParafac2(non_negativity=True),
                bcd_tenkit.Mode2RLS(non_negativity=True),
            ],
            convergence_tol=RELATIVE_TOLERANCE,
            absolute_tol=ABSOLUTE_TOLERANCE,
            loggers=list(loggers.values()),
            max_its=MAX_ITERATIONS,
            checkpoint_path=checkpoint_path,
            checkpoint_frequency=2000,
            problem_order=(0, 1, 2),
            convergence_method="flex"
        )
        pf2.fit(X)
        if best_pf2 is None:
            best_pf2 = pf2
            best_path = checkpoint_path
        elif pf2.SSE + pf2.regularisation_penalty < best_pf2.SSE + best_pf2.regularisation_penalty:
            best_pf2 = pf2
            best_path = checkpoint_path

    shutil.copyfile(best_path, CHECKPOINTS_FOLDER/f"flexible_coupling_{dataset_num:03d}.h5")
    return best_pf2

def generate_results(dataset_num, decomposer):
    logger_names = create_loggers(dataset_num).keys()
    logs = [logger.log_metrics for logger in decomposer.loggers]
    results = dict(zip(logger_names, logs))
    results['iteration'] = decomposer.loggers[0].log_iterations
    return results
    loggers = decomposer.loggers
    results = {
        'iteration': loggers[0].log_iterations,
        'loss': loggers[0].log_metrics,
        'Fit': loggers[1].log_metrics,
        'Relative SSE': loggers[2].log_metrics,
        'SSE': loggers[3].log_metrics,
        'FMS': loggers[4].log_metrics,
        'FMS_A': loggers[5].log_metrics,
        'FMS_B': loggers[6].log_metrics,
        'FMS_C': loggers[7].log_metrics,
        'coupling_error': loggers[8].log_metrics,
        'time': loggers[9].log_metrics,
        'rho_A': loggers[10].log_metrics,
        'rho_B': loggers[11].log_metrics,
        'rho_C': loggers[12].log_metrics,
        'num_sub_iterations_A': loggers[13].log_metrics,
        'num_sub_iterations_B': loggers[14].log_metrics,
        'num_sub_iterations_C': loggers[15].log_metrics,
    }
    return results

def store_results(dataset_num, prefix, decomposer):
    results = generate_results(dataset_num, decomposer)
    with open(RESULTS_FOLDER/f"{prefix}_{dataset_num:03d}.json", "w") as f:
        json.dump(results, f)

def run_experiment(dataset_num):
    np.random.seed(dataset_num)
    rng = np.random.RandomState(dataset_num)
    decomposition = generate_component(I, J, K, RANK, rng)
    noise = generate_noise(I, J, K, rng)
    store_data(dataset_num, decomposition, noise)

    X = decomposition.construct_tensor()
    noisy_X = add_noise(X, noise, NOISE_LEVEL)

    double_pf2 = run_double_experiment(dataset_num, noisy_X, RANK)
    store_results(dataset_num, "double_split", double_pf2)

    pf2 = run_als_experiment(dataset_num, noisy_X, RANK)
    store_results(dataset_num, "als", pf2)

    flexible_pf2 = run_flexible_experiment(dataset_num, noisy_X, RANK)
    store_results(dataset_num, "flexible_coupling", flexible_pf2)


if __name__ == "__main__":
    np.random.seed(0) 
    for dataset_num in trange(NUM_DATASETS):
        run_experiment(dataset_num)
