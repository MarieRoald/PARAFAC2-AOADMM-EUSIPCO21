import sys
from datetime import datetime

import h5py
import json
from pathlib import Path
from shutil import copy
import shutil

import tenkit
import numpy as np
from scipy import io
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

I = 20
J = 201
K = 40
RANK = 3

RIDGE_B = False
RIDGE_PENALTY = float(sys.argv[2])
L2_LEVEL = float(sys.argv[3])
L2_SIMILARITY = np.zeros((J, J))
for j in range(J):
    if j > 0:
        L2_SIMILARITY[j, j] += 1
        L2_SIMILARITY[j, j-1] -= 1
    if j < J-1:
        L2_SIMILARITY[j, j] += 1
        L2_SIMILARITY[j, j+1] -= 1
L2_SIMILARITY *= L2_LEVEL
INIT = sys.argv[4]
print(INIT, bcd_tenkit.INIT_METHOD_A, bcd_tenkit.INIT_METHOD_B, bcd_tenkit.INIT_METHOD_C)


date = datetime.date(datetime.now())
print("Experiment with")
print(f"  noise  level: {NOISE_LEVEL}")
print(f"  smooth level: {L2_LEVEL}")
print(f"  ridge  level: {RIDGE_PENALTY}")
print(f"  ridge  B: {RIDGE_B}")

if RIDGE_B:
    SUFFIX = "RIDGEABC"
else:
    SUFFIX = "RIDGEAC"

OUTPUT_PATH = Path(
    f"{date}/noise_{NOISE_LEVEL}_20_201_40_INIT_{INIT}_SMOOTH_{L2_LEVEL}_RIDGE_{RIDGE_PENALTY}_{SUFFIX}".replace(".", "-")
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
    blueprint_B = io.loadmat("B.mat")["temp"]
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
        "SSE": tenkit.decomposition.logging.SSELogger(),
        "FMS": tenkit.decomposition.logging.EvolvingTensorFMSLogger(dataset_filename, "evolving_tensor", fms_reduction="min"),
        "FMS A": tenkit.decomposition.logging.EvolvingTensorFMSALogger(dataset_filename, "evolving_tensor", fms_reduction="min"),
        "FMS B": tenkit.decomposition.logging.EvolvingTensorFMSBLogger(dataset_filename, "evolving_tensor", fms_reduction="min"),
        "FMS C": tenkit.decomposition.logging.EvolvingTensorFMSCLogger(dataset_filename, "evolving_tensor", fms_reduction="min"),
        "FMS (avg)": tenkit.decomposition.logging.EvolvingTensorFMSLogger(dataset_filename, "evolving_tensor", fms_reduction="mean"),
        "FMS A (avg)": tenkit.decomposition.logging.EvolvingTensorFMSALogger(dataset_filename, "evolving_tensor", fms_reduction="mean"),
        "FMS B (avg)": tenkit.decomposition.logging.EvolvingTensorFMSBLogger(dataset_filename, "evolving_tensor", fms_reduction="mean"),
        "FMS C (avg)": tenkit.decomposition.logging.EvolvingTensorFMSCLogger(dataset_filename, "evolving_tensor", fms_reduction="mean"),
        "Time": tenkit.decomposition.logging.Timer(),
        "Coupling error": tenkit.decomposition.logging.CouplingErrorLogger(),
        "Coupling error 0": tenkit.decomposition.logging.SingleCouplingErrorLogger(0),
        "Coupling error 1": tenkit.decomposition.logging.SingleCouplingErrorLogger(1),
        "Coupling error 2": tenkit.decomposition.logging.SingleCouplingErrorLogger(2),
        "Coupling error 3": tenkit.decomposition.logging.SingleCouplingErrorLogger(3),
        "Regulariser 0": tenkit.decomposition.logging.SingleModeRegularisationLogger(0),
        "Regulariser 1": tenkit.decomposition.logging.SingleModeRegularisationLogger(1),
        "Regulariser 2": tenkit.decomposition.logging.SingleModeRegularisationLogger(2),
    }

def run_smooth_experiment(dataset_num, X, rank):
    best_pf2 = None
    bcd_tenkit.INIT_METHOD_A = INIT
    bcd_tenkit.INIT_METHOD_B = INIT

    for init in range(NUM_INITS):
        loggers = create_loggers(dataset_num)
        checkpoint_path = ALL_CHECKPOINTS_FOLDER/f"double_split_init-{init:03d}_dataset-{dataset_num:03d}.h5"
        pf2 = bcd_tenkit.BlockEvolvingTensor(
            rank,
            sub_problems=[
                bcd_tenkit.Mode0ADMM(
                    ridge_penalty=RIDGE_PENALTY,
                    non_negativity=False,
                    max_its=INNER_SUB_ITS,
                    tol=INNER_TOL
                ),
                bcd_tenkit.DoubleSplittingParafac2ADMM(
                    l2_similarity=L2_SIMILARITY,
                    ridge_penalty=RIDGE_B*RIDGE_PENALTY,
                    non_negativity=False,
                    max_it=INNER_SUB_ITS,
                    tol=INNER_TOL
                ),
                bcd_tenkit.Mode2ADMM(
                    ridge_penalty=RIDGE_PENALTY,
                    non_negativity=True,
                    max_its=INNER_SUB_ITS,
                    tol=INNER_TOL
                ),
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
            non_negativity_constraints=[False, False, True],
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

def generate_results(dataset_num, decomposer):
    logger_names = create_loggers(dataset_num).keys()
    logs = [logger.log_metrics for logger in decomposer.loggers]
    results = dict(zip(logger_names, logs))
    results['iteration'] = decomposer.loggers[0].log_iterations
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

    double_pf2 = run_smooth_experiment(dataset_num, noisy_X, RANK)
    store_results(dataset_num, "smooth", double_pf2)

    #pf2 = run_als_experiment(dataset_num, noisy_X, RANK)
    #store_results(dataset_num, "als", pf2)


if __name__ == "__main__":
    np.random.seed(0)
    from joblib import delayed, Parallel
    Parallel(n_jobs=NUM_DATASETS)(delayed(run_experiment)(i) for i in range(NUM_DATASETS))
    #for dataset_num in trange(NUM_DATASETS):
    #    run_experiment(dataset_num)