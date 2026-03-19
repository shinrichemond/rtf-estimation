# Multiprocessing the process of sweeping across varying values input values
# Inputs: Sex, Height, Weight, LT3, LT4, RTF
# Outputs: FT4, FT3, TT3, TSH
import numpy as np
import pandas as pd
from itertools import product
from concurrent.futures import ProcessPoolExecutor
from thyrosim_model import simulate_patient
from tqdm import tqdm
import os

def load_completed_params(csv_file):
    df = pd.read_csv(csv_file)

    # float fix
    df["RTF"]= df["RTF"].round(5)

    return set(zip(
        df["height"],
        df["weight"],
        df["sex"],
        df["lt4"],
        df["lt3"],
        df["RTF"]
    ))

def get_missing_param_grid(param_grid, completed_set):
    for params in param_grid:
        if params not in completed_set:
            yield params

def run_single_simulation(args):
    h, w, s, lt4, lt3, rtf = args

    df = simulate_patient(
        height=h,
        weight=w,
        sex=s,
        lt4_dose=lt4,
        lt3_dose=lt3,
        rtf=rtf
    )

    # means
    ft4 = df["FT4"].values[-50:]
    ft3 = df["FT3"].values[-50:]
    tt3 = df["TT3"].values[-50:]
    tsh = df["TSH"].values[-50:]

    # float fix
    rtf = round(rtf, 5)

    return (
        h, w, s, lt4, lt3, rtf, # df,
        ft4.mean(), ft3.mean(), tt3.mean(), tsh.mean()
    )

def generate_full_dataset_parallel():
    # Full sweep
    heights = range(150, 185, 5)
    weights = range(50, 75, 5)
    sexes = ["male", "female"]
    lt4_vals = range(25, 55, 5)
    lt3_vals = range(5, 10)
    rtf_vals = np.linspace(0.0, 1.0, 101)

    # test sweep
    # heights = [180]
    # weights = [70]
    # sexes = ["female"]
    # lt4_vals = range(30, 55, 5)
    # lt3_vals = range(5, 10)
    # rtf_vals = np.linspace(0.0, 1.0, 101)

    param_grid_full = product(
        heights, weights, sexes, lt4_vals, lt3_vals, rtf_vals
    )

    output_file = "simulation/thyrosim_cut_dataset_v2.csv"

    completed = set()
    if os.path.exists(output_file):
        print("Resuming from existing file...")
        completed = load_completed_params(output_file)

    param_grid = get_missing_param_grid(param_grid_full, completed)

    total = len(heights)*len(weights)*len(sexes)*len(lt4_vals)*len(lt3_vals)*len(rtf_vals)
    remaining = total - len(completed)

    print(f"Total simulations: {total}")
    print(f"Already completed: {len(completed)}")
    print(f"Remaining: {remaining}")

    columnlist = [
        "height", "weight", "sex", "lt4", "lt3", "RTF", # "timeseries",
        "FT4_mean", "FT3_mean", "TT3_mean", "TSH_mean"
    ]

    if not os.path.exists(output_file):
        pd.DataFrame(columns=columnlist).to_csv(output_file, index=False)
    batch_size = 10000
    buffer = []

    with ProcessPoolExecutor(max_workers=os.cpu_count()-1) as executor:
        for res in tqdm(executor.map(run_single_simulation, param_grid, chunksize=20), total=remaining):
            buffer.append(res)

            # flush periodically
            if len(buffer) >= batch_size:
                pd.DataFrame(buffer, columns=columnlist).to_csv(
                    output_file, mode="a", header=False, index=False
                )
                buffer.clear()

    return output_file

if __name__ == "__main__":
    generate_full_dataset_parallel()

