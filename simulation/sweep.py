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

    ft4 = df["FT4"].values[-50:]
    ft3 = df["FT3"].values[-50:]
    tt3 = df["TT3"].values[-50:]
    tsh = df["TSH"].values[-50:]

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
    rtf_vals = np.linspace(0.0, 1.0, 4)

    # test sweep
    # heights = [150, 180]
    # weights = [50, 70]
    # sexes = ["male", "female"]
    # lt4_vals = [25, 50]
    # lt3_vals = [5, 10]
    # rtf_vals = np.linspace(0.0, 1.0, 2)

    param_grid = product(
        heights, weights, sexes, lt4_vals, lt3_vals, rtf_vals
    )

    total = len(heights) * len(weights) * len(sexes) * len(lt4_vals) * len(lt3_vals) * len(rtf_vals)
    print(f"Total simulations: {total}")

    results = []

    columnlist = [
        "height", "weight", "sex", "lt4", "lt3", "RTF", # "timeseries",
        "FT4_mean", "FT3_mean", "TT3_mean", "TSH_mean"
    ]

    output_file = "simulation/thyrosim_staging.csv"
    pd.DataFrame(columns=columnlist).to_csv(output_file, index=False)
    batch_size = 10
    buffer = []

    with ProcessPoolExecutor(max_workers=os.cpu_count()-1) as executor:
        for res in tqdm(executor.map(run_single_simulation, param_grid, chunksize=1), total=total):
            results.append(res)

            buffer.append(res)

            # flush periodically
            if len(buffer) >= batch_size:
                pd.DataFrame(buffer, columns=columnlist).to_csv(
                    output_file, mode="a", header=False, index=False
                )
                buffer.clear()

    return output_file

if __name__ == "__main__":
    dataset = generate_full_dataset_parallel()
    dataset.to_csv("simulation/thyrosim_cut_dataset.csv", index=False)
