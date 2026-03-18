# Multiprocessing the process of sweeping across varying values input values
# Inputs: Sex, Height, Weight, LT3, LT4, RTF
# Outputs: FT4, FT3, TT3, TSH
import numpy as np
import pandas as pd
from itertools import product
from concurrent.futures import ProcessPoolExecutor
from thyrosim_model import simulate_patient
from tqdm import tqdm

def compute_means(df, window=50):
    tail = df.tail(window)

    return {
        "FT4_mean": tail["FT4"].mean(),
        "FT3_mean": tail["FT3"].mean(),
        "TT3_mean": tail["TT3"].mean(),
        "TSH_mean": tail["TSH"].mean(),
    }

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

    means = compute_means(df)

    return {
        "height": h,
        "weight": w,
        "sex": s,
        "lt4": lt4,
        "lt3": lt3,
        "RTF": rtf,
        "timeseries": df,
        **means
    }

def generate_full_dataset_parallel():
    # Full sweep
    heights = list(range(150, 185, 5))
    weights = list(range(50, 75, 5))
    sexes = ["male", "female"]
    lt4_vals = list(range(25, 55, 5))
    lt3_vals = list(range(5, 10))
    rtf_vals = np.linspace(0.0, 1.0, 101)

    # test sweep
    # heights = [150, 180]
    # weights = [50, 70]
    # sexes = ["male", "female"]
    # lt4_vals = [25, 50]
    # lt3_vals = [5, 10]
    # rtf_vals = np.linspace(0.0, 1.0, 2)

    param_grid = list(product(
        heights, weights, sexes, lt4_vals, lt3_vals, rtf_vals
    ))

    total = len(param_grid)
    print(f"Total simulations: {total}")

    results = []

    with ProcessPoolExecutor() as executor:

        for res in tqdm(executor.map(run_single_simulation, param_grid), total=total):
            results.append(res)

    dataset = pd.DataFrame(results)

    return dataset

if __name__ == "__main__":
    dataset = generate_full_dataset_parallel()
    dataset.to_pickle("thyrosim_full_dataset.pkl")