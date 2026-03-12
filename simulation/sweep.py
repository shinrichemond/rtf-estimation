import numpy as np
import pandas as pd
from thyrosim_model import Thyrosim, run_simulation, compute_ft4_ft3


def create_default_params():
    params = {f"p{i}": 1.0 for i in range(1,49)}
    return params


def simulate_rtf(rtf):

    params = create_default_params()
    params["p1"] = rtf

    model = Thyrosim(
        dial=[1,1,1,1],
        inf=[0,0],
        kdelay=0.1,
        params=params
    )

    initial_conditions = np.ones(19)

    t, states = run_simulation(model, 0, 200, initial_conditions)

    ft4, ft3 = compute_ft4_ft3(states, params)
    tsh = states[6]

    # average last part of trajectory (approx steady state)
    ft4_ss = np.mean(ft4[-20:])
    ft3_ss = np.mean(ft3[-20:])
    tsh_ss = np.mean(tsh[-20:])

    return ft4_ss, ft3_ss, tsh_ss


def generate_big_dataset(
        rtf_min=0.0,
        rtf_max=1.0,
        n_points=500,
        output_file="rtf_simulation_table.csv"
    ):

    rtf_values = np.linspace(rtf_min, rtf_max, n_points)

    rows = []

    for rtf in rtf_values:

        ft4, ft3, tsh = simulate_rtf(rtf)

        rows.append({
            "RTF": rtf,
            "FT4": ft4,
            "FT3": ft3,
            "TSH": tsh
        })

        print(f"Simulated RTF={rtf:.4f}")

    df = pd.DataFrame(rows)

    df.to_csv(output_file, index=False)

    print("\nSaved dataset to:", output_file)
    print("Rows:", len(df))

    return df


if __name__ == "__main__":

    dataset = generate_big_dataset(
        rtf_min=0.0,
        rtf_max=1.0,
        n_points=1000,  # large table
        output_file="rtf_lookup_table.csv"
    )

    print(dataset.head())