# From the simulated output parameters of "patients" with varying input parameters. 
# We randomly select from these "patient"  
# And select a timepoint to simulate the process of clinical sampling. 

import numpy as np
import pandas as pd

n_samples = 10000

def sample_clinical_dataset(df, n_samples, window=50):

    sampled_rows = []

    n_patients = len(df)

    for i in range(n_samples):

        # sample a patient with replacement
        idx = np.random.randint(0, n_patients)
        patient = df.iloc[idx]

        ts = patient["timeseries"]

        # randomly select a timepoint from last "window" timesteps
        tail = ts.tail(window)

        rand_idx = np.random.randint(0, len(tail))
        sample_point = tail.iloc[rand_idx]

        # construct sampled row
        sampled_rows.append({
            "height": patient["height"],
            "weight": patient["weight"],
            "sex": patient["sex"],
            "lt4": patient["lt4"],
            "lt3": patient["lt3"],
            "RTF": patient["RTF"],

            "FT4_sample": sample_point["FT4"],
            "FT3_sample": sample_point["FT3"],
            "TT3_sample": sample_point["TT3"],
            "TSH_sample": sample_point["TSH"],
        })

        # print progress 
        if (i+1) % 1000 == 0:
            print(f"{i+1}/{n_samples} samples generated")

    sampled_df = pd.DataFrame(sampled_rows)

    return sampled_df

df = pd.read_pickle("thyrosim_full_dataset.pkl")
sampled_df = sample_clinical_dataset(df, n_samples)

sampled_df.to_csv("simulation/thyrosim_sample_data.csv", index=False)

# for clearing memory (files too big)
import gc
del df
gc.collect()