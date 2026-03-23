import pandas as pd

SAMPLE_PATH = "thyrosim_sample_data.csv"
GRID_PATH = "thyrosim_cut_dataset_v2.csv"
OUTPUT_PATH = "grid_search_results.csv"

MATCH_COLUMNS = ["sex", "height", "weight", "lt4", "lt3"]
NUMERIC_MATCH_COLUMNS = ["height", "weight", "lt4", "lt3"]
LAB_SAMPLE_COLUMNS = ["ft4_sample", "ft3_sample", "tt3_sample", "tsh_sample"]
LAB_GRID_COLUMNS = ["ft4_mean", "ft3_mean", "tt3_mean", "tsh_mean"]


def load_data():
    sample_df = pd.read_csv(SAMPLE_PATH)
    grid_df = pd.read_csv(GRID_PATH)

    sample_df.columns = sample_df.columns.str.strip().str.lower()
    grid_df.columns = grid_df.columns.str.strip().str.lower()

    return sample_df, grid_df


def build_group_lookup(grid_df):
    return {
        key: group.reset_index(drop=True)
        for key, group in grid_df.groupby(MATCH_COLUMNS, sort=False)
    }


def build_combo_df(grid_df):
    return grid_df[MATCH_COLUMNS].drop_duplicates().reset_index(drop=True)


def get_candidate_rows(patient, combo_df, group_lookup):
    candidate_combos = combo_df

    sex_matches = candidate_combos["sex"] == patient["sex"]
    if sex_matches.any():
        candidate_combos = candidate_combos.loc[sex_matches]

    for column in NUMERIC_MATCH_COLUMNS:
        distances = (candidate_combos[column] - patient[column]).abs()
        candidate_combos = candidate_combos.loc[distances == distances.min()]

    candidate_groups = []
    for _, combo in candidate_combos.iterrows():
        key = tuple(combo[column] for column in MATCH_COLUMNS)
        candidate_groups.append(group_lookup[key])

    return pd.concat(candidate_groups, ignore_index=True)


def add_squared_error(candidate_rows, patient):
    error_df = candidate_rows.copy()
    error_df["squared_error"] = (
        (error_df["ft4_mean"] - patient["ft4_sample"]) ** 2
        + (error_df["ft3_mean"] - patient["ft3_sample"]) ** 2
        + (error_df["tt3_mean"] - patient["tt3_sample"]) ** 2
        + (error_df["tsh_mean"] - patient["tsh_sample"]) ** 2
    )
    return error_df


def estimate_patient(patient, combo_df, group_lookup):
    candidate_rows = get_candidate_rows(patient, combo_df, group_lookup)
    scored_candidates = add_squared_error(candidate_rows, patient)
    best_match = scored_candidates.loc[scored_candidates["squared_error"].idxmin()]
    return best_match


def check_first_patient_exact_match(sample_df, grid_df):
    first_patient = sample_df.iloc[0]
    exact_mask = pd.Series(True, index=grid_df.index)

    for column in MATCH_COLUMNS:
        exact_mask &= grid_df[column] == first_patient[column]

    exact_candidates = grid_df.loc[exact_mask].reset_index(drop=True)

    print("FIRST PATIENT EXACT MATCH CHECK")
    print(f"Exact candidate count: {len(exact_candidates)}")
    print(first_patient[MATCH_COLUMNS].to_dict())
    print()


def build_results(sample_df, combo_df, group_lookup):
    results = []

    for patient_index, patient in sample_df.iterrows():
        best_match = estimate_patient(patient, combo_df, group_lookup)
        results.append(
            {
                "patient_index": patient_index,
                "true_rtf": patient["rtf"],
                "rtf_hat": best_match["rtf"],
                "rtf_error": best_match["rtf"] - patient["rtf"],
                "abs_rtf_error": abs(best_match["rtf"] - patient["rtf"]),
                "best_squared_error": best_match["squared_error"],
                "matched_sex": best_match["sex"],
                "matched_height": best_match["height"],
                "matched_weight": best_match["weight"],
                "matched_lt4": best_match["lt4"],
                "matched_lt3": best_match["lt3"],
                "ft4_sample": patient["ft4_sample"],
                "ft3_sample": patient["ft3_sample"],
                "tt3_sample": patient["tt3_sample"],
                "tsh_sample": patient["tsh_sample"],
                "ft4_mean": best_match["ft4_mean"],
                "ft3_mean": best_match["ft3_mean"],
                "tt3_mean": best_match["tt3_mean"],
                "tsh_mean": best_match["tsh_mean"],
            }
        )

    return pd.DataFrame(results)


def main():
    sample_df, grid_df = load_data()
    group_lookup = build_group_lookup(grid_df)
    combo_df = build_combo_df(grid_df)

    check_first_patient_exact_match(sample_df, grid_df)

    first_patient = sample_df.iloc[0]
    first_patient_best_match = estimate_patient(first_patient, combo_df, group_lookup)
    print("FIRST PATIENT BEST MATCH")
    print(first_patient_best_match[["sex", "height", "weight", "lt4", "lt3", "rtf", "squared_error"]].to_dict())
    print()

    results_df = build_results(sample_df, combo_df, group_lookup)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {OUTPUT_PATH} with {len(results_df)} rows.")
    print(f"Mean absolute RTF error: {results_df['abs_rtf_error'].mean():.6f}")
    print(f"Median absolute RTF error: {results_df['abs_rtf_error'].median():.6f}")


if __name__ == "__main__":
    main()
