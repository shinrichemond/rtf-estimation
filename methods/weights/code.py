# weighted least squares for RTF estimation - March 2026 Megan Lynch


# import libraries
import pandas as pd
import matplotlib.pyplot as plt


# import in data and define weights (weight value specified by Dr. Joe)
simulated_data = pd.read_csv("thyrosim_cut_dataset_v2.csv")
observed_data = pd.read_csv("thyrosim_sample_data.csv")

weights = {
"FT4": 0.45,
"FT3": 0.10,
"TSH": 0.45
} 

# precompute the available lt4 values in the simulation grid for nearest-neighbor lookup
sim_lt4_values = simulated_data["lt4"].unique()
 
def nearest_lt4(observed_lt4):
        """Return the closest lt4 value available in the simulation grid."""
        return min(sim_lt4_values, key=lambda x: abs(x - observed_lt4))


results = pd.DataFrame(columns=["RTF", "Weighted Error"]) # create dataframe to save estimated RTF and its weighted error for each patient

# define function to estimate RTF for each patient
def estimate_RTF(obs_patient_row):
        global results # use global variable to save results for each patient

        # take one patient and get their observed outputs (FT4, FT3, TSH)
        obs_output = {
                "FT4": observed_data.loc[obs_patient_row, "FT4_sample"], 
                "FT3": observed_data.loc[obs_patient_row, "FT3_sample"],
                "TSH": observed_data.loc[obs_patient_row, "TSH_sample"]
        }

        # find the nearest lt4 value in the simulation grid for this patient
        matched_lt4 = nearest_lt4(observed_data.loc[obs_patient_row, "lt4"])
 
        # create a dataframe to save the simulated outputs
        sim = pd.DataFrame(columns = ["FT4", "FT3", "TSH"])  

        # if the patient's data matches the simulation data (match height, weight, sex, lt4, lt3) add the simulated output and RTF value to dataframe sim (keep that simulated row)
        sim = simulated_data[
                (simulated_data["height"] == observed_data.loc[obs_patient_row, "height"]) &
                (simulated_data["weight"] == observed_data.loc[obs_patient_row, "weight"]) &
                (simulated_data["sex"] == observed_data.loc[obs_patient_row, "sex"]) &
                 (simulated_data["lt4"] == matched_lt4) &
                (simulated_data["lt3"] == observed_data.loc[obs_patient_row, "lt3"])][["RTF", "FT4_mean", "FT3_mean", "TSH_mean"]].reset_index(drop=True)
        
        # compute the weighted error for each RTF estimate in the simulation data compared to the observed data for that patient
        rtf_and_error = pd.DataFrame(columns=["RTF", "Weighted Error"]) # create dataframe to save estimated RTF and its weighted error
        for i in range(len(sim)): # calculate weighted error for RTF estimate for that patient (loop through each row of the sim dataframe)
                error = (
                weights["FT4"] * (obs_output["FT4"] - sim.loc[i, "FT4_mean"])**2 + 
                weights["FT3"] * (obs_output["FT3"] - sim.loc[i, "FT3_mean"])**2 + 
                weights["TSH"] * (obs_output["TSH"] - sim.loc[i, "TSH_mean"])**2 )

                #add the RTF and its error to a dataframe
                rtf_and_error.loc[len(rtf_and_error)] = {
                        "RTF": sim.loc[i, "RTF"], 
                        "Weighted Error": error
                        }
        
        # the best RTF estimate is the one with the least weighted error 
        best_RTF = rtf_and_error.loc[rtf_and_error["Weighted Error"].idxmin()] # find minimum weighted error
        # save the best RTF and its error to the results dataframe
        results.loc[len(results)] = {
                "RTF": best_RTF["RTF"],
                "Weighted Error": best_RTF["Weighted Error"]
                }
        

# loop through each patient in the observed data and estimate their RTF
for i in range(len(observed_data)): # loop through each patient 
        estimate_RTF(i) # estimate RTF for each patient

results.to_csv("RTF_estimates_and_errors.csv", index=False) # save results to csv file


# evaluation

# compare the estimate to the true RTF value (found in observed data)... plot estimate vs true RTF
plt.scatter(observed_data["RTF"], results["RTF"], marker="o")
plt.xlabel("True RTF")
plt.ylabel("Estimated RTF")
plt.title("Comparison of True and Estimated RTF Values")
plt.savefig("estimate vs true RTF.pdf")
plt.close()