import pandas as pd
import numpy as np

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression


###########################
# helper function

# define function to calculate NRMSE
def nrmse(ori, imp):
    arr1 = ori.to_numpy()
    arr2 = imp.to_numpy()

    # Calculate RMSE
    mse = np.mean((arr1 - arr2) ** 2)
    nrmse = np.sqrt(mse / (np.max(arr1) - np.min(arr1)))

    return nrmse

# read in complete data
complete = pd.read_csv("~/Documents/Imputation/Data/completeExpr.csv", index_col=0)
complete = complete.transpose()

# reset colnames
complete.columns = ['p' + str(i) for i in range(1,len(complete.columns)+1)]


###########################

### Cutoff 10%

print("Read in data...")

# read in data
cutoff1 = pd.read_csv("~/Documents/Imputation/Data/cutoff0.1.csv", index_col=0)
cutoff1.columns = ['p' + str(i) for i in range(1,len(cutoff1.columns)+1)]

# convert string 'NA' to np.nan
cutoff1 = cutoff1.replace("NA", np.nan)

print("Cutoff 10% data ready")


print("Start MICE imputation...")

# Create an instance of IterativeImputer
imputer = IterativeImputer(estimator=LinearRegression(), max_iter=1, random_state=0)

# Perform MICE imputation
mice_imputed = imputer.fit_transform(cutoff1)

# Convert back to DataFrame
mice_imputed_df = pd.DataFrame(mice_imputed, columns=cutoff1.columns)

print("MICE imputation for cutoff 10% done")
print(mice_imputed_df.head())

print(f"NRMSE for cutoff 10% = {nrmse(complete, mice_imputed_df)}")

# save imputed data
mice_imputed_df.to_csv("~/Documents/Imputation/Data/mice_imputed_cutoff0.1.csv")


###########################

### Cutoff 20%

print("Read in data...")

# read in data
cutoff2 = pd.read_csv("~/Documents/Imputation/Data/cutoff0.2.csv", index_col=0)
cutoff2.columns = ['p' + str(i) for i in range(1,len(cutoff2.columns)+1)]

# convert string 'NA' to np.nan
cutoff2 = cutoff2.replace("NA", np.nan)

print("Cutoff 20% data ready")


print("Start MICE imputation...")

# Create an instance of IterativeImputer
imputer = IterativeImputer(estimator=LinearRegression(), max_iter=1, random_state=0)

# Perform MICE imputation
mice_imputed = imputer.fit_transform(cutoff2)

# Convert back to DataFrame
mice_imputed_df = pd.DataFrame(mice_imputed, columns=cutoff2.columns)

print("MICE imputation for cutoff 20% done")
print(mice_imputed_df.head())

print(f"NRMSE for cutoff 20% = {nrmse(complete, mice_imputed_df)}")

# save imputed data
mice_imputed_df.to_csv("~/Documents/Imputation/Data/mice_imputed_cutoff0.2.csv")



###########################

### Cutoff 30%

print("Read in data...")

# read in data
cutoff3 = pd.read_csv("~/Documents/Imputation/Data/cutoff0.3.csv", index_col=0)
cutoff3.columns = ['p' + str(i) for i in range(1,len(cutoff3.columns)+1)]

# convert string 'NA' to np.nan
cutoff3 = cutoff3.replace("NA", np.nan)

print("Cutoff 30% data ready")


print("Start MICE imputation...")

# Create an instance of IterativeImputer
imputer = IterativeImputer(estimator=LinearRegression(), max_iter=1, random_state=0)

# Perform MICE imputation
mice_imputed = imputer.fit_transform(cutoff3)

# Convert back to DataFrame
mice_imputed_df = pd.DataFrame(mice_imputed, columns=cutoff3.columns)

print("MICE imputation for cutoff 30% done")
print(mice_imputed_df.head())

print(f"NRMSE for cutoff 30% = {nrmse(complete, mice_imputed_df)}")

# save imputed data
mice_imputed_df.to_csv("~/Documents/Imputation/Data/mice_imputed_cutoff0.3.csv")










