# Missing Values Imputation

import numpy as np
import pandas as pd


claimants_data = pd.read_csv(r"C:\Users\D\Desktop\New Assignments\Datasets\claimants.csv")
claimants_data


claimants_data.isna().sum()


# for Mean,Meadian,Mode imputation we can use Simple Imputer or df.fillna()
from sklearn.impute import SimpleImputer

# Mean Imputer 
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
claimants_data["CLMAGE"] = pd.DataFrame(mean_imputer.fit_transform(claimants_data[["CLMAGE"]]))
claimants_data["CLMAGE"].isnull().sum() 

# Median Imputer
median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
claimants_data["CLMAGE"] = pd.DataFrame(median_imputer.fit_transform(claimants_data[['CLMAGE']]))
claimants_data["CLMAGE"].isnull().sum() 

# Mode Imputer
mode_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
claimants_data["CLMSEX"] = pd.DataFrame(mode_imputer.fit_transform(claimants_data[['CLMSEX']]))
claimants_data["CLMINSUR"] = pd.DataFrame(mode_imputer.fit_transform(claimants_data[['CLMINSUR']]))
claimants_data["SEATBELT"] = pd.DataFrame(mode_imputer.fit_transform(claimants_data[['SEATBELT']]))
claimants_data.isnull().sum() 



