# Standardization is another scaling technique where the values are centered around the mean with a unit standard deviation.
# This means that the mean of the attribute becomes zero and the resultant distribution has a unit standard deviation.



import pandas as pd

seed = pd.read_csv("Seeds_data.csv")
seed


#Standardization
# Normalization function using z std. all are continuous data.
def norm_func(i):
    x = (i-i.mean())/(i.std()) 
    return(x)

seed1_norm = norm_func(seed)
seed1_norm.describe() # mean=0, std = 1

#Normalization
# or denominator (i.max()-i.min())
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min()) # or denominator (i.max()-i.min())
    return(x)

seed2_norm = norm_func(seed)
seed2_norm.describe() # min=0, max=1



