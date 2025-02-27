# Duplication & Typecasting
## Typecasting

import pandas as pd
import numpy as np

OnlineRetail = pd.read_csv(r"C:\Users\D\Desktop\New Assignments  Keys\Datasets\OnlineRetail.csv",encoding='unicode_escape')
print(OnlineRetail.head())


OnlineRetail.dtypes

#type casting

OnlineRetail['UnitPrice'] = OnlineRetail['UnitPrice'].astype('int64') 
OnlineRetail['UnitPrice'].dtypes

"Error while converting CustomerID(float64) to (int64)"
"Cannot convert non-finite values (NA or inf) to integer"
"1ts we have to fill NA values"

from sklearn.impute import SimpleImputer

# Mean Imputer 
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
OnlineRetail["CustomerID"] = pd.DataFrame(mean_imputer.fit_transform(OnlineRetail[["CustomerID"]]))
OnlineRetail["CustomerID"].isnull().sum()  

# Now we will convert 'float64' into 'int64' type. 
OnlineRetail['CustomerID'] = OnlineRetail['CustomerID'].astype('int64') 
OnlineRetail['CustomerID'].dtypes

## Duplication

#Identify duplicates records in the data
duplicate = OnlineRetail.duplicated()
sum(duplicate) 

#Removing Duplicates
data1 = OnlineRetail.drop_duplicates() 


#Graphical Representation
import matplotlib.pyplot as plt 
import numpy as np


#Quantity
plt.hist(OnlineRetail.Quantity);plt.show() #histogram

plt.boxplot(OnlineRetail.Quantity);plt.show() #boxplot

#UnitPrice
plt.hist(OnlineRetail.UnitPrice);plt.show() #histogram

plt.boxplot(OnlineRetail.UnitPrice);plt.show() #boxplot

#CustomerID
plt.hist(OnlineRetail.CustomerID);plt.show() #histogram

plt.boxplot(OnlineRetail.CustomerID);plt.show() #boxplot

