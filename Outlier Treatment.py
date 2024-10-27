# Outlier Treatment 


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv(r"C:\Users\D\Desktop\New Assignments\Datasets\boston_data.csv")
df

df.dtypes 
df.isna().sum()


sns.boxplot(df.crim);plt.title('Boxplot');plt.show()
sns.boxplot(df.zn);plt.title('Boxplot');plt.show()
sns.boxplot(df.indus);plt.title('Boxplot');plt.show() 
sns.boxplot(df.chas);plt.title('Boxplot');plt.show()
sns.boxplot(df.nox);plt.title('Boxplot');plt.show() 
sns.boxplot(df.rm);plt.title('Boxplot');plt.show()
sns.boxplot(df.age);plt.title('Boxplot');plt.show() 
sns.boxplot(df.dis);plt.title('Boxplot');plt.show()
sns.boxplot(df.rad);plt.title('Boxplot');plt.show() 
sns.boxplot(df.tax);plt.title('Boxplot');plt.show()  
sns.boxplot(df.ptratio);plt.title('Boxplot');plt.show() 
sns.boxplot(df.black);plt.title('Boxplot');plt.show()
sns.boxplot(df.lstat);plt.title('Boxplot');plt.show()
sns.boxplot(df.medv);plt.title('Boxplot');plt.show()

# we have only 7 variables which has outliers
# 1) crim 2) zn 3) rm 4) dis 5) black 6) lstat 7) medv

#1) crim
# Detection of Outliers
IQR = df['crim'].quantile(0.75) - df['crim'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = df['crim'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = df['crim'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR


# Trimming Technique
# let's flag the outliers in the data set
outliers_df = np.where(df['crim'] > upper_limit,True,np.where(df['crim'] < lower_limit,True,False))
# if value is greater than upper limit consider it as outliers and if the value is less than lower limit consider it as outliers
df_trimmed = df.loc[~(outliers_df),] # ~ means not - it shows all false value (not outliers)
df.shape, df_trimmed.shape # we trim 12 outliers

sns.boxplot(df_trimmed.crim);plt.title('Boxplot');plt.show()

#we still have outiers

# 2.Replace 
# Now let's replace the outliers by the maximum and minimum limit
df['df_replaced']= pd.DataFrame(np.where(df['crim'] > upper_limit, upper_limit,
                                         np.where(df['crim'] < lower_limit, lower_limit,
                                                  df['crim'])))
                                 
sns.boxplot(df.df_replaced);plt.title('Boxplot');plt.show()



"2) zn"

# Detection of Outliers
IQR = df['zn'].quantile(0.75) - df['zn'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = df['zn'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = df['zn'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR

#1. Remove 
# Trimming Technique

outliers_df = np.where(df['zn'] > upper_limit,True,np.where(df['zn'] < lower_limit,True,False))
# if value is greater than upper limit consider it as outliers and if the value is less than lower limit consider it as outliers
df_trimmed = df.loc[~(outliers_df),] # ~ means not - it shows all false value (not outliers)
df.shape, df_trimmed.shape # we trim 49 outliers

sns.boxplot(df_trimmed.zn);plt.title('Boxplot');plt.show()

# 2.Replace
# Now let's replace the outliers by the maximum and minimum limit
df['df1_replaced']= pd.DataFrame(np.where(df['zn'] > upper_limit, upper_limit,
                                         np.where(df['zn'] < lower_limit, lower_limit,
                                                  df['zn'])))
                                 
sns.boxplot(df.df1_replaced);plt.title('Boxplot');plt.show()



"3) rm"

# Detection of Outliers
IQR = df['rm'].quantile(0.75) - df['rm'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = df['rm'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = df['rm'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR

#Replace 
# Now let's replace the outliers by the maximum and minimum limit
df['df2_replaced']= pd.DataFrame(np.where(df['rm'] > upper_limit, upper_limit,
                                         np.where(df['rm'] < lower_limit, lower_limit,
                                                  df['rm'])))
                                 
sns.boxplot(df.df2_replaced);plt.title('Boxplot');plt.show()



"4) dis"

# Detection of Outliers
IQR = df['dis'].quantile(0.75) - df['dis'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = df['dis'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = df['dis'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR

# 3. Winsorization 

# import feature_engine.outliers.winsorizer
from feature_engine.outliers.winsorizer import Winsorizer
winsorizer = Winsorizer(capping_method ='iqr', # choose skewed for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5, # 1.5 times of iqr
                          variables=['dis'])
# capping_methods = 'iqr' - 25th quantile & 75th quantile
df_t = winsorizer.fit_transform(df[['dis']])

# we can inspect the minimum caps and maximum caps 
winsorizer.left_tail_caps_, winsorizer.right_tail_caps_


sns.boxplot(df_t.dis);plt.title('Boxplot');plt.show()

"5) black"

# Detection of Outliers
IQR = df['black'].quantile(0.75) - df['black'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = df['black'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = df['black'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR

# import feature_engine.outliers.winsorizer
from feature_engine.outliers.winsorizer import Winsorizer
winsorizer = Winsorizer(capping_method ='iqr', # choose skewed for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5, # 1.5 times of iqr
                          variables=['black'])
# capping_methods = 'iqr' - 25th quantile & 75th quantile
df_t = winsorizer.fit_transform(df[['black']])

# we can inspect the minimum caps and maximum caps 
winsorizer.left_tail_caps_, winsorizer.right_tail_caps_


sns.boxplot(df_t.black);plt.title('Boxplot');plt.show()

"6) lstat"

# Detection of Outliers
IQR = df['lstat'].quantile(0.75) - df['lstat'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = df['lstat'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = df['lstat'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR

# import feature_engine.outliers.winsorizer
from feature_engine.outliers.winsorizer import Winsorizer
winsorizer = Winsorizer(capping_method ='iqr', # choose skewed for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5, # 1.5 times of iqr
                          variables=['lstat'])
# capping_methods = 'iqr' - 25th quantile & 75th quantile
df_t = winsorizer.fit_transform(df[['lstat']])

# we can inspect the minimum caps and maximum caps 
winsorizer.left_tail_caps_, winsorizer.right_tail_caps_


sns.boxplot(df_t.lstat);plt.title('Boxplot');plt.show()

"7) medv"

# Detection of Outliers
IQR = df['medv'].quantile(0.75) - df['medv'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = df['medv'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = df['medv'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR

# import feature_engine.outliers.winsorizer
from feature_engine.outliers.winsorizer import Winsorizer
winsorizer = Winsorizer(capping_method ='iqr', # choose skewed for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5, # 1.5 times of iqr
                          variables=['medv'])
# capping_methods = 'iqr' - 25th quantile & 75th quantile
df_t = winsorizer.fit_transform(df[['medv']])

# we can inspect the minimum caps and maximum caps 
winsorizer.left_tail_caps_, winsorizer.right_tail_caps_


sns.boxplot(df_t.medv);plt.title('Boxplot');plt.show()



