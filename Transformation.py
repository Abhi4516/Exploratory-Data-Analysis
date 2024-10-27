# Transformation

import pandas as pd
import matplotlib.pyplot as plt

calories_consumed = pd.read_csv(r"C:\Users\D\Desktop\New Assignments\Datasets\calories_consumed.csv")
calories_consumed

#Normal Quantile-Quantile Plot
import scipy.stats as stats
import pylab

# Checking Whether data is normally distributed

stats.probplot(calories_consumed['Weight gained (grams)'], dist='norm',plot=pylab);plt.show() #pylab is visual representation

stats.probplot(calories_consumed['Calories Consumed'], dist='norm',plot=pylab);plt.show() # it is normally distributed

##### So the data is not normal we have to transform it to normal

#transformation to make Weight .gained..grams variable normal
import numpy as np
stats.probplot(np.log(calories_consumed['Weight gained (grams)']),dist="norm",plot=pylab) #best transformation

calories_consumed.describe()



