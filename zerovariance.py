# Zero Variance


import pandas as pd


Z_data = pd.read_csv(r"C:\Users\D\Desktop\New Assignments\Datasets\Z_dataset.csv")
Z_data

Z_data.dtypes

print(Z_data['square.length'].var())
print(Z_data['square.breadth'].var())
print(Z_data['rec.Length'].var())
print(Z_data['rec.breadth'].var())

