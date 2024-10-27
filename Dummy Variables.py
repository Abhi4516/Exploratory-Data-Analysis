# Dummy Variables
# OneHotEncoding

import pandas as pd
animal_category = pd.read_csv(r"C:\Users\D\Desktop\New Assignments\Datasets\animal_category.csv")
animal_category.dtypes

# Create dummy variables on categorical columns

animal_category_new = pd.get_dummies(animal_category)
animal_category_new

animal_category.drop(['Index'],axis = 1, inplace=True)
from sklearn.preprocessing import OneHotEncoder
# creating instance of one-hot-encoder
enc = OneHotEncoder(handle_unknown='ignore')

enc_animal_category = pd.DataFrame(enc.fit_transform(animal_category).toarray()) 

enc_animal_category

## LabelEncoding

from sklearn.preprocessing import LabelEncoder
# creating instance of labelencoder
labelencoder = LabelEncoder()
X = animal_category.iloc[:,0:4] 
y = animal_category['Types'] 

animal_category.columns

X['Animals']= labelencoder.fit_transform(X['Animals'])
X['Gender']= labelencoder.fit_transform(X['Gender'])
X['Homly']= labelencoder.fit_transform(X['Homly'])


y = labelencoder.fit_transform(y)
y = pd.DataFrame(y)


# concatenate X and y

animal_category1_new = pd.concat([X,y], axis = 1)
## rename column name
animal_category1_new.columns

animal_category1_new = animal_category1_new.rename(columns={0:'Type1'})
animal_category1_new