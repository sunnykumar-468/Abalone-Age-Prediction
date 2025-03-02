# Abalone-Age-Prediction
This project helps to predict the age of a person based on certain parameters
# Some Important Libraries are Imported 
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split 

# Uploading of Dataset
df = pd.read_csv('/content/abalone.csv')

# To Read the Dataset
df.head()

# One-Hot Encoding of Categorical Variables
df= pd.get_dummies(df,dtype = "int").astype(int)

# To Retrieve Information About the Dataset
df.info()

#Remove mulitcollinearity
df.head()
df.drop(columns="Sex_I", inplace= True)   
columns = list(df.columns)
columns.remove("Rings")
X= columns
y = "Rings"
df_X = df[X]
df_y = df[y]   

#Sclaer.fit(test)
from sklearn.preprocessing import StandardScaler

Scaler = StandardScaler()
Scaler.fit_transform(df[X])

# Division of Dataset into Training and Testing
train_X , test_X , train_y, test_y = train_test_split( df_X, df_y , test_size = 0.2 , random_state = 123)

#KNeighborsRegressor is imported of Regression Tasks
from sklearn.neighbors import KNeighborsRegressor

model = KNeighborsRegressor(n_neighbors=11)

model.fit(X=train_X, y=train_y)

model.score(test_X,test_y)

# Matplot Library is Imported for Data Visualisation
import matplotlib.pyplot as plt
scores= []
for i in range(1, 20):
    model = KNeighborsRegressor(n_neighbors=i)
    model.fit(X=train_X, y=train_y)
    score = model.score(test_X, test_y)
    scores.append(score)


plt.plot(range(1, 20), scores)
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Score')
plt.title('KNN Regression Performance')
plt.show()

scores= []
for i in range(1, 100):
    model = KNeighborsRegressor(n_neighbors=i)
    model.fit(X=train_X, y=train_y)
    score = model.score(test_X, test_y)
    scores.append(score)


plt.plot(range(1, 100), scores)
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Score')
plt.title('KNN Regression Performance')
plt.show()
