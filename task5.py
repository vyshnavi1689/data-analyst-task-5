#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

#load the dataset 
train_data=pd.read_csv("train.csv")
test_data=pd.read_csv("test.csv")
#describing data
print(train_data.head())
print(train_data.describe())
print(train_data.info())
print(train_data.value_counts())

df_numeric=train_data.select_dtypes(include=[float,int])
print(df_numeric.corr)

#using heat map and pair plots for plotting
#plt.figure(figsize=(10,5))
#sns.heatmap(df_numeric.corr(),annot=True,cmap='coolwarm')
#plt.title('correlation heatmap')
#plt.show()

sns.pairplot(train_data)
plt.title('sample pairplot')
plt.show()



