#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

#load the dataset 
train_data=pd.read_csv("train.csv")
test_data=pd.read_csv("test.csv")

print(train_data.head())

#univarant
#plotting histogrm
train_data['Survived'].hist(bins=8,color='green',edgecolor='blue')
plt.title('number of passengers survived')
plt.xlabel('survived')
plt.ylabel('number of passengers')
plt.show()

#bivariant
#plotting bar plots using seaborn
sns.barplot(x='Pclass',y='Survived',data=train_data)
plt.title('number of passengers survived by class')
plt.show()