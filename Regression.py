#importing importnat librabies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#importing the file and extracting the independent and dependent values
car_details = pd.read_csv("Linear_regression\car_details.csv")

x = car_details.select_dtypes(include=['number']).drop(columns=['selling_price']).values
y = car_details['selling_price'].values

#print(car_details.head())
#print(car_details.dtypes)

numeric_df = car_details.select_dtypes(include=['number'])
#sns.heatmap(numeric_df.corr(), annot=True)
#plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
regression = LinearRegression()
regression.fit(x_train, y_train)
plt.show()