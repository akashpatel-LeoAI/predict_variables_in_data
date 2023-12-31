# -*- coding: utf-8 -*-
"""Amazon_Stock_prediction_model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12a9VSFgW7xt_Pl3x9hXRimOVGcuPRsDB
"""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from google.colab import files

uploaded = files.upload()

from os import rename
amzn_df = pd.read_csv('AMZN_1Y_Historical_Data.csv')

amzn_df = amzn_df.rename(columns={'Close/Last' : 'Close_price'})

columns_to_process = ['Close_price','Open','High','Low']

for col in columns_to_process:
  amzn_df[col] = amzn_df[col].str.replace('$','').astype(float)

amzn_df.head(5)

from scipy.sparse import random
df_close_price = amzn_df[['Close_price']]

future_days = 15
df_close_price['Prediction'] = df_close_price[['Close_price']].shift(-future_days)
X = np.array(df_close_price.drop(['Prediction'], 1))[:-future_days]
y = np.array(df_close_price['Prediction'])[:-future_days]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

regression_model = LinearRegression()
regression_model.fit(X_train, y_train)

x_future = df_close_price.drop(['Prediction'], 1)[:-future_days]
x_future = x_future.tail(future_days)
x_future = np.array(x_future)
x_future
lr_prediction = regression_model.predict(x_future)

predictions = lr_prediction

valid = df_close_price[X.shape[0]:]
valid['Prediction'] = predictions

plt.figure(figsize=(16,8))
plt.title('Basic Linear Regression Model by AP')
plt.xlabel('Days')
plt.ylabel('Close Price USD ($)')

plt.plot(amzn_df['Close_price'])
plt.plot(valid[['Close_price', 'Prediction']])
plt.legend(['original','valid','prediction'])
plt.show()

#Adding Additional Graph to view data labels with cursor
import plotly.express as px
data = pd.concat([amzn_df['Close_price'], valid[['Close_price', 'Prediction']]], axis=1)
data.columns = ['Original', 'Valid', 'Prediction']

fig = px.line(data, x=data.index, y=data.columns, labels={'value': 'Close Price USD ($)'})

fig.update_layout(
    title='Basic Linear Regression Model by AP',
    xaxis_title='Days',
    yaxis_title='Close Price USD ($)',
)

fig.show()

# Calculate the RMSE
rmse = np.sqrt(mean_squared_error(valid['Close_price'], valid['Prediction']))
rmse
