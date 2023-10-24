#!/usr/bin/env python
# coding: utf-8

# In[65]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[66]:


stock_data = pd.read_csv("Apple_HistoricalData.csv")
stock_data.head()


# In[67]:


stock_data.info()


# # Cleaning and Organizing DataSet

# In[68]:


stock_data.rename(columns={'Close/Last':'Close_Price'}, inplace=True)


# In[69]:


columns_to_clean = ['Close_Price','Open','High','Low']

for col in columns_to_clean:
    stock_data[col] = stock_data[col].str.replace('$','').astype(float)


# In[70]:


stock_data.head()


# # # Simple Linear Regression to Predict Stock prices and Accuracy testing

# In[71]:


## X, Y and allocating test_size for regression model
X= stock_data[['Open', 'High', 'Low', 'Volume']].values
y = stock_data['Close_Price'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[72]:


regression = LinearRegression()


# In[73]:


##Refression, coefficient and Intercept

regression.fit(X_train, y_train)


print(regression.coef_)
print(regression.intercept_)


# In[74]:


#Test Data, Actual vs Predicted and Variance in stock price for that day
prediction = regression.predict(X_test)

test = pd.DataFrame({'Actual': y_test.flatten(),'Predicted' : prediction.flatten()})

test["Variance"] = stock_data['High'] - stock_data['Low']

test.head()


# In[75]:


stock_data['prev_close'] = stock_data['Close_Price'].shift(-1)
stock_data['prev_volume'] = stock_data['Volume'].shift(-1)
stock_data['prev_high'] = stock_data['High'].shift(-1)
stock_data['prev_low'] = stock_data['Low'].shift(-1)

stock_data.dropna(inplace=True)


stock_data.head()



# In[76]:


stock_data.info()


# In[77]:


stock_data['Date_formatted'] = pd.to_datetime(stock_data['Date'])

stock_data.head()


# In[78]:


X= stock_data[['Open', 'prev_high', 'prev_low', 'prev_volume']].values
y = stock_data['Close_Price'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

regression.fit(X_train, y_train)


print(regression.coef_)
print(regression.intercept_)


# In[79]:


prediction = regression.predict(X_test)

test = pd.DataFrame({'Actual': y_test.flatten(),'Predicted' : prediction.flatten()})

test["Variance"] = stock_data['High'] - stock_data['Low']

test.head()


# In[80]:


# Create a fancy and unique scatter plot
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Actual', y='Predicted', data=test, hue='Variance', palette='coolwarm', size='Variance', sizes=(50, 200), edgecolor='k', alpha=0.8)

plt.title('Scatter Plot of Actual vs. Predicted Values with Variance', fontsize=16)
plt.xlabel('Actual Values', fontsize=14)
plt.ylabel('Predicted Values', fontsize=14)

# Custom color bar
colorbar = plt.colorbar(orientation='vertical')
colorbar.set_label('Variance (High - Low)', fontsize=12)

# Adding trendline
sns.regplot(x='Actual', y='Predicted', data=test, scatter=False, color='green', line_kws={'linestyle': '--'})

# Adding annotations
for i, row in test.iterrows():
    plt.annotate(f'Variance: {row["Variance"]:.2f}', (row['Actual'], row['Predicted']), fontsize=10, alpha=0.6)

# Adding gridlines
plt.grid(linestyle='--', alpha=0.6)

# Custom background
plt.style.use('seaborn-darkgrid')

# Show the plot
plt.show()


# In[ ]:


##Additional Pair Plot to understand relation between Variables to predict prices
sns.set(style="ticks")
sns.pairplot(stock_data, hue="Close_Price")
plt.show()

