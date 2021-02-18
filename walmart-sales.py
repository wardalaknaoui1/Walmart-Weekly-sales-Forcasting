#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import Important libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose as sd
import warnings
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


# Load our data
features = pd.read_csv("C:/Users/WAHES CUSTOMER/Desktop/Kaggle/features.csv", parse_dates=['Date'])
stores = pd.read_csv("C:/Users/WAHES CUSTOMER/Desktop/Kaggle/stores.csv")
train = pd.read_csv("C:/Users/WAHES CUSTOMER/Desktop/Kaggle/train.csv", parse_dates=['Date'])
test = pd.read_csv("C:/Users/WAHES CUSTOMER/Desktop/Kaggle/test.csv", parse_dates=['Date'])


# In[ ]:


print(features.head(3))
print('\n')
print(stores.head(3))
print('\n')
print(train.head(3))


# In[ ]:


print(features.shape)
print(stores.shape)
print(train.shape)


# In[ ]:


# We will merge our datasets
df= train.merge(features, 'left').merge(stores, 'left')


# In[ ]:


df.head(5)


# In[23]:


df.info()


# In[24]:


df.describe().transpose()


# In[25]:


# Percentage of missing Values
df.isna().sum()/len(df)*100


# In[26]:


# Visualize our missing data
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# ## IMPUTING MISSING DATA

# In[27]:


# Missing data is for Markdowns only (Quantitative veriables). We can imput the missing data 
# using a 0, which indicates that there is no markdown.
df= df.fillna(0)
# DISPLAY MISSING DATA
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[28]:


# Correlation matrix
plt.figure(figsize= (15,10))
sns.heatmap(df.corr(), annot= True, cmap= 'coolwarm')


# In[29]:


# DISTRIBUTION OF THE DEPENDENT VARIABLE
plt.figure(figsize=(10,5))
sns.distplot(df['Weekly_Sales'], bins=40, kde=True, color='red')
plt.title('Weekly_Sales distribution')


# In[30]:


# Walmart weekly sales by Date(Using Plotly)
fig= px.line(df, x= 'Date', y= 'Weekly_Sales', title='Walmart Weekly Sales by Date', width=1000, height= 500)
fig.show()


# The highest sales were on Nov/25 and Nov/26, these are thanksgiving holidays.

# In[31]:


fig, ax = plt.subplots(2, 2, figsize= (10,10))
ax[0,0].scatter(df['Temperature'], df['Weekly_Sales'])
ax[0,0].set_title('Weekly_Sales by tempreture')
ax[0,1].scatter(df['Fuel_Price'], df['Weekly_Sales'])
ax[0,1].set_title('Weekly_Sales by fuel price')
ax[1,0].scatter(df['CPI'], df['Weekly_Sales'])
ax[1,0].set_title('Weekly_Sales by CPI')
ax[1,1].scatter(df['IsHoliday'], df['Weekly_Sales'])
ax[1,1].set_title('Weekly_Sales in holidays and not holidays')


# In[32]:


# COVERT "IsHoliday" into a dummy variable
df['IsHoliday'] = [int(x) for x in list(df.IsHoliday)]


#  # Modeling

# # Arima Model (Auto Regressive + Integration+  Moving Average)

# In[33]:


Series=train.groupby("Date")["Weekly_Sales"].sum()


# In[34]:


df_series= pd.DataFrame(Series)


# In[35]:


df_series.tail(3)


# In[36]:


df_series.describe()


# In[39]:


# Checking for Stationarity
rolmean= df_series.rolling(window=12).mean()
rolstd= df_series.rolling(window=12).std()
print(rolmean, rolstd)


# In[41]:


plt.figure(figsize=(15,5))
original= plt.plot(df_series, color= 'blue', label= 'Original')
mean= plt.plot(rolmean, color='red', label= 'Rolling Mean')
std= plt.plot(rolstd, color='black', label= 'Rolling std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
plt.show(block=False)


# In[ ]:


# From this graph, it is obvious that the mean and the standard deviation are constant, but we will test for it.


# In[ ]:


# Testing for Stationary
# TS is a partucular behavior over time, there is a very high probability that it will follow the same in the future.
# In time series we need to make sure that the data is, and if not fix that.
#stationary, which means the mean, variance, and autocorrelation structure do not change over time.


# In[23]:


# We have to test the hypothesis that the data is non-stationary
# H0:The data is non-stationary
# H1: The data is stationary


# In[327]:


from statsmodels.tsa.stattools import adfuller


# In[328]:


def adfuller_test(sales):
    result=adfuller(sales, autolag= 'AIC')
    labels = ['ADF Test Statistic','P-Value','Number of Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )


# In[329]:


results= adfuller(df_series.Weekly_Sales)


# In[330]:


adfuller_test(df_series['Weekly_Sales'])


# In[331]:


# Let's suppose that Alfa = 5% (0.05), which means if P value is less than alfa, we reject the null hypothesis and 
# conclude that the data is stationary
# In our case P-Value : 2.67597915898623e-07 which is too small (<0.05), therefore, we have enough evidence to
# reject the null hypothesis and conclude that the data is stationary (Which is what we are looking for.)


# # Decomposing Time Series Data into Trend and Seasonality
# A Series is an aggregate or combination of 4 components. All series have a level and noise. The trend and seasonality components are optional.
# * Level: The average value in the series.
# * Trend: The increasing or decreasing value in the series.
# * Seasonality: The repeating short-term cycle in the series.
# * Noise: The random variation in the series.

# In[332]:


# Weekly sales by date
sales_bydate = df.groupby("Date")["Weekly_Sales"].sum()
sales_bydate.head()


# In[333]:


from statsmodels.tsa.seasonal import seasonal_decompose
res = seasonal_decompose(sales_bydate.values, model='multiplicative', period=52)
res.plot()
plt.show()


# Based on these graphs, the trend line is going upward in certain areas along with that we have also seasonality present in high scale.

# In[334]:


# ACF and PACF tests


# In[335]:


shifted_data= df_series-df_series.shift()
shifted_data.head(5)


# In[336]:


# Drop NANs
shifted_data.dropna(inplace=True)


# ## Finding:
# ### p: The order of the AR term.
# ### d: The number of differencing required to make the time series stationary
# ### q:The order of the moving average term

# In[337]:


# Plotting ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf
lag_acf= acf(shifted_data, nlags=20)
lag_pacf= pacf(shifted_data, nlags= 20, method= 'ols')


#Plot ACF:
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0, linestyle= '--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(shifted_data)), linestyle= '--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(shifted_data)), linestyle= '--', color='gray')
plt.title('Autocorrelation Function')

#pLOT PACF
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle= '--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(shifted_data)), linestyle= '--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(shifted_data)), linestyle= '--', color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()


# In[338]:


# P and Q value are the values where the graph drops to 0 for the first time. So from the grpah 


# In[339]:


# Order:
from pmdarima import auto_arima


# In[340]:


stepwise_fit= auto_arima(df_series['Weekly_Sales'], trace= True, supress_warnings= True)
stepwise_fit.summary()


# In[341]:


from statsmodels.tsa.arima_model import ARIMA
train= df_series.iloc[:-30]
test= df.iloc[-30:]
print(train.shape, test.shape)


# In[342]:


model= ARIMA(train['Weekly_Sales'], order=(2,0,0))
model= model.fit()
model.summary()


# In[343]:


start= len(train)
end= len(train)+len(test)-1
pred= model.predict(start=start, end=end, typ= 'levels')
print(pred)


# In[ ]:


from sklearn.metric import mean_squared_error
from math import sqrt
rmse= sqrt(mean_squared_error(pred, test["Weekly_Sales"]))


# In[296]:


from sklearn.metrics import accuracy_score
from sklearn import metrics


# In[346]:


# Fitting an ARIMA model:
from statsmodels.tsa.arima_model import ARIMA
model= ARIMA(df_series, order=(2,0,0))
results_AR= model.fit(disp= -1)
plt.plot(shifted_data)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum(results_AR.fittedvalues - shifted_data['Weekly_Sales'])**2)
print('Plotting AR model')

