import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
from scipy.stats import norm
import statsmodels.formula.api as smf
import statsmodels.graphics as sgr
#from patsy.contrasts import Sum

from statsmodels.stats.proportion import proportions_ztest

from statsmodels.formula.api import logit

import datetime as dt

import matplotlib.pyplot as plt

import seaborn as sns
from statsmodels.graphics.gofplots import ProbPlot

# Read WOW DataSet from link
WOWdata = pd.read_csv('https://query.data.world/s/mJ6FBWT74epb8HilYA-uoba8yp2zFq')
WOWdata = WOWdata.rename(columns={"Time left on Auction": "Time_left_on_Auction"})
WOWdata['Date'] = pd.to_datetime(WOWdata['Date'])

#Check for nulls; No nulls found in the columns that we are interested.
print(WOWdata.isnull().any())

print(WOWdata.head(10))
#print(WOWdata.describe())
print(WOWdata.dtypes)

#Convert Date to an ordinal
WOWdata['DateOrdin'] = WOWdata['Date'].map(dt.datetime.toordinal)
WOWdata_grouped = WOWdata.groupby('DateOrdin', as_index=False)['Price'].agg(np.mean)

# Since we have an ordinal variable with many values we can go with simple OLS as our model.
mod = smf.ols("Price ~ DateOrdin", data=WOWdata_grouped).fit()
print(mod.summary())
print(mod.pvalues) #p-value less than 0.05.


#Regression Plot
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)

price = WOWdata.Price.values
date = WOWdata.DateOrdin.values
ax.scatter(date, price)
fig = sgr.regressionplots.abline_plot(model_results=mod, ax=ax)
plt.show()

# Normality Checks.
# fitted values
model_norm_residuals = mod.get_influence().resid_studentized_internal

QQ = ProbPlot(model_norm_residuals)
plot_lm_2 = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)

plot_lm_2.set_figheight(8)
plot_lm_2.set_figwidth(12)

plot_lm_2.axes[0].set_title('Normal Q-Q')
plot_lm_2.axes[0].set_xlabel('Theoretical Quantiles')
plot_lm_2.axes[0].set_ylabel('Standardized Residuals');
plt.show()
#Normality Test
print(stats.normaltest(model_norm_residuals)) #fails since p-value < 0.05.

#Check homoscedasticity
# fitted values
model_fitted_y = mod.fittedvalues

plot_lm_1 = plt.figure(1)
plot_lm_1.set_figheight(8)
plot_lm_1.set_figwidth(12)

plot_lm_1.axes[0] = sns.residplot(model_fitted_y, 'Price', data=WOWdata_grouped,
                          lowess=True,
                          scatter_kws={'alpha': 0.5},
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

plot_lm_1.axes[0].set_title('Residuals vs Fitted')
plot_lm_1.axes[0].set_xlabel('Fitted values')
plot_lm_1.axes[0].set_ylabel('Residuals')
plot_lm_1.show()


######################
# Find Upgrade Date
######################
tmp1 = WOWdata.copy()
#tmp1['Date'] = pd.to_datetime(tmp1['Date'])
tmp1.set_index('DateOrdin', inplace=True)
tmp1['Price'].plot(figsize=(16, 12))
plt.show()

dates = tmp1.index.values
#dates = pd.to_datetime(dates)
#print(dates.where((dates.year==2016) & (dates.month==9)).unique())
#print(dates[(dates > 736300) & (dates <= 736400)])

tmp2 = WOWdata.copy()
#print(tmp1.dtypes)
tmp2.set_index('DateOrdin', inplace=True)
#tmp2.truncate(before='2016-09-01 00:06:00')['Price'].plot(figsize=(16, 12))
tmp2.truncate(before=736355)['Price'].plot(figsize=(16, 12))
plt.show()

#We observe that the big pick is around 736360.
#Find the date for that DateOrd
print(tmp2.loc[736355]['Date'].unique())

#Split Data
WOWdata_before = WOWdata.loc[WOWdata.Date < '2017-01-26']
print('Max-Min dates before: ', WOWdata_before.Date.max(), WOWdata_before.Date.min())
WOWdata_after = WOWdata.loc[WOWdata.Date >= '2017-01-26']
print('Max-Min dates after: ',WOWdata_after.Date.max(), WOWdata_after.Date.min())

WOWdata_grouped_before = WOWdata_before.groupby('DateOrdin',as_index=False)['Price'].agg(np.mean)

mod_before = smf.ols("Price ~ DateOrdin", data=WOWdata_before).fit()
print(mod_before.summary())

WOWdata_grouped_after = WOWdata_after.groupby('DateOrdin',as_index=False)['Price'].agg(np.mean)

mod_after = smf.ols("Price ~ DateOrdin", data=WOWdata_grouped_after).fit()
print(mod_after.summary())
