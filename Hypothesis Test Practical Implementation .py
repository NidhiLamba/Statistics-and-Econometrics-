#!/usr/bin/env python
# coding: utf-8

# # Normality test using Shapiro-Wilk Test : tests If data is normally distributed 
# Assumption : Observations are identically distributed

# In[2]:


#Data Import
import pandas as pd
WineData = pd.read_csv('../train.csv')


# In[3]:


WineData.head()


# In[4]:


#Cheking Histogram
import matplotlib
from matplotlib import pyplot 
get_ipython().run_line_magic('matplotlib', 'inline')
pyplot.figure(figsize=(14,6))
pyplot.hist(WineData['LotFrontage'])
pyplot.show()


# In[5]:


#Help from Python
from scipy.stats import shapiro

DataToTest = WineData['LotFrontage']

stat, p = shapiro(DataToTest)

print('stat=%.2f, p=%.30f' % (stat, p))

if p > 0.05:
    print('Normal distribution')
else:
    print('Not a normal distribution')


# In[6]:


#Lets genrate normally distributed data from Python
from numpy.random import randn
DataToTest = randn(100)


# In[7]:


DataToTest


# In[9]:


stat, p = shapiro(DataToTest)

print('stat=%.2f, p=%.30f' % (stat, p))

if p > 0.05:
    print('Normal distribution')
else:
    print('Not a normal distribution')


# # Normality test using K^2 Normality Test Test : tests If data is normally distributed 
# Assumption : Observations are identically distributed

# In[11]:


# Example of the D'Agostino's K^2 Normality Test
from scipy.stats import normaltest
DataToTest = WineData['LotFrontage']

stat, p = normaltest(DataToTest)

print('stat=%.10f, p=%.10f' % (stat, p))

if p > 0.05:
    print('Normal')

else:
    print('Not Normllay distributed')


# # Correlation Test - Pearson and Spearman’s Rank Correlation
# Asumption - Identical and Normal Distribution

# In[12]:


FirstSample = WineData[1:30]['LotFrontage']
SecondSample = WineData[1:30]['LotArea']

pyplot.plot(FirstSample,SecondSample)
pyplot.show()


# In[13]:


#Spearman Rank Correlation
from scipy.stats import spearmanr
stat, p = spearmanr(FirstSample, SecondSample)

print('stat=%.3f, p=%5f' % (stat, p))
if p > 0.05:
    print('independent samples')
else:
    print('dependent samples')


# In[14]:


#pearson correlation
from scipy.stats import pearsonr
stat, p = pearsonr(FirstSample, SecondSample)

print('stat=%.3f, p=%5f' % (stat, p))
if p > 0.05:
    print('independent samples')
else:
    print('dependent samples')


# In[15]:


WineData[1:30].corr(method="pearson")


# # Correlation of categorical variable - Chi square test

# In[17]:


#Tests whether two categorical variables are related or independent.
#Assumptions - independent observation, size in each box of contingency table > 25
# Example of the Chi-Squared Test


# In[16]:


from scipy.stats import chi2_contingency


# In[17]:


contingency_data = [[25,125],[1200,240]] #Observe the numbers carefully


# In[18]:


stat, p, dof, expected = chi2_contingency(contingency_data)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('independent categories')
else:
    print('dependent categories')


# # Parametic test 1- T-test

# # Definiton of Parametric test - Main assumption - data is normally distributed

# In[20]:


#Scores of me and Virat
my_score = [23,21,31,20,19,35,26,22,21,19]
virat_score = [46,42,62,40,38,70,52,44,42,38]


# In[21]:


#Lets check mean of our scores
import numpy as np
print('Aman mean score:', np.mean(my_score))
print('Virat mean score:', np.mean(virat_score))


# In[22]:


#One Sample T-test
import scipy
scipy.stats.ttest_1samp(my_score,20)


# In[23]:


#Independent Sample T-test
scipy.stats.ttest_ind(my_score,virat_score)


# In[24]:


my_score_second_Tour = [46,42,62,40,38,70,52,44,42,38]


# In[25]:


#Apired sample t-test
scipy.stats.ttest_rel(my_score,my_score_second_Tour)


# # Parametic test 2 - Anova - Tests whether the means of two or more independent samples are significantly different.

# In[26]:


# Assumption -  Normal distributon, same variance, identical distribution


# In[27]:


average_score = [40,44,60,50,48,68,55,46,44,54]


# In[28]:


my_score


# In[29]:


average_score


# In[30]:


virat_score


# In[31]:


tstat, p = scipy.stats.f_oneway(my_score, average_score, virat_score)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Same distribution of scores')
else:
    print('Different distributions of scores')


# # Non Parametric test 1 - Mann-Whitney U Test-Tests whether the distributions of two or more independent samples are equal or not.

# In[32]:


#Assumptions - Idential distribution, observations can be ranked


# In[33]:


class_1_score = [91,90,81,80,76]
class_2_score = [88,86,85,84,83]


# In[34]:


tstat, p = scipy.stats.mannwhitneyu(class_1_score, class_2_score)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Same distribution')
else:
    print('Different distributions')


# In[35]:


# Similarly check for Wilcoxon Signed-Rank Test/Kruskal-Wallis H Test


# # Test of Stationarity - very Important for time series analysis

# In[36]:


#Augmented Dickey-Fuller Test -  null hypothesis - Series is non stationary


# In[37]:


#Definition of stationary time series - constant mean and variance


# In[38]:


from statsmodels.tsa.stattools import adfuller
stock_price_data = [121,131,142,121,131,142,121,131,142]
stat, p, lags, obs, crit, t = adfuller(stock_price_data)


# In[39]:


print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Series is not Stationary')
else:
    print('Series is stationary')


# In[ ]:


#Also check for Kwiatkowski-Phillips-Schmidt-Shin

