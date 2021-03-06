""" Assignment 3 """

# libraries  - general
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd

# libraries - statistics
from scipy import interpolate

# library for clustering
import sklearn.cluster as cluster

# library for plotting
from scipy.optimize import curve_fit

# libraries for other methods
import random

# libraries for map method
import plotly.express as px

""" Functions for Data Pre-Processing """

# data pre-processing function 
def pre_processing(x):
    '''
    Parameters
    ----------
    x : dataset, performing functions described below on the dataset
    Returns
    -------
    the column values with ',' replaced and 
    stored as an int value
    '''
    head = x.head()             # displaying the first 5 lines in the data
    tail = x.tail()             # displaying the last 5 lines in the data
    columns = x.columns         # name of columns in dataset
    describe = x.describe       # general statistics on the dataset 
    info = x.info               # general statistics on the dataset 
    null = x.isna().sum()       # any nan values in columns of dataset
    dtype = x.dtypes            # data types of the columns in the dataset
    index = x.index             # the row identifirs
    
    return (f'The top 5 columns in the dataset = \n {head} \n \
            The bottom 5 columns in the dataset = \n {tail} \n \
            The name of the columns in the dataset = \n {columns} \n \
            The statistic description of the dataset = \n {describe} \n \
            The information on the dataset = \n {info} \n \
            The presence of any NA values = \n {null} \n \
            The datatype of the columns in the dataset = \n {dtype} \n \
            The index of the dataset = \n {index}') 
            
        
""" Functions to normalise columns in dataframe """

def normalise(x):
    
    min_val = np.min(x)
    max_val = np.max(x)
    
    scaled = (x - min_val) / (max_val - min_val)
    
    return scaled

def normalised_df(df):
    '''
    returns all columns of the dataframe normalised to [0,1]
    with the exception of the first column as it is categorical
    calls function normalise to do the normalisation
    '''
    # iterate over all columns
    for col in df.columns[1:]:  #except categorical one
        df[col] = normalise(df[col])
        
    return df



""" Importing Mortality Dataset and performing functions """

# load in the mortality dataframe 
df1 = pd.read_csv('C:/Users/sjjby/Documents/Applied Data Science 1/Assignment 3/mortality.csv',
                  header=2,
                  engine='python')            

# drop columns that are unrelevant
df1.drop(['Country Code', 'Indicator Name', 'Indicator Code', '2020', 'Unnamed: 65'],
         axis=1,
         inplace=True)

# performing pre-processing on dataset
print(pre_processing(df1))

# doing the normalise_df function on dataset
df2 = df1.copy()
df2 = normalised_df(df2)


""" Importing Dataset HDI 2019 and performing functions """

# importing dataset 
df_hdi = pd.read_excel('C:/Users/sjjby/Documents/Applied Data Science 1/Assignment 3/clustering/HDI.xlsx') 

# rename column in table 
df_hdi.rename(columns={"hdi2019": "Human Development Index"},
              inplace=True)

# pre-processing the data 
print(pre_processing(df_hdi))

##########################################################################

""" Figure 1 - World map of HDI levels """

# create figure for the newest measurements
fig1 = px.choropleth(df_hdi,
                     locations='country',
                     locationmode="country names",
                     color='Human Development Index', 
                     scope="world")

# plotly is browser based so had to write as a html to view image
fig1.write_html('C:/Users/sjjby/Documents/Applied Data Science 1/Assignment 3/worldmap.html')

###############################################################################


""" Importing Dataset HDI 2000 and performing functions """

# importing dataset
df_hdi2000 = pd.read_excel('C:/Users/sjjby/Documents/Applied Data Science 1/Assignment 3/hdi2000.xlsx')

# pre-processing the data 
print(pre_processing(df_hdi2000))                           

""" Importing Dataset HDI 2017 and performing functions """

# importing dataset
df_hdi2017 = pd.read_excel('C:/Users/sjjby/Documents/Applied Data Science 1/Assignment 3/hdi2017.xlsx')

# pre-processing the data 
print(pre_processing(df_hdi2017)) 

########################################################################

""" Merging Mortality and 2019 HDI dataset """

# renaming columns so they are the same in both datasets
df_hdi.rename(columns={"country": "Country Name"},
              inplace=True)

# merging df2(normalised df1) and df_hdi together
df_comb = df2.copy()
df_comb = df2.merge(df_hdi, 
                    on=['Country Name'],  # the column name that is the same in both datasets
                    how='inner')  #the method of merging (inner)

# export to excel to make sure it did it correctly
df_comb.to_excel('C:/Users/sjjby/Documents/Applied Data Science 1/Assignment 3/combined_df.xlsx', 
                engine='xlsxwriter')  

'''
the df_comb now contains both the HDI of 2019 and 
the mortality rates for countries over the period 
from 1960 until 2019. This table only contains 
the country names that are the same in both tables 
e.g. Aruba is not in the new dataset because the 
df_hdi did not have the HDI value for that country
'''

""" Merging Mortality and 2000 HDI dataset """

# merging df2(normalised df1) and df_hdi together
df_comb2000 = df2.copy()
df_comb2000 = df2.merge(df_hdi2000, 
                        on=['Country Name'],  # the column name that is the same in both datasets
                        how='inner')  #the method of merging (inner)

# export to excel to make sure it did it correctly
df_comb2000.to_excel('C:/Users/sjjby/Documents/Applied Data Science 1/Assignment 3/combined_dfhdi2000.xlsx', 
                     engine='xlsxwriter')

""" Merging Mortality and 2017 HDI dataset """

# merging df2(normalised df1) and df_hdi together
df_comb2017 = df2.copy()
df_comb2017 = df2.merge(df_hdi2017, 
                        on =['Country Name'],  # the column name that is the same in both datasets
                        how='inner')  #the method of merging (inner)

# export to excel to make sure it did it correctly
df_comb2000.to_excel('C:/Users/sjjby/Documents/Applied Data Science 1/Assignment 3/combined_dfhdi2000.xlsx', 
                     engine='xlsxwriter')

##############################################################################

""" Clustering Aspect of Assignment """

""" Correlation between Mortality Dataset columns """

# correlation 1:
print(f'Correlation between columns (Years vs Years) \
      \n {df1.corr()}')

#correlation 2:

# copying the data
mort_df = df1.copy()

# transposing the data
mort_df.set_index('Country Name').transpose()

# resetting the index
mort_df.reset_index(level=0,
                    inplace=True)

# the column had wrong name so have to rename
mort_df.rename(columns={"index": "Year"},
               inplace=True)

print(f'Correlation between columns (Country vs Country) \
      \n {mort_df.corr()}')


""" Filtering the correlated Mortality dataset """

# filter to be not highly correlated or anti-correlated

# have to copy the data to remove categorical first column
fil_df = df1.copy()

# remove the categorical column 
fil_df.drop(['Country Name'],
            axis=1,
            inplace=True)

# creating filtered dataset
filter_df1 = fil_df[(fil_df < 0.5) & (fil_df > -0.5)]

# removing rows with NA values
filter_df1.dropna(axis=0,     # rows
                  how='any',  # any value need to be NA
                  inplace=True)

# exporting to xlsx file for eye inspection 
filter_df1.to_excel('C:/Users/sjjby/Documents/Applied Data Science 1/Assignment 3/filtered_mortality_Data.xlsx')

#######################################################################

""""""" Performing clustering on the Combined dataset """""""

""" Affinity CLustering """

ap = cluster.AffinityPropagation(max_iter=1000, 
                                 verbose=True)

# selecting columns 
df_fit1 = df_comb[['2019', '1960']].copy()
df_fit1.dropna(inplace=True)

# runnig it
ap.fit(df_fit1)

# extract labels and centres
labels1 = ap.labels_

# finding out how many labels there are to assign colours to
print(labels1)

cen = ap.cluster_centers_

# plot using the labels to select colour
plt.figure(figsize=(5.0,5.0))

col = ["pink", "tomato", "navy", "gold", "slateblue", "orange", "darkmagenta"]

# loop over the different labels
for l in range(0, 7): 
    plt.plot(df_fit1['2019'][labels1==l], 
             df_fit1['1960'][labels1==l], 
             "o", 
             markersize=3, 
             color=col[l])

# show cluster centres
for ic in range(7):
    xc, yc = cen[ic,:]
    plt.plot(xc, yc, 
             "dk", 
             markersize=8,
             color='black')
    
plt.xlabel("2019")
plt.ylabel("1960")
plt.title("Affinity Clustering")
plt.show()


""" KMEANS Clustering """

kmeans = cluster.KMeans(n_clusters=5)

# extract columns for fitting 
df_fit2 = df_comb[['2019', '1960']].copy()
df_fit2.dropna(inplace=True)

kmeans.fit(df_fit2)

# extract labels and cluster centres
labels2 = kmeans.labels_
cen = kmeans.cluster_centers_

# plot using the labels to select colour 
plt.figure(figsize=(5, 5))

col = ["gold", "navy", "orange", "tomato", "darkmagenta"]
for l in range(5):
    plt.plot(df_fit2['2019'][labels2==l], 
             df_fit2['1960'][labels2==l],
             "o",
             markersize=3,
             color=col[l])
    
# show cluster centres
for ic in range(5):
    xc, yc = cen[ic, :]
    plt.plot(xc, yc,
             "dk",
             markersize=8,
             color='black')
    
plt.xlabel("2019")
plt.title('Kmeans clustering')
plt.ylabel("1960")
plt.show()


""" Agglomerative clustering """

# do agglomerative etc 
ac = cluster.AgglomerativeClustering(n_clusters = 5)

# extract columns for fitting 
df_fit3 = df_comb[['2019', '1960']].copy()
df_fit3.dropna(inplace=True)

# carry out the fitting
ac.fit(df_fit3)

labels3 = ac.labels_

# the agglomerative clusteres does not return cluster centers
# so am making them :)
xcen = []
ycen = []

for ic in range(5):
    xc = np.average(df_fit3["2019"][labels3==ic])
    yc = np.average(df_fit3["1960"][labels3==ic])
    xcen.append(xc)
    ycen.append(yc)

# plot using the labels to select colour 
plt.figure(figsize = (5, 5))

col = ["orange", "slateblue", "navy", "gold", "tomato"]
for l in range(0, 5):
    plt.plot(df_fit3['2019'][labels3==l], 
             df_fit3['1960'][labels3==l],
             "o",
             markersize=3,
             color=col[l])
    
# show cluster centres
for ic in range(5):
    plt.plot(xcen[ic], ycen[ic],
             "dk",
             markersize=8,
             color='black')
    
plt.xlabel("2019")
plt.ylabel("1960")
plt.title('Agglomerative clustering')
plt.show()



""" Exporting clustering labels to excel column """

# making a new dataframe for ease of analysis
df_new = df_comb[['Country Name', '1960', '2019']]

# removing NA values to mirror the clustering algorithms
df_new.dropna(inplace=True)

# making new columns
df_new['affinity'] = labels1
df_new['kmeans'] = labels2
df_new['agglomerative'] = labels3

# export to excel
df_new.to_excel('C:/Users/sjjby/Documents/Applied Data Science 1/Assignment 3/output.xlsx', 
                engine='xlsxwriter')  


""" Figure 4 - Subplot of each label from affinity clustering """

# subplot of each of the labels in affinity clustering
fig4, axs4 = plt.subplots(2, 4, 
                          figsize=(20, 10),
                          sharey=True,
                          sharex=True)

axs4[0,0].scatter(df_new[df_new['affinity']==3]['2019'],
                  df_new[df_new['affinity']==3]['1960'],
                  color='gold')
axs4[0,1].scatter(df_new[df_new['affinity']==5]['2019'],
                  df_new[df_new['affinity']==5]['1960'],
                  color='darkmagenta')
axs4[0,2].scatter(df_new[df_new['affinity']==0]['2019'],
                  df_new[df_new['affinity']==0]['1960'],
                  color='pink')
axs4[0,3].scatter(df_new[df_new['affinity']==6]['2019'],
                  df_new[df_new['affinity']==6]['1960'],
                  color='orange')
axs4[1,0].scatter(df_new[df_new['affinity']==1]['2019'],
                  df_new[df_new['affinity']==1]['1960'],
                  color='navy')
axs4[1,2].scatter(df_new[df_new['affinity']==2]['2019'],
                  df_new[df_new['affinity']==2]['1960'],
                  color='indigo')
axs4[1,1].scatter(df_new[df_new['affinity']==4]['2019'],
                  df_new[df_new['affinity']==4]['1960'],
                  color='slateblue')

plt.show()


""" Filtering the Clustered Mortality data """

# setting the seed 
random.seed(180418)

# obtaining the country names in label 0
print(df_new[df_new['affinity'] == 0]['Country Name'])
# randomly selecting three country names for plotting
print(df_new[df_new['affinity'] == 0]['Country Name'].sample(n=3))

# obtaining the country names in label 1
print(df_new[df_new['affinity'] == 1]['Country Name'])
# randomly selecting three country names for plotting
print(df_new[df_new['affinity'] == 1]['Country Name'].sample(n=3))

# obtaining the country names in label 2
print(df_new[df_new['affinity'] == 2]['Country Name'])
# randomly selecting three country names for plotting
print(df_new[df_new['affinity'] == 2]['Country Name'].sample(n=3))

# obtaining the country names in label 3
print(df_new[df_new['affinity'] == 3]['Country Name'])
# randomly selecting three country names for plotting
print(df_new[df_new['affinity'] == 3]['Country Name'].sample(n=3))

# obtaining the country names in label 4
print(df_new[df_new['affinity'] == 4]['Country Name'])
# randomly selecting three country names for plotting
print(df_new[df_new['affinity'] == 4]['Country Name'].sample(n=3))

# obtaining the country names in label 5
print(df_new[df_new['affinity'] == 5]['Country Name'])
# randomly selecting three country names for plotting
print(df_new[df_new['affinity'] == 5]['Country Name'].sample(n=3))

# obtaining the country names in label 6
print(df_new[df_new['affinity'] == 6]['Country Name'])
# randomly selecting three country names for plotting
print(df_new[df_new['affinity'] == 6]['Country Name'].sample(n=3))

# obtaining the country names in label 7
print(df_new[df_new['affinity'] == 0]['Country Name'])
# randomly selecting three country names for plotting
print(df_new[df_new['affinity'] == 0]['Country Name'].sample(n=3))


""" Figure 5 - Line plots of the clusters of countries """

df3 = df1.copy()
df3 = df3.set_index('Country Name').transpose()

fig5, axs5 = plt.subplots(figsize=(15, 10))

# label 3 (MDC)
axs5.plot(df3['Israel'],
          color='darkmagenta',
          linestyle='solid',
          label='Israel')

# label 2 (LDC)
axs5.plot(df3['Guinea'],
          color='gold',
          linestyle='solid',
          label='Guinea')

# label 3 (MDC)
axs5.plot(df3['Zambia'],
          color='magenta',
          linestyle='solid',
          label='Zambia')

# label 2 (LDC)
axs5.plot(df3['Togo'],
          color='orange',
          linestyle='solid',
          label='Togo')

# set x label
axs5.tick_params(labelrotation=90)
axs5.set_xlabel('Years',
                   fontsize=20)

# set y label
axs5.set_ylabel('Mortality Rate',
                fontsize=20)
axs5.set_ylabel('Mortality Rate',
                fontsize=20)

# set legend
axs5.legend(loc='upper right',
            fontsize=15)

# showing the figure
fig5.show()


""" Performing Statistics on Affinity Clustering result """

"""
Owing to the HDI level being between 0-1 and seven (0-6)
labels were achieved using affinity clustering. I can use 
this to see if the mortality clustering reflects the 
HDI level. Therefore can assume 7 groups and have
normal distribution (starting HDI level = 0.394).
Thus, can compare the binary observed and expected for 
each country. Use Students un-Paired T-Test.
"""

print("Mean of HDI =", '\n', 
      df_comb['Human Development Index'].mean())

print("standard deviation of HDI", '\n',
      np.std(df_comb['Human Development Index']))

df10 = df_comb.copy()
df10.dropna(inplace=True)

# rounding HDI column to 2 decimal places 
df10['Human Development Index'] = df10['Human Development Index'].round(2)

# add new column to dataframe
df10['Expected'] = df10['Human Development Index']

# applying function .loc(condition = value) to the dataframe
df10.loc[df10['Human Development Index'] < 0.47, 'Expected'] = 3
df10.loc[(df10['Human Development Index'] > 0.475) & (df10['Human Development Index'] < 0.555), 'Expected'] = 2
df10.loc[(df10['Human Development Index'] > 0.556) & (df10['Human Development Index'] < 0.635), 'Expected'] = 1
df10.loc[(df10['Human Development Index'] > 0.636) & (df10['Human Development Index'] < 0.715), 'Expected'] = 6
df10.loc[(df10['Human Development Index'] > 0.716) & (df10['Human Development Index'] < 0.795), 'Expected'] = 0
df10.loc[(df10['Human Development Index'] > 0.796) & (df10['Human Development Index'] < 0.87), 'Expected'] = 5
df10.loc[df10['Human Development Index'] > 0.865, 'Expected'] = 4

# export to excel
df10.to_excel('C:/Users/sjjby/Documents/Applied Data Science 1/Assignment 3/expected.xlsx', 
              engine = 'xlsxwriter')

""" Combining two dataframes (not merging) """

# making a new dataframe for ease of analysis
df_compare = df10[['Country Name', 'Expected']]

# removing NA values to mirror the clustering algorithms
df_compare.dropna(inplace=True)

# making new column
df_compare['affinity'] = labels1

# export to excel
df_compare.to_excel('C:/Users/sjjby/Documents/Applied Data Science 1/Assignment 3/compare.xlsx', 
                    engine='xlsxwriter') 

""" Performing un - Paired Students T-test """

'''
why?
the observations are un- paired but, 
the means should be similar.
h0 = means of sample are equal
h1 = means of sample are unequal
'''

from scipy.stats import ttest_ind

stat, p = ttest_ind(df_compare['Expected'],
                    df_compare['affinity'])

print("############################################")

print('stat = %.3f, p = %.3f' % (stat, p))

if p > 0.05:
    print('Same distribution - ACCEPT H0')
else:
    print('Different distribution REJECT H0')

print("############################################")

#######################################################################

""" Predictive Modelling """

# use just LDC and MDC countries 

# making dataset of just HDI levels and 2019 mortality rates
pred_df = df_comb[['Country Name', '2019', 'Human Development Index']]

pred_df.dropna(inplace=True)

""" Model Fitting the data """

""" Linear """

x = pred_df['Human Development Index']
y = pred_df['2019']

# calculate the mean of x and y 
xmean = np.mean(x)
ymean = np.mean(y)

# calculating the terms for numerator and denominator 
pred_df['xycov'] = (pred_df['Human Development Index'] - xmean) * (pred_df['2019'] - ymean)
pred_df['xvar'] = (pred_df['Human Development Index'] - xmean)**2

# coefficients
beta = pred_df['xycov'].sum() / pred_df['xvar'].sum()
alpha = ymean - (beta*xmean)

print(f'alpha = {alpha}')
print(f'beta = {beta}')

# the line 
ypred = alpha + beta * x

""" Exponential """

# Function to calculate the exponential with constants a and b
def exponential(x, a, b):
    return a*np.exp(b*x)

# making the function above fit data
pars2, cov2 = curve_fit(f=exponential, 
                        xdata=x, 
                        ydata=y)
print(pars2)
# arranging the x values
x_exp = np.arange(min(pred_df['Human Development Index']), 
                  max(pred_df['Human Development Index']), 
                  step=0.01)

""" Polynomial """

# Function to calculate the power-law with constants a and b
def power_law(x, a, b):
    return a*np.power(x, b)

pars3, cov3 = curve_fit(f=power_law, 
                        xdata=x, 
                        ydata=y)
print('pars3 =', pars3)

""" New polynomial """

# define the true objective function
def objective(x, a, b, c):
	return a * x + b * x**2 + c

pars4, cov4 = curve_fit(objective, x, y)

# summarize the parameter values
a, b, c = pars4
print('y = %.5f * x + %.5f * x^2 + %.5f' % (a, b, c))

x1 = np.arange(min(pred_df['Human Development Index']), 
               max(pred_df['Human Development Index']), 
               step = 0.01)
y1 = -4.83938 * x1 + 2.45677 * x1**2 + 2.38284


""" Figure 8 - Original Data and Models """

fig8, axs8 = plt.subplots(2, 3, 
                          figsize=(18, 15))

# original data + linear line of best fit
axs8[0,0].scatter(pred_df['Human Development Index'],
                  pred_df['2019'],
                  color='blue')
# line of best-fit
axs8[0,0].plot(pred_df['Human Development Index'],
               ypred,
               color='orange')


# original data + exponential line of best fit (no-log scale)
axs8[0,1].scatter(pred_df['Human Development Index'],
                  pred_df['2019'],
                  color='blue')
# exponential line
axs8[0,1].plot(x_exp, 
               exponential(x_exp, *pars2), 
               linestyle='--', 
               linewidth=2, 
               color='orange')

# original data + exponential line of best fit (log scale)
axs8[0,2].scatter(pred_df['Human Development Index'],
                  pred_df['2019'],
                  color='blue')
# Set the y-axis scaling to logarithmic
axs8[0,2].set_yscale('log')
# exponential line
axs8[0,2].plot(x, 
               exponential(x, *pars2), 
               linestyle='--', 
               linewidth=2, 
               color='orange')


# original data + polynomial line of best fit
axs8[1,0].scatter(pred_df['Human Development Index'],
                  pred_df['2019'],
                  color='blue')
# Set the y-axis and x-axis scaling to logarithmic
axs8[1,0].set_yscale('log')
axs8[1,0].set_xscale('log')
# polynomial line 
axs8[1,0].plot(x, 
               power_law(x, *pars3), 
               linestyle='--', 
               linewidth=2, 
               color='orange')


# original data + objective line
axs8[1,1].scatter(pred_df['Human Development Index'],
                  pred_df['2019'],
                  color='blue')
# line plot
axs8[1,1].plot(x1, y1, 
               linestyle='--', 
               color='orange')


# original data + objective line
axs8[1,2].scatter(pred_df['Human Development Index'],
                  pred_df['2019'],
                  color='blue',
                  label='Original Data')
# Set the y-axis scaling to logarithmic
axs8[1,2].set_yscale('log')
axs8[1,2].set_xscale('log')
# line plot
axs8[1,2].plot(x1, y1, 
               linestyle='--', 
               color='orange')

# set titles 
axs8[0,0].set_title('Linear')
axs8[0,1].set_title('Exponential')
axs8[0,2].set_title('Exponential')
axs8[1,0].set_title('Polynomial')
axs8[1,1].set_title('Polynomial')
axs8[1,2].set_title('Polynomial')

# setting y-labels
axs8[0,0].set_ylabel('Mortality Rates in 2019')
axs8[0,1].set_ylabel('Mortality Rates in 2019')
axs8[0,2].set_ylabel('Mortality Rates in 2019 (log scale)')
axs8[1,0].set_ylabel('Mortality Rates in 2019')
axs8[1,1].set_ylabel('Mortality Rates in 2019')
axs8[1,2].set_ylabel('Mortality Rates in 2019')

# setting x-labels
axs8[0,0].set_xlabel('Human Development Index in 2019')
axs8[0,1].set_xlabel('Human Development Index in 2019')
axs8[0,2].set_xlabel('Human Development Index in 2019 (log scale)')
axs8[1,0].set_xlabel('Human Development Index in 2019')
axs8[1,1].set_xlabel('Human Development Index in 2019')
axs8[1,2].set_xlabel('Human Development Index in 2019')

# show figure
fig8.show()

#######################################################################

""" Residual analysis to observe the best-fitting model """

# define x
x = pred_df['Human Development Index']

# define y
y = pred_df['2019']

""" Linear """
# Get the standard deviations of the parameters 
stdevs1 = np.std(x)
# Calculate the residuals
res_lin1 = y - ypred

""" Exponential """
# Get the standard deviations of the parameters 
stdevs2 = np.sqrt(np.diag(cov2))
# Calculate the residuals
res_exp1 = y - exponential(x, *pars2)

""" Polynomial """

# Get the standard deviations of the parameters 
stdevs3 = np.sqrt(np.diag(cov3))
# Calculate the residuals
res_poly1 = y - power_law(x, *pars3)

# Get the standard deviations of the parameters 
stdevs4 = np.sqrt(np.diag(cov4))
# Calculate the residuals
res_poly2 = y - objective(x, *pars4)

""" Figure 10 - residuals + original data (for report) """

fig10, axs10 = plt.subplots(4, 2,
                          figsize=(15, 18))

# original data + linear line of best fit
axs10[0,0].scatter(pred_df['Human Development Index'],
                   pred_df['2019'],
                   color='darkmagenta')
# line of best-fit
axs10[0,0].plot(pred_df['Human Development Index'],
                ypred,
                linestyle='--',
                color='orange',
                label="y = 1.246 - 1.414x")
# residuals of linear
axs10[0,1].scatter(x, res_lin1,
                   color='magenta')
axs10[0,1].hlines(y=0,
                  xmin=0.35,
                  xmax=1.0,
                  linewidth=2,
                  color='black')

# original data + exponential line of best fit (no-log scale)
axs10[1,0].scatter(pred_df['Human Development Index'],
                   pred_df['2019'],
                   color='darkmagenta')
# exponential line
axs10[1,0].plot(x_exp, 
                exponential(x_exp, *pars2), 
                linestyle='--', 
                linewidth=2, 
                color='orange',
                label="y = 7.69e^-5.282x")
# residuals of exponential
axs10[1,1].scatter(x, res_exp1,
                   color='magenta')
axs10[1,1].hlines(y=0,
                  xmin=0.35,
                  xmax=1.0,
                  linewidth=2,
                  color='black')

# original data + polynomial line of best fit
axs10[2,0].scatter(pred_df['Human Development Index'],
                   pred_df['2019'],
                   color='darkmagenta')
# polynomial line 
axs10[2,0].plot(x1, 
                power_law(x1, *pars3), 
                linestyle='--', 
                linewidth=2, 
                color='orange',
                label="y = 0.066x^-295")
# residuals of polynomial 1
axs10[2,1].scatter(x, res_poly1,
                   color='magenta')
axs10[2,1].hlines(y=0,
                  xmin=0.35,
                  xmax=1.0,
                  linewidth=2,
                  color='black')

# original data + objective line
axs10[3,0].scatter(pred_df['Human Development Index'],
                  pred_df['2019'],
                  color='darkmagenta')
# line plot
axs10[3,0].plot(x1, y1, 
               linestyle='--', 
               color='orange',
               label="y = 2.46x^2 - 4.84x + 2.38")
# residuals of polynomial 2
axs10[3,1].scatter(x, res_poly2,
                   color='magenta')
axs10[3,1].hlines(y=0,
                  xmin=0.35,
                  xmax=1.0,
                  linewidth=2,
                  color='black')

# set x-label
axs10[0,0].set_xlabel('Human Development Index in 2019', 
                      fontsize=15)
axs10[1,0].set_xlabel('Human Development Index in 2019', 
                      fontsize=15)
axs10[2,0].set_xlabel('Human Development Index in 2019', 
                      fontsize=15)
axs10[3,0].set_xlabel('Human Development Index in 2019', 
                      fontsize=15)
axs10[0,1].set_xlabel('Original Data', 
                      fontsize=15)
axs10[1,1].set_xlabel('Original Data', 
                      fontsize=15)
axs10[2,1].set_xlabel('Original Data', 
                      fontsize=15)
axs10[3,1].set_xlabel('Original Data', 
                      fontsize=15)

# set legend
axs10[0,0].legend()
axs10[1,0].legend()
axs10[2,0].legend()
axs10[3,0].legend()

# set y-label
axs10[0,0].set_ylabel('Mortality Rates in 2019', 
                      fontsize=15)
axs10[1,0].set_ylabel('Mortality Rates in 2019', 
                      fontsize=15)
axs10[2,0].set_ylabel('Mortality Rates in 2019', 
                      fontsize=15)
axs10[3,0].set_ylabel('Mortality Rates in 2019', 
                      fontsize=15)
axs10[0,1].set_ylabel('Residuals', 
                      fontsize=15)
axs10[1,1].set_ylabel('Residuals', 
                      fontsize=15)
axs10[2,1].set_ylabel('Residuals', 
                      fontsize=15)
axs10[3,1].set_ylabel('Residuals', 
                      fontsize=15)

# show figure
fig10.show()

'''
from the graphs the best line of fit is the second polynomial one
in axs10 position [2,0] because the residuals are around 0
indicating a good fit.
'''

#######################################################################

""" Applying Predictive model to other years for analysis """

'''
to observe the fit on other years - 
how? by distribution of the residuals
'''

# re-defining x1 and y1 from above (for YOUR ease :) )
x1 = np.arange(min(pred_df['Human Development Index']), 
               max(pred_df['Human Development Index']), 
               step=0.01)
y1 = -4.83938 * x1 + 2.45677 * x1**2 + 2.38284

# residuals of the year 2000 compared to line of best-fit
w1 = df_comb2000['Human Development Index']
z1 = df_comb2000['2000']
# Calculate the residuals
res_poly5 = z1 - (2.46*w1**2 - 4.84*w1 + 2.38)

# residuals of the year 2017 compared to line of best-fit
w2 = df_comb2017['Human Development Index']
z2 = df_comb2017['2017']
# Calculate the residuals
res_poly6 = z2 - (2.46*w2**2 - 4.84*w2 + 2.38)


""" Figure 12 """

fig12, axs12 = plt.subplots(2, 2,
                            figsize = (18, 15))

# data from df_comb2000
axs12[0,0].scatter(df_comb2000['Human Development Index'],
                   df_comb2000['2000'],
                   color='navy')
# predictive line from above code (equation in label)
axs12[0,0].plot(x1, y1, 
                linestyle='--', 
                color='orange',
                label="y = 2.46x^2 - 4.84x + 2.38")
# residuals of polynomial 2
axs12[0,1].scatter(z1, res_poly5,
                   color='slateblue')
axs12[0,1].hlines(y=0,
                  xmin=0,
                  xmax=1.0,
                  linewidth=2,
                  color='black')

# data from df_comb2017
axs12[1,0].scatter(df_comb2017['Human Development Index'],
                   df_comb2017['2017'],
                   color='navy')
# predictive line from above
axs12[1,0].plot(x1, y1, 
                linestyle='--', 
                color='orange',
                label="y = 2.46x^2 - 4.84x + 2.38")
# residuals of polynomial 2
axs12[1,1].scatter(z2, res_poly6,
                   color='slateblue')
axs12[1,1].hlines(y=0,
                  xmin=0,
                  xmax=1.0,
                  linewidth=2,
                  color='black')

# set x-labels
axs12[0,0].set_xlabel('Human Development Index (2000)', 
                      fontsize=15)
axs12[0,1].set_xlabel('Original Data', 
                      fontsize=15)
axs12[1,0].set_xlabel('Human Development Index (2017)', 
                      fontsize=15)
axs12[1,1].set_xlabel('Original Data', 
                      fontsize=15)

# set y-labels
axs12[0,0].set_ylabel('Mortality Rate (2000)', 
                      fontsize=15)
axs12[1,0].set_ylabel('Mortality Rate (2017)', 
                      fontsize=15)
axs12[0,1].set_ylabel('Residuals', 
                      fontsize=15)
axs12[1,1].set_ylabel('Residuals', 
                      fontsize=15)

# set legend
axs12[0,0].legend()
axs12[1,0].legend()

# show figure 12
fig12.show()

'''
therefore this prediction is applicable for years surrounding 2019
and cannot be back-dated 
'''

#######################################################################

""" Predictive Model using Interpolation """

# transposing data for ease of manipulation
new_df = df2.copy()
new_df = new_df.set_index('Country Name').transpose()

# resetting the index
new_df.reset_index(level=0,
                   inplace=True)

# the column had wrong name so have to rename
new_df.rename(columns={"index": "Year"},
              inplace=True)

new_df['Year'] = new_df['Year'].astype(float)

# performing interpolation on the Netherlands for Predictive model:
a = new_df['Year']
b = new_df['Netherlands']

# nearest-neighbour 
nn_interp = interpolate.interp1d(a, b,
                                 kind="nearest", 
                                 copy=True)

# linear 
lin_interp = interpolate.interp1d(a, b,
                                  kind="linear", 
                                  copy=True)
# cubic
cub_interp = interpolate.interp1d(a, b,
                                  kind="cubic", 
                                  copy=True)

xnew = a
ynew1 = nn_interp(xnew)
ynew2 = lin_interp(xnew)
ynew3 = cub_interp(xnew)


""" Figure 13 - Prediction using Interpolation only """


# defining the figure
fig13, axs13 = plt.subplots(2,2,
                            figsize=(18, 15))

# plot data
axs13[0,0].scatter(new_df['Year'],
                   new_df['Netherlands'], 
                   color='indigo')

# predictive model
axs13[0,1].plot(xnew, ynew1, 
                color='magenta')
axs13[1,0].plot(xnew, ynew2, 
                color='orange')
axs13[1,1].plot(xnew, ynew3, 
                color='pink')

# plot figure 13
fig13.show()

'''
not used in report owingg to the other predictive modelling
looking better/ can be used more 
'''

#######################################################################

""" Object-oriented programming """

'''
create a class and use its methods 
get and extra 3.3%

i have made a random OOP to aid my knowledge of 
how the class works.
better than lists because do not need to find indexes etc.
'''

# define object 

# class name using camel method
class Dog:
       
    def bark(self):
        print("bark")
        
# assigning variable d is an object of the class dog
d = Dog()

# seeing if the class so far works 
d.bark()

class Dog:
    # this method enables to pass any variable to this method
    # stored permanetly as object
    def __init__(self, name):
        self.name = name
        print(name)


d = Dog("Jimmy")
print(d.name)


d2 = Dog("Lucy")
print(d.name)


class Dog:
    # this method enables to pass any variable to this method
    # stored permanetly as object
    def __init__(self, name, age):
        self.name = name
        self.age = age
        
    def get_name(self):
        return self.name
    
    def get_age(self):
        return self.age
    
    
d1 = Dog("Jimmy", 15)
print(d1.get_name()) 
print(d1.get_age()) 

  
class Dog:
    # this method enables to pass any variable to this method
    # stored permanetly as object
    def __init__(self, name, age):
        self.name = name
        self.age = age
        
    def get_name(self):
        return self.name
    
    def get_age(self):
        return self.age
    
    def set_age(self, age):
        self.age = age
    
    
d1 = Dog("Jimmy", 15)
print(d1.get_name()) 
print(d1.get_age()) 

# changing the age of dog
d1.set_age(16)
# seeing it worked
print(d1.get_age()) 
    
  
####################################################################

# multiple classes and how they interact
# using grades of a group of students


class Student:
    
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade # 0-100
        
    def get_grade(self):
        return self.grade
    

class Course:
    
    def __init__(self, name, max_students):
        self.name = name
        self.max_students = max_students
        self.students = []
        
    def add_student(self, student):
        if len(self.students) < self.max_students:
            self.students.append(student)
            return True
        return False
    
    def get_average_grade(self):
        num = 0
        for student in self.students:
            num += student.get_grade()
        
        return num / len(self.students)


# defining variables and what the class requires (name, age, grade)
s1 = Student("James", 23, 70)
s2 = Student("Karen", 54, 80)
s3 = Student("SJ", 21, 100)
s4 = Student("John", 56, 99)
s5 = Student("Jimmy", 16, 40)


# define course so previous class can fit in the 
# second class (course name, max students)
course = Course("Data Science", 2)

# add students above onto the course 
course.add_student(s1)
course.add_student(s2)

print(course.get_average_grade())

#########################################################################
