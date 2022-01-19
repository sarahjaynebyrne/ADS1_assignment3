""" Assignment 3 """

# libraries  - general
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd

# libraries - statistics
from scipy import interpolate
from scipy import stats
from scipy.stats import kurtosis, skew
from scipy.stats import shapiro


import sklearn.cluster as cluster

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
            
            
# load in the mortality dataframe 
df1 = pd.read_csv('C:/Users/sjjby/Documents/Applied Data Science 1/Assignment 3/mortality.csv',
                  header = 2,
                  engine = 'python')            

# drop columns that are unrelevant
df1.drop(['Country Code', 'Indicator Name', 'Indicator Code', '2020', 'Unnamed: 65'],
         axis = 1,
         inplace = True)

# performing pre-processing on dataset
print(pre_processing(df1))


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


# doing the normalise_df function on dataset
print(normalised_df(df1))

df2 = normalised_df(df1)


""" Correlation between columns """

# correlation 1:
print(f'Correlation between columns (Years vs Years) \
      \n {df1.corr()}')

#correlation 2:

# copying the data
mort_df = df1.copy()

# transposing the data
mort_df.set_index('Country Name').transpose()

# resetting the index
mort_df.reset_index(level = 0,
                    inplace = True)

# the column had wrong name so have to rename
mort_df.rename(columns = {"index": "Year"},
               inplace = True)

print(f'Correlation between columns (Country vs Country) \
      \n {mort_df.corr()}')



""" Filtering the correlated dataset """

# filter to be not highly correlated or anti-correlated

# have to copy the data to remove categorical first column
fil_df = df1.copy()

# remove the categorical column 
fil_df.drop(['Country Name'],
            axis = 1,
            inplace = True)

# creating filtered dataset
filter_df1 = fil_df[(fil_df < 0.5) & (fil_df > -0.5)]

# removing rows with NA values
filter_df1.dropna(axis = 0,     # rows
                  how = 'any',  # any value need to be NA
                  inplace = True)

# exporting to xlsx file for eye inspection 
# filter_df1.to_excel('C:/Users/sjjby/Documents/Applied Data Science 1/Assignment 3/filtered_mortality_Data.xlsx')


""" Plotting combinations of columns """

'''
Producing plots of combinations of columns and finding
combinations which are suitable for clustering. 
Columns which are highly correlated (or anti-correlated) 
are not good for clustering.
'''

'''
for col in filter_df1.columns[1:]:
    filter_df1.plot(x = col, y = '1960',
                    kind = 'scatter',
                    subplots = True)
'''


""" Plotting Normalised 1960 and 2019 """

plt.plot(df2['2019'])
plt.title('2019')
plt.show()

plt.plot(df1['1969'])
plt.title('1960')
plt.show()


fig1, axs1 = plt.subplots(1,2,
                          figsize = (20, 15))

axs1[0].scatter(df2['Country Name'].index, 
                df2['2019'], 
                color = 'green')
axs1[1].scatter(df2['Country Name'].index, 
                df2['1960'], 
                color = 'blue')

plt.xlabel('Country Names')
plt.ylabel('Mortality Rate')
plt.show()


""" Performing clustering on the dataset """

'''
expecting LD countries to have higher mortality rates
than MD countries.
'''

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

""" Affinity CLustering """

ap = cluster.AffinityPropagation(max_iter = 1000, 
                                 verbose = True)

# selecting columns 
df_fit1 = df1[['2019', '1960']].copy()
df_fit1.dropna(inplace = True)

# runnig it
ap.fit(df_fit1)

# extract labels and centres
labels1 = ap.labels_
cen = ap.cluster_centers_

# plot using the labels to select colour
plt.figure(figsize=(5.0,5.0))

col = ["blue", "red", "green", "orange", "pink", "yellow", "black"]

# loop over the different labels
for l in range(0, 7): 
    plt.plot(df_fit1['2019'][labels1 == l], 
             df_fit1['1960'][labels1 == l], 
             "o", 
             markersize = 3, 
             color = col[l])

# show cluster centres
for ic in range(7):
    xc, yc = cen[ic,:]
    plt.plot(xc, yc, "dk", markersize=10)
    
plt.xlabel("2019")
plt.ylabel("1960")
plt.title("Affinity Clustering")
plt.show()


'''
le = LabelEncoder()
encoded = le.fit_transform(df1['Country Name'])

df1['encoded'] = encoded
'''



""" Inputting numerical cluster values """

kmeans = cluster.KMeans(n_clusters = 5)

# extract columns for fitting 
df_fit2 = df1[['2019', '1960']].copy()
df_fit2.dropna(inplace = True)

kmeans.fit(df_fit2)

# extract labels and cluster centres
labels2 = kmeans.labels_
cen = kmeans.cluster_centers_

# plot using the labels to select colour 
plt.figure(figsize = (5, 5))

col = ["blue", "red", "green", "magenta", "orange"]
for l in range(5):
    plt.plot(df_fit2['2019'][labels2 == l], 
             df_fit2['1960'][labels2 == l],
             "o",
             markersize = 3,
             color = col[l])
    
# show cluster centres
for ic in range(5):
    xc, yc = cen[ic, :]
    plt.plot(xc, yc,
             "dk",
             markersize = 10)
    
plt.xlabel("2019")
plt.title('Kmeans clustering')
plt.ylabel("1960")
plt.show()



""" Agglomerative clustering """

# do agglomerative etc 
ac = cluster.AgglomerativeClustering(n_clusters = 5)

# extract columns for fitting 
df_fit3 = df1[['2019', '1960']].copy()
df_fit3.dropna(inplace = True)

# carry out the fitting
ac.fit(df_fit3)

labels3 = ac.labels_

# the agglomerative clusteres does not return cluster centers
# so am making them :)
xcen = []
ycen = []

for ic in range(5):
    xc = np.average(df_fit3["2019"][labels3 == ic])
    yc = np.average(df_fit3["1960"][labels3 == ic])
    xcen.append(xc)
    ycen.append(yc)

# plot using the labels to select colour 
plt.figure(figsize = (5, 5))

col = ["blue", "red", "green", "magenta", "orange"]
for l in range(0, 5):
    plt.plot(df_fit3['2019'][labels3 == l], 
             df_fit3['1960'][labels3 == l],
             "o",
             markersize = 3,
             color = col[l])
    
# show cluster centres
for ic in range(5):
    plt.plot(xcen[ic], ycen[ic],
             "dk",
             markersize = 10)
    
plt.xlabel("2019")
plt.ylabel("1960")
plt.title('Agglomerative clustering')
plt.show()

# export to excel and compare which clustering technique 
# agreed/ disagreed the most 

""" Exporting clustering labels to excel column """

# making a new dataframe for ease of analysis
df_new = df1[['Country Name', '1960', '2019']]

# removing NA values to mirror the clustering algorithms
df_new.dropna(inplace = True)

# making new columns
df_new['affinity'] = labels1
df_new['kmeans'] = labels2
df_new['agglomerative'] = labels3

# export to excel
df_new.to_excel('C:/Users/sjjby/Documents/Applied Data Science 1/Assignment 3/output.xlsx', 
                engine = 'xlsxwriter')  

# then plot two clusters on one graph to see if they 
# do have the same shape as one another 

""" Filtering the data """






""" Predictive Modelling """

# use just LDC and MDC countries 









