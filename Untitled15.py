#!/usr/bin/env python
# coding: utf-8

# In[654]:


# basic libraries to work on the dataframe
import pandas as pd
import numpy as np
# data Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# libraries
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Suppressing Warnings
import warnings
warnings.filterwarnings('ignore')

#Increasing the columns views limit
pd.options.display.max_columns = None
pd.options.display.max_rows = 150
pd.options.display.float_format = '{:.2f}'.format


# In[655]:


#Process 1 Defining And Reading the Dataset


# In[656]:


#Reading the data file using pandas
df = pd.read_csv('/Users/b0288661/Downloads/Lead Scoring Assignment/Leads.csv')

df.head()


# In[657]:


# checking the shape of the dataset
df.shape


# In[658]:


# checking the statistics for numerical columns
df.describe()


# In[659]:


# checking whether there are any duplicates
df.duplicated().sum()


# In[660]:


#Lets have a look at all the columns, their datatypes and also get an idea of null values present
df.info()


# In[661]:


#Cleaning of DATA


# In[662]:


# change nomenclature to snakecase
df.columns = df.columns.str.replace(' ', '_').str.lower()

# test
df.columns


# In[663]:


# shorten column names
df.rename(columns = {'totalvisits': 'total_visits', 'total_time_spent_on_website': 'time_on_website', 
                    'how_did_you_hear_about_x_education': 'source', 'what_is_your_current_occupation': 'occupation',
                    'what_matters_most_to_you_in_choosing_a_course' : 'course_selection_reason', 
                    'receive_more_updates_about_our_courses': 'courses_updates', 
                     'update_me_on_supply_chain_content': 'supply_chain_content_updates',
                    'get_updates_on_dm_content': 'dm_content_updates',
                    'i_agree_to_pay_the_amount_through_cheque': 'cheque_payment',
                    'a_free_copy_of_mastering_the_interview': 'mastering_interview'}, inplace = True)

df.head(1)


# In[664]:


df.drop('prospect_id', axis = 1, inplace = True)


# In[665]:


# Select all non-numeric columns
df_obj = df.select_dtypes(include='object')

# Find out columns that have "Select"
s = lambda x: x.str.contains('Select', na=False)
l = df_obj.columns[df_obj.apply(s).any()].tolist()
print (l)


# In[666]:


# select all the columns that have a "Select" entry
sel_cols = ['specialization', 'source', 'lead_profile', 'city']

# replace values
df[sel_cols] = df[sel_cols].replace('Select', np.NaN)


# In[667]:


# Calculate percentage of null values for each column
(df.isnull().sum() / df.shape[0]) * 100


# In[668]:


df.drop(['source', 'lead_quality', 'lead_profile', 'asymmetrique_activity_index', 
                      'asymmetrique_profile_index', 'asymmetrique_activity_score', 'asymmetrique_profile_score',
        'tags', 'last_activity', 'last_notable_activity'], 
        axis = 1, inplace = True)

df.head(1)


# In[669]:


# Lets look at what are we left with
# Calculate percentage of null values for each column
(df.isnull().sum() / df.shape[0]) * 100


# In[670]:


df.country.value_counts(normalize = True, dropna = False) * 100


# In[671]:


df.drop('country', axis = 1, inplace = True)


# In[672]:


df.course_selection_reason.value_counts(normalize = True, dropna = False) * 100


# In[673]:


df.drop('course_selection_reason', axis = 1, inplace = True)


# In[674]:


df.occupation.value_counts(normalize = True, dropna = False) * 100


# In[675]:


# combine low representing categories
df.loc[(df.occupation == 'Student') | (df.occupation == 'Other') | (df.occupation == 'Housewife') | 
       (df.occupation == 'Businessman') , 'occupation'] = 'Student and Others'


# In[676]:


df.occupation.value_counts(normalize = True) * 100


# In[677]:


# impute proportionately
df['occupation'] = df.occupation.fillna(pd.Series(np.random.choice(['Unemployed', 'Working Professional', 
                                                                    'Student and Others'], 
                                                                   p = [0.8550, 0.1078, 0.0372], size = len(df))))


# In[678]:


df.specialization.value_counts(normalize = True, dropna = False) * 100


# In[679]:


# categorize all management courses
df.loc[(df.specialization == 'Finance Management') | (df.specialization == 'Human Resource Management') | 
       (df.specialization == 'Marketing Management') |  (df.specialization == 'Operations Management') |
       (df.specialization == 'IT Projects Management') | (df.specialization == 'Supply Chain Management') |
       (df.specialization == 'Healthcare Management') | (df.specialization == 'Hospitality Management') |
       (df.specialization == 'Retail Management') , 'specialization'] = 'Management Specializations'

# categorize all busines courses
df.loc[(df.specialization == 'Business Administration') | (df.specialization == 'International Business') | 
       (df.specialization == 'Rural and Agribusiness') | (df.specialization == 'E-Business') 
        , 'specialization'] = 'Business Specializations'

# categorize all industry courses
df.loc[(df.specialization == 'Banking, Investment And Insurance') | (df.specialization == 'Media and Advertising') |
       (df.specialization == 'Travel and Tourism') | (df.specialization == 'Services Excellence') |
       (df.specialization == 'E-COMMERCE'), 'specialization'] = 'Industry Specializations'


# In[680]:


df.specialization.value_counts(normalize = True) * 100


# In[681]:


# impute proportionately
df['specialization'] = df.specialization.fillna(pd.Series(np.random.choice(['Management Specializations',  
                                                    'Business Specializations', 'Industry Specializations'], 
                                                                   p = [0.7258, 0.1213, 0.1529 ], size = len(df))))


# In[682]:


df.city.value_counts(normalize = True, dropna = False) * 100


# In[683]:


# categorize all non-mumbai, but Maharashtra cities
df.loc[(df.city == 'Thane & Outskirts') | (df.city == 'Other Cities of Maharashtra'), 
       'city'] = 'Non-Mumbai Maharashtra Cities'

# categorize all other cities
df.loc[(df.city == 'Other Cities') | (df.city == 'Other Metro Cities') | (df.city == 'Tier II Cities') , 
       'city'] = 'Non-Maharashtra Cities'


# In[684]:


df.city.value_counts(normalize = True) * 100


# In[685]:


# impute proportionately
df['city'] = df.city.fillna(pd.Series(np.random.choice(['Mumbai', 'Non-Mumbai Maharashtra Cities', 
                                                                    'Non-Maharashtra Cities'], 
                                                                   p = [0.5784, 0.2170, 0.2046 ], size = len(df))))


# In[686]:


(df.isnull().sum() / df.shape[0]) * 100


# In[687]:


# determine unique values for all object datatype columns
for k, v in df.select_dtypes(include='object').nunique().to_dict().items():
    print('{} = {}'.format(k,v))


# In[688]:


df.lead_origin.value_counts(normalize = True, dropna = False) * 100


# In[689]:


#There are a lot of smaller values which will not be used as definitive factors, lets group them together
df.loc[(df.lead_origin == 'Lead Import') | (df.lead_origin == 'Quick Add Form') | (df.lead_origin == 'Lead Add Form')
       , 'lead_origin'] = 'Lead Add Form and Others'


# In[690]:


df.lead_source.value_counts(normalize = True, dropna = False) * 100


# In[691]:


# Lets impute the missing values with the mode of data i.e. clearly 'Google'
df.lead_source.fillna(df.lead_source.mode()[0], inplace=True)


# In[692]:


#There are a lot of smaller values which will not be used as definitive factors, lets group them together
df['lead_source'] = df['lead_source'].apply(lambda x: x if 
                                            ((x== 'Google') | (x=='Direct Traffic') | (x=='Olark Chat') | 
                                             (x=='Organic Search') | (x=='Reference')) 
                                            else 'Other Social Sites')


# In[693]:


# determine unique values
for k, v in df.select_dtypes(include='object').nunique().to_dict().items():
    print('{} = {}'.format(k,v))


# In[694]:


# select rest of the binary columns in a new dataframe
df_bin = df[['do_not_email', 'do_not_call', 'search', 'newspaper_article', 'x_education_forums', 
           'newspaper', 'digital_advertisement', 'through_recommendations', 'mastering_interview']]

# see value counts for each of the columns
for i in df_bin.columns:
    x = (df_bin[i].value_counts(normalize = True)) * 100
    print(x)
    print()


# In[695]:


drop_bin = ['do_not_call', 'search', 'newspaper_article', 'x_education_forums', 
           'newspaper', 'digital_advertisement', 'through_recommendations', 'magazine', 'courses_updates', 
           'supply_chain_content_updates', 'dm_content_updates', 'cheque_payment']

df.drop(drop_bin, axis = 1, inplace = True)


# In[696]:


df.lead_number = df.lead_number.astype('object')


# In[697]:


df.total_visits.fillna(df.total_visits.median(), inplace=True)
df.total_visits = df.total_visits.astype('int')


# In[698]:


df.page_views_per_visit.fillna(df.page_views_per_visit.median(), inplace=True)


# In[699]:


df.info()


# In[700]:


# Set style
plt.style.use('ggplot')

# See distribution of each of these columns
fig = plt.figure(figsize = (14, 10))
plt.subplot(2, 2, 1)
plt.hist(df.total_visits, bins = 20)
plt.title('Total website visits')

plt.subplot(2, 2, 2)
plt.hist(df.time_on_website, bins = 20)
plt.title('Time spent on website')

plt.subplot(2, 2, 3)
plt.hist(df.page_views_per_visit, bins = 20)
plt.title('Average number of page views per visit')

plt.show()


# In[701]:


plt.figure(figsize = (14,12))
sns.heatmap(df[['total_visits', 'time_on_website', 'page_views_per_visit']].corr(), cmap="YlGnBu", annot = True)
plt.show()


# In[702]:


plt.figure(figsize = (10, 14))

plt.subplot(3,1,1)
sns.boxplot(df.total_visits)

plt.subplot(3,1,2)
sns.boxplot(df.time_on_website)

plt.subplot(3,1,3)
sns.boxplot(df.page_views_per_visit)
plt.show()


# In[703]:


plt.figure(figsize = (14, 8))

df.groupby('lead_origin')['lead_number'].count().sort_values(ascending = False).plot(kind= 'barh', width = 0.8, 
                                                            edgecolor = 'black', 
                                                            color = plt.cm.Paired(np.arange(len(df))))
plt.show()


# In[704]:


df.head(1)


# In[705]:


plt.figure(figsize = (14, 8))

df.groupby('lead_source')['lead_number'].count().sort_values(ascending = False).plot(kind= 'barh', width = 0.8, 
                                                            edgecolor = 'black', 
                                                            color = plt.cm.Paired(np.arange(len(df))))
plt.show()


# In[706]:


plt.figure(figsize = (10, 8))

df.groupby('specialization')['lead_number'].count().sort_values(ascending = False).plot(kind= 'barh', width = 0.8, 
                                                            edgecolor = 'black', 
                                                            color = plt.cm.Paired(np.arange(len(df))))
plt.show()


# In[707]:


plt.figure(figsize = (14, 8))

df.groupby('occupation')['lead_number'].count().sort_values(ascending = False).plot(kind= 'barh', width = 0.8, 
                                                            edgecolor = 'black', 
                                                            color = plt.cm.Paired(np.arange(len(df))))
plt.show()


# In[708]:


plt.figure(figsize = (14, 8))

df.groupby('city')['lead_number'].count().sort_values(ascending = False).plot(kind= 'barh', width = 0.8, 
                                                            edgecolor = 'black', 
                                                            color = plt.cm.Paired(np.arange(len(df))))
plt.show()


# In[709]:


plt.figure(figsize = (14, 8))

df.groupby('do_not_email')['lead_number'].count().sort_values(ascending = False).plot(kind= 'barh', width = 0.8, 
                                                            edgecolor = 'black', 
                                                            color = plt.cm.Paired(np.arange(len(df))))
plt.show()


# In[710]:


# determine unique values
for k, v in df.select_dtypes(include='object').nunique().to_dict().items():
    print('{} = {}'.format(k,v))


# In[711]:


binlist = ['do_not_email', 'mastering_interview']

# Defining the map function
def binary_map(x):
    return x.map({'Yes': 1, "No": 0})

# Applying the function to the housing list
df[binlist] = df[binlist].apply(binary_map)

# check the operation was success
df.head()


# In[712]:


# Creating a dummy variable for some of the categorical variables and dropping the first one.
dummy1 = pd.get_dummies(df[['lead_origin', 'lead_source', 'specialization', 'occupation', 'city']], drop_first = True)

# Adding the results to the master dataframe
df = pd.concat([df, dummy1], axis=1)


# In[713]:


# Dropping the columns for which dummies have been created
df.drop(['lead_origin', 'lead_source', 'specialization', 'occupation', 'city'], axis = 1, inplace = True)

df.head()


# In[714]:


num_cols = df[['total_visits', 'time_on_website', 'page_views_per_visit']]

# Checking outliers at 25%, 50%, 75%, 90%, 95% and 99%
num_cols.describe(percentiles=[.25, .5, .75, .90, .95, .99])


# In[715]:


# capping at 99 percentile
df.total_visits.loc[df.total_visits >= df.total_visits.quantile(0.99)] = df.total_visits.quantile(0.99)
df.page_views_per_visit.loc[df.page_views_per_visit >= 
                            df.page_views_per_visit.quantile(0.99)] = df.page_views_per_visit.quantile(0.99)


# In[716]:


plt.figure(figsize = (10, 14))

plt.subplot(2,1,1)
sns.boxplot(df.total_visits)

plt.subplot(2,1,2)
sns.boxplot(df.page_views_per_visit)
plt.show()


# In[717]:


# Putting feature variable to X
X = df.drop(['lead_number', 'converted'], axis=1)

X.head(1)


# In[718]:


# Putting response variable to y
y = df['converted']

y.head(1)


# In[719]:


# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# In[720]:


scaler = StandardScaler()

X_train[['total_visits','time_on_website','page_views_per_visit']] = scaler.fit_transform(
    X_train[['total_visits','time_on_website','page_views_per_visit']])

X_train.head()


# In[721]:


# checking the conversion rate
conversion = (sum(df['converted'])/len(df['converted'].index))*100
conversion


# In[722]:


# Let's see the correlation matrix 
plt.figure(figsize = (14,10))       
sns.heatmap(df.corr(),annot = True, cmap="YlGnBu")
plt.show()


# In[723]:


X_test.drop(['lead_origin_Lead Add Form and Others', 'specialization_Industry Specializations', 
                     'occupation_Working Professional'], axis = 1, inplace = True)

X_train.drop(['lead_origin_Lead Add Form and Others', 'specialization_Industry Specializations', 
                     'occupation_Working Professional'], axis = 1, inplace = True)


# In[724]:


## lets check the correlation matrix again
plt.figure(figsize = (14,10))       
sns.heatmap(X_train.corr(),annot = True, cmap="YlGnBu")
plt.show()


# In[725]:


# Logistic regression model
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()


# In[726]:


# initiate logistic regression
logreg = LogisticRegression()

# initiate rfe
rfe = RFE(logreg, n_features_to_select=13)             # running RFE with 13 variables as output
rfe = rfe.fit(X_train, y_train)


# In[727]:


rfe.support_


# In[728]:


list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[729]:


# assign columns
col = X_train.columns[rfe.support_]


# In[730]:


# check what columns were not selected by RFE
X_train.columns[~rfe.support_]


# In[731]:


X_train_sm = sm.add_constant(X_train[col])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[ ]:


#Extra


# In[855]:


#Extra


# In[ ]:


#Extra


# In[ ]:


#Extra


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[799]:


#Lets create empty lists of categorical columns and numerical columns, then we will review the columns one by one and see what needs to be done with each of them
cat= []
num = []
#Reading the data file using pandas
df = pd.read_csv('/Users/b0288661/Downloads/Lead Scoring Assignment/Leads.csv')

df.head()


# In[800]:


#Creating a function to get the column details
def details(x):
    print(df[x].value_counts())
    print(df[x].isnull().sum(),'null values')
    print(df[x].isnull().sum()/df.count()['Lead Number']*100,'% values are null')


# In[801]:


df.head()


# In[802]:


# Do not email column
details('Do Not Email')


# In[803]:


bi_cat = []
bi_cat.append('Do Not Email')


# In[804]:


#Do Not Call column
details('Do Not Call')


# In[805]:


#Lets drop the column since its only two records and it doesn't make sense to keep this column
df.drop('Do Not Call',axis=1,inplace = True)
#df[['Do Not Call','Converted']].value_counts()


# In[806]:


df[df['total_visits'].isnull()]['total_visits'] = df['total_visits'].mode()[0]
#df[df.TotalVisits > df.TotalVisits.quantile(0.96)] = df.TotalVisits.quantile(0.96)
#Also add this column to our numerical columns list
num.append('total_visits')


# In[807]:


details('TotalVisits')


# In[808]:


# We can see that there is a lot of outliers here, we can also plot a boxplot to get a visual idea of the same
sns.boxplot(df['TotalVisits'])
#and see the median(sometimes we impute the null values with median), and 95th to 99th percentiles for the data
df.TotalVisits.quantile([0.50,0.95,0.96,0.97,0.98,0.99])


# In[809]:


df[df['TotalVisits'].isnull()]['TotalVisits'] = df['TotalVisits'].mode()[0]
#df[df.TotalVisits > df.TotalVisits.quantile(0.96)] = df.TotalVisits.quantile(0.96)
#Also add this column to our numerical columns list
num.append('TotalVisits')


# In[810]:


def cap(col,typ='right',value=0.95):
    if typ == 'left':
        df[df[col]<df[col].quantile(value)][col] = df[col].quantile(value)
    else:
        df[df[col]>df[col].quantile(value)][col] = df[col].quantile(value)
        


# In[811]:


# and capping the column as mentioned earlier
cap('TotalVisits')


# In[812]:


# Lets look at Total Time Spent on Website column
details('Total Time Spent on Website')


# In[813]:


def num_details(x):
    print(df[x].value_counts())
    print(df[x].isnull().sum(),'null values')
    print(df[x].isnull().sum()/df.count()[x]*100,'% values are null')
    print('Percentiles are as follows')
    print(df[x].quantile([0.50,0.95,0.96,0.97,0.98,0.99]))
    sns.boxplot(df[x])


# In[814]:


#Lets look at column Page Views Per Visit
num_details('Page Views Per Visit')


# In[815]:


df[df['Page Views Per Visit'].isnull()]['Page Views Per Visit'] = df['Page Views Per Visit'].mode()[0]
cap('Page Views Per Visit')
#Also add this column to our numerical columns list
num.append('Page Views Per Visit')


# In[816]:


df.head()


# In[817]:


details('Last Activity')


# In[818]:


#Lets have a look at the column 'Country'
details('Country')


# In[819]:


df.drop('Country', axis =1, inplace = True)


# In[820]:


details('Specialization')


# In[821]:


df.drop('mastering_interview', axis =1, inplace = True)


# In[822]:


details('Specialization')


# In[823]:


details('How did you hear about X Education')


# In[824]:


df.drop('How did you hear about X Education', axis =1, inplace = True)


# In[825]:


details('What is your current occupation')


# In[826]:


details('What matters most to you in choosing a course')


# In[827]:


details('Search')


# In[828]:


details('Magazine')


# In[829]:


details('Newspaper Article')


# In[830]:


details('X Education Forums')


# In[831]:


details('Newspaper')


# In[832]:


details('Digital Advertisement')


# In[833]:


#Lets look at the column 'Through Recommendation'
details('Through Recommendations')


# In[834]:


details('Receive More Updates About Our Courses')


# In[835]:


df.drop(['Search','Magazine','Newspaper Article','X Education Forums','Newspaper','Digital Advertisement','Through Recommendations','Receive More Updates About Our Courses'],axis=1,inplace = True)


# In[836]:


details('Tags')


# In[837]:


details('Lead Quality')


# In[838]:


details('Update me on Supply Chain Content')


# In[839]:


# There's only singular value in this, so lets drop this column
df.drop('Update me on Supply Chain Content', axis=1, inplace = True)


# In[840]:


details('Get updates on DM Content')


# In[841]:


# There's only singular value in this, so lets drop this column
df.drop('Get updates on DM Content', axis=1, inplace = True)


# In[842]:


details('Lead Profile')


# In[843]:


details('City')


# In[844]:


details('Asymmetrique Activity Index')


# In[845]:


details('Asymmetrique Profile Index')


# In[846]:


details('Asymmetrique Activity Score')


# In[847]:


details('Asymmetrique Profile Score')


# In[848]:


details('I agree to pay the amount through cheque')


# In[849]:


# There's only singular value in this, so lets drop this column
df.drop('I agree to pay the amount through cheque', axis=1, inplace = True)


# In[850]:


details('A free copy of Mastering The Interview')


# In[851]:


bi_cat.append('A free copy of Mastering The Interview')


# In[852]:


details('Last Notable Activity')


# In[853]:


#Lets have a look at our three categories of column
print(cat)
print(num)
print(bi_cat)


# In[854]:


df.head(1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




