
# coding: utf-8

# ## Import libraries and data

# In[1]:


# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from re import sub
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from datetime import datetime
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# read data
df_sea_cal = pd.read_csv("data/seattle/calendar.csv")
df_sea_list = pd.read_csv("data/seattle/listings.csv")
df_sea_review = pd.read_csv("data/seattle/reviews.csv")


# ## Business question #1
# ## Which neighborhoods in Seattle are most popular among AirBnB visitors?
# 
# An intuitive way to measure the popularity of a neighborhood or district is to look at the vacancy rate of all listings in this area. If the vacancy rate is low, then it is safe to say that this neighborhood is popular. The confidence in the calculated vacancy rate is supported by an large number of listing in this neighborhood as well. Therefore, we will look at two set of data: one is the vacancy rate, the other is the count of listing in each neighborhood.

# ### Step 1.1 Data Clensing

# In[3]:


# drop any row of NaN in columns ["neighbourhood_cleansed"]
df_sea_neighborhood = df_sea_list.dropna(subset=["neighbourhood_cleansed"])
# drop rows where is_location_exact==false and has_availability==false 
df_sea_neighborhood = df_sea_neighborhood[df_sea_neighborhood["is_location_exact"]=='t']
df_sea_neighborhood = df_sea_neighborhood[df_sea_neighborhood["has_availability"]=='t']
print("dropped {} rows out of {} in total".format(df_sea_list.shape[0]-df_sea_neighborhood.shape[0],
                                                 df_sea_list.shape[0]))


# ### Step 1.2 Data Processing and Visulization
# 
# Group data by neighbourhood with number of listings and availability_365 respectively. Select Top 10 neighbourhoods in terms of number of listings and the reverse availability_365, visualize them to show which neighbourhoods are most populator among AirBnB visitors.

# In[4]:


# topk 10
TopK = 10

# topk neighborhoods in listing numbers
df_sea_neighborhood.neighbourhood_cleansed.value_counts()[:TopK].plot(kind='bar',
                                                                      title='Counts of Top '
                                                                      '{} listings in neighborhoods of '      
                                                                      'Seattle'.format(TopK))

# topk busiest neighborhoods in availability_365 and add a column of listing counts to the data frame
grouped_median_avail = df_sea_neighborhood.groupby(["neighbourhood_cleansed"])["availability_365"].median().reset_index()
grouped_median_avail = grouped_median_avail.sort_values('availability_365', ascending=True).head(TopK)
grouped_median_avail['list_counts'] = pd.Series([df_sea_neighborhood.neighbourhood_cleansed.value_counts().at[x] for x in grouped_median_avail["neighbourhood_cleansed"].tolist()],
                                                index=grouped_median_avail.index)

fig = plt.figure() # Create matplotlib figure

ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.

grouped_median_avail.plot(x='neighbourhood_cleansed', y='availability_365', width=0.3,
                          legend=False, kind='bar', color='red', ax=ax, position=1)
grouped_median_avail.plot(x='neighbourhood_cleansed', y='list_counts', width=0.3,    
                          legend=False, kind='bar', color='blue', ax=ax2, position=0)

ax.set_ylabel('availability_365', color='red')
ax.set_title('The Busiest {} Neighborhoods in A Year(Median availability_365)'.format(TopK))
ax2.set_ylabel('list_counts', color='blue')
    
plt.show()


# ### Step 1.3 Data Analysis
# 
# Because a popular neighborhood mean that its AirBnB host have one of the lowest available days in a year and this data is statistically meaningful with a large number of listings.

# In[5]:


# find the common neighborhood in both list
set1 = set(df_sea_neighborhood.neighbourhood_cleansed.value_counts()[:TopK].index.tolist())
set2 = set(grouped_median_avail.neighbourhood_cleansed.tolist())
print("Neighborhoods in both top {} list of most listed and busiest: {}".format(TopK, set1.intersection(set2)))


# Belltown and Lower Queen Anne are the answers
# 
# They are two adjacent neighborhoods, which are both located near downtown Seattle and the water front. Not surprised!

# ## Business question #2
# ## What is the busiest time of a year to visit Seattle?
# 
# One measure of how busy each time of a year is is to calculate the vacant ratio of all listings in Seattle. The lower the ratio is, the more visitors is visiting Seattle. For this metric, we will look at dataset we obtained before, df_sea_cal

# ### Step 2.1 Data Understanding

# In[6]:


df_sea_cal.head()


# In[7]:


df_sea_cal.tail()


# In[8]:


print("A snapshot of df_sea_cal")
print("Number of rows {}".format(df_sea_cal.shape[0]))
print("Column names: ", df_sea_cal.columns.tolist())
print("Number of listing_id", len(np.unique(df_sea_cal.listing_id)))
print("Number of date", len(np.unique(df_sea_cal.date)))


# It turns out that the availability and the price are recorded for each listing_id on each day b/w 2016-01-04 and 2017-01-02

# ### Step 2.2 Data Processing
# 
# Calculate the vacant ratio for each day during recorded period.

# In[9]:


# group df_sea_cal by date with column "available" being summed up by the number of available listings and then divided
# by the totol number of listing_id
df_sea_available_cal = df_sea_cal.groupby('date')['available'].agg(lambda x: sum(1 for i in x if i is 't')/len(x)).reset_index()
df_sea_available_cal.columns = ['date', 'vacant_ratio']


# ### Step 2.3 Data Visualization
# 
# Plot the trend of vacant ratio over dates of year 2016

# In[10]:


# define a function to plot trends of vacant rates
def plotTrendVacantRates(df, cols, legends):
    d = []
    for x in df.date.tolist():
        d.append(datetime.strptime(x, '%Y-%m-%d'))
        
    # set figure format
    days = mdates.DayLocator()
    months = mdates.MonthLocator()
    years = mdates.YearLocator()
    dfmt = mdates.DateFormatter('%b')

    datemin = d[0]
    datemax = d[-1]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(dfmt)
    ax.xaxis.set_minor_locator(days)
    ax.set_xlim(datemin, datemax)
    ax.set_ylabel('Vacant rate')
    ax.set_title("Trend of Vacant Rate")
    
    for col in cols:
        t = df_sea_available_cal[col].tolist()
        ax.plot(d, t, linewidth=2)
    
    ax.legend(legends)
    fig.set_size_inches(8, 4)


# In[11]:


# draw trends of vacant ratio throughout year 2016
plotTrendVacantRates(df_sea_available_cal, ['vacant_ratio'], ['all neighborhoods'])


# ### Step 2.4 Analysis
# 
# Based on the figure above, we would like to know which dates have a low vacant ratio(<0.6). These days with a low vacant ratio will be considered as most popular for visiting Seattle

# In[12]:


# print out the dates when the vacant ratio is below 0.6
df_sea_available_cal.query("vacant_ratio<0.6").date.tolist()


# January is the most popular!
# 
# Based on the vacant raito of AirBnB listing in 2016, the most busiest time period of visiting Seattle is the first month. This makes sense as people most likely goes on vacations after new year.

# ### Step 2.5 Extension
# 
# Could it be different for popular neighborhoods that we discovered previously?
# 
# Let's take a look at the vacant ratio throughout a year for those popular neighborhoods in Seattle.

# In[13]:


# get the time serie data for the most popular neighborhoods
neighborhood_ids = df_sea_neighborhood[df_sea_neighborhood["neighbourhood_cleansed"].isin(
    list(set1.intersection(set2)))].id.tolist()
df_sea_available_popular_cal = df_sea_cal[df_sea_cal.listing_id.isin(neighborhood_ids)].groupby(
    'date')['available'].agg(lambda x: sum(1 for i in x if i is 't')/len(x)).reset_index()
df_sea_available_cal['vacant_ratio_popular'] = df_sea_available_popular_cal['available']


# In[14]:


# draw trends of vacant ratio(for general and popular neighborhoods) throughout year 2016
plotTrendVacantRates(df_sea_available_cal, ['vacant_ratio', 'vacant_ratio_popular'],
                     ['all neighborhoods', 'most popular neighborhoods'])


# In[15]:


# print out the dates when the vacant ratio for popular neighborhood is below 0.51
df_sea_available_cal.query("vacant_ratio_popular<0.51").date.tolist()


# The first two or three weeks into the new year and summer times during July and August are looking popular for the most popular neighborhoods in Seattle. Overall, it still holds that the first two or three weeks after new year day are hottest times for visitors to Seattle.

# ## Business question #3
# 
# ## Which factors are most important in determining the pricing of AirBnB listings?
# 
# There are a couple of factors we think might be relavent to pricing of these listings based on common sense:
# * location
# * property type(house/apartment) and room type(entire house/private root)
# * size(bedroom number, bathroom number)
# * user reviews
# 
# We will look at dataset df_sea_list

# ### Step 3.1 Location
# 
# First, we will look at how location effect prices. To be statistically meaningful, compare the median price of all listings to those of listings in the top ten neighborhoods with most listings. We will notice the price difference among neighborhoods.

# #### Step 3.1.1 Data Preparation

# In[16]:


print("Number of missing values in column[price]:", df_sea_list.price.isnull().sum())
print("Number of missing values in column[neighbourhood_cleansed]:",
      df_sea_list.neighbourhood_cleansed.isnull().sum())


# In[17]:


# convert currency string to float in column[price]
df_sea_list_price = df_sea_list.copy()
df_sea_list_price['price'] = df_sea_list_price.price.apply(lambda i: float(sub(r'[^\d.]', '', i)))


# In[18]:


# print out the median price for all airbnb listings
print("Median price of all listings in Seattle: {}".format(
    df_sea_list_price.price.median()))


# #### Step 3.1.2 Data Processing

# In[19]:


# group df_sea_list_price by neighborhood with a median price for each group calculated
df_sea_list_price_neighborhood = df_sea_list_price.groupby('neighbourhood_cleansed')['price'].agg(
    np.median).reset_index()
df_sea_list_price_neighborhood.columns = ['neighbourhood_cleansed', 'median_price']


# #### Step 3.1.3 Data preparation

# In[20]:


# plot the median prices for top ten most listed neighborhoods
df_sea_list_price_neighborhood[df_sea_list_price_neighborhood.neighbourhood_cleansed.isin(
    list(set1))].sort_values(by='median_price', ascending=False).plot(
    x='neighbourhood_cleansed', y='median_price', kind='bar', legend=False,
    title="Median price among neighborhoods")
plt.show()


# #### Step 3.1.4 Analysis

# Neighborhoods like Belltown and Central Business District are priced significantly higher than the general median price, because they are either located near the downtown and waterfront, or at the heart of downtown Seattle. No surprise that they are more expensive and locations actually matter a lot in pricing AirBnB listings.

# ### Step 3.2 Property type & room type
# 
# We will look at the median price of listings grouped according to property_type and room_type. In reference, we will also display the listing numbers of each property_type and room_type pair

# #### Step 3.2.1 Data Understanding

# In[21]:


print("Number of missing values in column property_type:", df_sea_list_price.property_type.isnull().sum())
print("Number of missing values in column room_type:", df_sea_list_price.room_type.isnull().sum())


# The number of missing rows in these columns is only ONE. We could simply drop this row because it is statistically insignificant. Luckily function "groupby" will automatically excluse NaN values in these columns.

# #### Step 3.2.2 Data Processing

# In[22]:


# group df_sea_list_price by property_type and room_type with a median price and count calculated for listings in each group
df_sea_list_property_price = df_sea_list_price.groupby(
    ['property_type','room_type']).price.agg(np.median).reset_index()
df_sea_list_property_price['count'] = df_sea_list_price.groupby(
    ['property_type','room_type']).price.agg(lambda x: len(x)).reset_index()['price']

# only keep the topk most listed groups and rank it in descending order by price
df_sea_list_property_price = df_sea_list_property_price.sort_values(by='count', ascending=False)[:TopK]
df_sea_list_property_price = df_sea_list_property_price.sort_values(by='price', ascending=False)
df_sea_list_property_price


# #### Step 3.2.3 Analysis

# In the above chart, we can find out that listings of "House and Entire home/apt" will be priced highest among major categories, with "Apartment and Entire home/apt" ranked second. "Apartment and Shared room" and "House and Shared room" will be priced significantly lower than general median price, which is $100

# ### Step 3.3 Size
# 
# Three features that could be related with the size of listing:
# 1. number of bedrooms
# 2. number of bathrooms
# 3. square feet

# #### Step 3.3.1 Data Understanding

# In[23]:


print("Number of missing values in column bedrooms:", df_sea_list_price.bedrooms.isnull().sum())
print("Number of missing values in column bathrooms:", df_sea_list_price.bathrooms.isnull().sum())
print("Number of missing values in column square_feet: {} out of {}".format(
    df_sea_list_price.square_feet.isnull().sum(), len(df_sea_list_price)))


# The number of missing values in bedrooms and bathrooms is very small and we could simply drop those rows. On the other hand, the number of missing values in square_feet is very high, more than 90% of total rows, thus we should not use this feature for analysis at all.

# #### Step 3.3.2 Data Processing

# In[24]:


# group df_sea_list_price by columns ['bedrooms', 'bathrooms'] with median price and count calculated in each group
df_sea_list_size_price = df_sea_list_price.groupby(
    ['bedrooms', 'bathrooms']).price.agg(np.median).reset_index()

df_sea_list_size_price['count'] = df_sea_list_price.groupby(['bedrooms', 'bathrooms']).price.agg(
    lambda x: len(x)).reset_index()['price']

# only keep the topk most listed groups and rank them in descending order by price
df_sea_list_size_price = df_sea_list_size_price.sort_values(by='count', ascending=False)[:TopK]
df_sea_list_size_price = df_sea_list_size_price.sort_values(by='price', ascending=False)
df_sea_list_size_price


# #### Step 3.3.3 Analysis

# The general trend is that the more bedrooms and the more bathrooms we have, the more expensive we could expect from a listing.

# ### Step 3.4 User reviews
# 
# Explore features related with review scores and their relations to pricing. We will group the data by each feature and compare the median price in each group. If price is correlated with this feature, we will observe the price difference among groups.

# #### Step 3.4.1 Data Preparation

# There is a significant amount of missing values in these features, where there is no review scores attached to the listing. We will replace these NaN values with -1, as a way to indicate NaN values as pandas.groupby will automatically drop NaN.

# In[25]:


review_features = ["review_scores_accuracy", "review_scores_checkin", "review_scores_cleanliness",
                  "review_scores_communication", "review_scores_location", "review_scores_rating",
                  "review_scores_value"]


# In[26]:


for feature in review_features:
    print("Number of missing values in column{}: {} out of {}".format(
    feature, df_sea_list_price[feature].isnull().sum(), len(df_sea_list_price)))


# In[27]:


review_na_replace_dict = {k: -1 for k in review_features}
df_sea_list_review_price = df_sea_list_price.fillna(review_na_replace_dict)


# #### Step 3.4.2 Data Processing and Visualization

# In[28]:


def display_review_groups(df_review):
    for feature in review_features:
        df_review_group = df_review.groupby(
            feature).price.agg(np.median).reset_index()
        df_review_group['count'] = df_sea_list_review_price.groupby(
            feature).price.agg(lambda x: len(x)).reset_index()['price']
        df_review_group = df_review_group.sort_values(by=feature, ascending=True)

        fig = plt.figure() # Create matplotlib figure

        ax = fig.add_subplot(111) # Create matplotlib axes
        ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.

        df_review_group.plot(x=feature, y='price', legend=False, width=0.3, figsize=(9,6),
                             kind='bar', color='red', ax=ax, position=1)
        df_review_group.plot(x=feature, y='count', legend=False, width=0.3, figsize=(9,6),
                             kind='bar', color='blue', ax=ax2, position=0)

        ax.set_ylabel('median price', color='red')
        ax.set_title('Median price grouped by {}'.format(feature))
        ax2.set_ylabel('count', color='blue')

        plt.show()


# In[29]:


display_review_groups(df_sea_list_review_price)


# #### Step 3.4.3 Analysis

# From the charts above, we don't see a clear connection of review scores to pricing. Whether the score is 10, 9, or -1(NaN), the median price in each group does not distinguish itself from the rest.

# #### Step 3.4.4 Extension
# 
# Look at things from a difference angle. Group the data by "number of reviews" and explore if it's related with pricing.

# In[30]:


def partition_by_number_of_reviews(df, idx, interval=50):
    col = 'number_of_reviews'
    return int(df[col].loc[idx]/interval)*interval+interval/2


# In[31]:


def display_grouping_by_number_of_reviews(df_review, interval=50):
    feature = "number_of_reviews"
    df_review_group = df_review.groupby(
        lambda x: partition_by_number_of_reviews(
            df_sea_list_review_price, x, interval)).price.agg(
        np.median).reset_index()
    df_review_group['count'] = df_sea_list_review_price.groupby(
        lambda x: partition_by_number_of_reviews(
            df_sea_list_review_price, x, interval)).price.agg(
        lambda x: len(x)).reset_index()['price']
    df_review_group.columns = [feature, 'price', 'count']
    df_review_group = df_review_group.sort_values(by=feature, ascending=True)

    fig = plt.figure() # Create matplotlib figure

    ax = fig.add_subplot(111) # Create matplotlib axes
    ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.

    df_review_group.plot(x=feature, y='price', legend=False, width=0.3, figsize=(9,6),
                         kind='bar', color='red', ax=ax, position=1)
    df_review_group.plot(x=feature, y='count', legend=False, width=0.3, figsize=(9,6),                        
                         kind='bar', color='blue', ax=ax2, position=0)

    ax.set_ylabel('median price', color='red')
    ax.set_title('Median price grouped by {} with interval {}'.format(feature, interval))
    ax2.set_ylabel('count', color='blue')

    plt.show()


# In[32]:


display_grouping_by_number_of_reviews(df_sea_list_review_price, 10)


# Surprisingly, the general trend is that the more reviews a listing has, the less the median price is. This could make sense since customers are more likely to post a review on an AirBnB/hotle stay if they have a negative experience of it. More reviews a listing has, more negative impressions it might present to the public, thus lower it is priced later.

# ### Step 3.5 Random Forest
# 
# There are around 3,800 number of data points and we might easily get **overfitting** problems with linear regression. Choose **Random Forest Classifier** instead and it performs well with limited training data.

# #### Step 3.5.1 Data Preparation

# In[33]:


print("Number of missing values in column price:", df_sea_list.price.isnull().sum())


# In[34]:


# Split into explanatory and response variables
y = df_sea_list['price']
X = df_sea_list.drop('price', axis=1)


# ##### Step 3.5.1.1 Convert Numeric Strings to Numbers
# 
# There are some features ought to be converted into numbers from string. And before that, we need to strip the dollar signs and percentage signs from these strings.

# In[35]:


# convert numeric strings to number type
# we will drop 'weekly_price', 'monthly_price' as they are redundant compared to 'price'
features_to_converted_to_number = ['cleaning_fee', 'security_deposit',
                                   'host_acceptance_rate', 'host_response_rate', 'extra_people']

y = y.apply(lambda i: float(sub(r'[^\d.]', '', i)) if i is not np.nan else i)
X[features_to_converted_to_number] = X[features_to_converted_to_number].apply(
    lambda array: [float(sub(r'[^\d.]', '', i)) if i is not np.nan else i for i in array])
X[features_to_converted_to_number].head()


# ##### Step 3.5.1.2 Parse Phrases in Selected Features
# 
# Some features contains string that we could parse into phrases and obtain some useful information out of them. For example, in "amenities" we could get whether a TV or wireless internet is provided by the host. They could have a impact on pricing as well.

# In[36]:


# parse keywords in features in features_to_parse
features_to_parse = ['host_verifications', 'amenities']

def getPhrases(text):
    return re.compile('\w[\w\s\/\(\)\-\_]*').findall(text.lower())


# In[37]:


# example of phrases parsed from host_verifications
set.union(*X['host_verifications'].apply(
    lambda x: set(getPhrases(x)) if x is not np.nan else x).tolist())


# In[38]:


# function to parse phrases in selected features
def parseFeaturesByPhrases(df, feature_list):
    for feature in feature_list:
        # set set of phrases from feature
        df[feature] = df[feature].apply(lambda x: set(
            getPhrases(x)) if x is not np.nan else x)
        set_phrases = set.union(*df[feature].tolist())
        # add new columns from phrases
        list_phrases = list(set_phrases)
        list_addon_features = []
        for phrase in list_phrases:
            addon_feature = feature+'-'+phrase
            list_addon_features.append(addon_feature)
            df[addon_feature] = [0]*len(df)
        # assign 1 or 0 to each column based on availability of phrases
        df[[feature]+list_addon_features] = df[[feature]+list_addon_features].apply(
            lambda array: pd.Series([array[0]]+
                                    [1 if array[0] is not np.nan and x in array[0] else 0 for x in list_phrases]
                                   ), axis=1)
    # drop original column
    return df.drop(feature_list, axis=1, inplace=True)


# In[39]:


# perform parsing
parseFeaturesByPhrases(X, feature_list=features_to_parse)
X.head()


# ##### Step 3.5.1.3 Drop Redundant Features
# 
# Some features are redundant in predicting price. For example, weekly_price and monthly_price are already good indicators of price. For another example, listing id is directly related to the price of exising data, use it in the model will result in overfitting.

# In[40]:


# drop columns that are subjective desriptions, or redundant informations
cols_to_drop = ['weekly_price', 'monthly_price', 'first_review', 'last_review',
                'calendar_last_scraped', 'calendar_updated',
                'country', 'country_code', 'smart_location', 'market', 'state', 'city',
                'neighbourhood_group_cleansed', 'neighbourhood', 'street', 'host_neighbourhood', 
                'host_picture_url', 'host_thumbnail_url', 'host_about', 'host_location',
                'host_since', 'host_name', 'host_url', 'xl_picture_url', 'picture_url', 'medium_url',
                'thumbnail_url', 'transit', 'notes', 'neighborhood_overview', 'description',
                'space', 'summary', 'name', 'last_scraped', 'listing_url', 'zipcode',
                'calculated_host_listings_count', 'host_total_listings_count', 'host_listings_count',
                'id', 'scrape_id', 'host_id']

print("Number of redundant features to drop: {}".format(len(cols_to_drop)))
X = X.drop(cols_to_drop, axis=1)
print("Number of remaining features: {}".format(X.shape[1]))
X.head()


# ##### Step 3.5.1.4 One-Hot-Encode Categorical Features
# 
# Perform one-hot-encoding on categorical features. Before that, drop features with more than half of their values missing.

# In[41]:


# perform one-hot-encoding
cat_cols = X.select_dtypes(include=[object]).columns
X = pd.get_dummies(X, dummy_na=True,                       
                   columns=cat_cols, drop_first=True)
X.head()


# ##### Step 3.5.1.5 Imputation of Missing Values
# 
# Impute missing values by average values in each column.

# In[42]:


# drop columns with more than half of its value missing
X_imputed = X.dropna(axis=1, thresh=int(0.5*len(X)))
X_imputed.head()


# In[43]:


# perform imputation
fill_mean = lambda col: col.fillna(col.mean())
X_imputed = X_imputed.apply(fill_mean, axis=0)
X_imputed.head()


# #### Step 3.5.2 Random Forest
# 
# Run a Random Forest model on dataset and print out the top ten most important features in predicting price.

# In[44]:


# split testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=42)


# In[45]:


# train a random forest model with y_train and X_train
clf = RandomForestClassifier(n_estimators=400, max_depth=6,
                             random_state=42)

clf.fit(X_train, y_train)

#Predict using your model
y_test_preds = clf.predict(X_test)
y_train_preds = clf.predict(X_train)

#Score using your model
test_score = r2_score(y_test, y_test_preds)
train_score = r2_score(y_train, y_train_preds)
print("test_score: {}, train_score: {}".format(test_score, train_score))


# In[46]:


def importance_weights(coefficients, X_train):
    '''
    INPUT:
    coefficients - the coefficients of the linear model 
    X_train - the training data, so the column names can be used
    OUTPUT:
    coefs_df - a dataframe holding the coefficient, estimate, and abs(estimate)
    
    Provides a dataframe that can be used to understand the most influential coefficients
    in a linear model by providing the coefficient estimates along with the name of the 
    variable attached to the coefficient.
    '''
    importance_df = pd.DataFrame()
    importance_df['feature'] = X_train.columns
    importance_df['importance'] = coefficients
    importance_df = importance_df.sort_values('importance', ascending=False)
    return importance_df

#Use the function
importance_df = importance_weights(clf.feature_importances_, X_train)

#A quick look at the top results
importance_df.head(TopK)


# In[47]:


# draw a bar charts for most important features
top_importance_df = importance_df.head(TopK)

fig = plt.figure() # Create matplotlib figure

ax = fig.add_subplot(111) # Create matplotlib axes
top_importance_df.plot(x='feature', y='importance', legend=False, width=0.3, figsize=(9,6),                         
                       kind='bar', ax=ax, title="Top {} Most Important Features".format(TopK))
ax.set_ylabel("Importance Score")


# #### Step 3.5.3 Analysis

# Based on the chart above, the top ten most importent features in determining price of a airbnb listing is:
# * accommodates
# * bedrooms
# * cleaning_fee
# * room_type_Private room
# * reviews_per_month
# * latitude
# * longitude
# * number_of_reviews
# * bathrooms
# * room_type_Shared room
# 
# Among these features, "accommodates", "beds", "bathrooms" and "bedrooms" are directly related with the size of a listing, which is further direclty related with pricing. "room_type_Private room" and "room_type_Shared room" represent the quality of the stay. "longitude" and "latitude" are about the locations of a listing, the impact of which on pricing we explored in a previous section. "cleaning_fee" is also correlated with pricing as well since the higher the cleaning fee is the pricier a listing is.
# 
# In a previous secitn, we explored "number_of_reviews" and found it is somehow negtively correlated with price. We would expect similar result from "reviews_per_month". The negative correlation might make sense with the consideration that customers are more likely to write a review on a AirBnB stay if they have a negative experience of it.

# ## Conclusion

# We asked three business questions in the beginning:
# * Which neighborhoods in Seattle are most popular among AirBnB visitors?
# * What is the busiest time of a year to visit Seattle?
# * Which factors are most important in determining the pricing of AirBnB listings?
# 
# For the first question, we group the whole dataset by neighborhoods and explore the median available days of a year in each group. The most popular neighborhoods are those with lower availabilities. It turns out to be Belltown and Lower Queen Anne, which are both located near/at the downtown Seattle.
# 
# For the second question, we group the whole dataset by date and calculate the vacant rate at each date. It turns out that the first three weeks into the new year are the most busiest time of a year for Seattle.
# 
# For the third question, we first explore the impact of neighborhood, property/room type, size, reviews on pricing by grouping the dataset accordingly. Then we have the data cleansed and run random forest method on dataset. It turns out that the following features are most import ones in determining the price of a listing: accommodates, bedrooms, cleaning_fee, room_type_Private room, reviews_per_month, latitude, longitude, number_of_reviews, bathrooms, and room_type_Shared room. It makes sense since all of them are related to with at least one of the aspects regarding a list: location, quality/privacy of the room/property, size of the room/property. 
