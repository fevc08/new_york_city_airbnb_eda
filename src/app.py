from utils import db_connect
engine = db_connect()

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
##### Step 1: Use the following online dataset: https://raw.githubusercontent.com/4GeeksAcademy/data-preprocessing-project-tutorial/main/AB_NYC_2019.csv
# Read the dataset from the URL and assing it to a Pandas DataFrame
df_airbnb_ny = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/data-preprocessing-project-tutorial/main/AB_NYC_2019.csv')
df_airbnb_ny

##### Step 2: Find patterns and valuable information as much as you can. Make graphs that help us understand the patterns found, get some statistics, create new variables if needed, etc.
# Obtain the information of the dataset
df_airbnb_ny.info()


# We can assume that the columns 'id', 'name', 'host_name', 'last_review' and 'reviews_per_month' are not relevant for our analysis because there are missing values and they are not relevant for our analysis
# We are going to drop these columns from the dataset
df_airbnb_ny = df_airbnb_ny.drop(['id', 'name', 'host_name', 'last_review', 'reviews_per_month'], axis=1)
df_airbnb_ny.head()
# let's rename the column 'neighbourhood_group' to 'areas' and 'neighbourhood' to 'neighborhood' to make it easier to understand
df_airbnb_ny.rename(columns={'neighbourhood_group':'areas', 'neighbourhood':'neighborhood'}, inplace=True)
# We check again the information of the dataset
df_airbnb_ny.info()

##### Univariant analysis of the categorical variables
# Now, we have a dataset with 48,895 rows and 7 columns and we can start to explore the data
# Let's check the distribution of the data
df_airbnb_ny.describe(include='all')

# We can see this better with a counterplot of the categorical columns
fig, ax = plt.subplots(1, 3, figsize=(20, 5))

# Countplot for 'areas'
sns.countplot(data = df_airbnb_ny, x = 'areas', ax=ax[0], palette='vlag')
# Countplot for 'room_type'
sns.countplot(data = df_airbnb_ny, x = 'room_type', ax=ax[1], palette='vlag')
# Countplot for 'neighborhood' without xticks
sns.countplot(data = df_airbnb_ny, x = 'neighborhood', ax=ax[2], palette='vlag').set_xticklabels([])
plt.show()

# Now, we are going to check the distribution of the numerical columns with a histogram and a boxplot
fig, ax = plt.subplots(4, 3, figsize=(20, 20))

colors = sns.color_palette("vlag", 5)

# Histogram for 'host_id'
sns.histplot(data = df_airbnb_ny, x = 'host_id', ax=ax[0,0], color=colors[0])
# Boxplot for 'host_id'
sns.boxplot(data = df_airbnb_ny, x = 'host_id', ax=ax[1,0], color=colors[0])
# Histogram for 'price'
sns.histplot(data = df_airbnb_ny, x = 'price', ax=ax[0,1], color=colors[0])
# Boxplot for 'price'
sns.boxplot(data = df_airbnb_ny, x = 'price', ax=ax[1,1], color=colors[0])
# Histogram for 'minimum_nights'
sns.histplot(data = df_airbnb_ny, x = 'minimum_nights', ax=ax[0,2], color=colors[1])
# Boxplot for 'minimum_nights'
sns.boxplot(data = df_airbnb_ny, x = 'minimum_nights', ax=ax[1,2], color=colors[1])
# Histogram for 'number_of_reviews'
sns.histplot(data = df_airbnb_ny, x = 'number_of_reviews', ax=ax[2,0], color=colors[2])
# Boxplot for 'number_of_reviews'
sns.boxplot(data = df_airbnb_ny, x = 'number_of_reviews', ax=ax[3,0], color=colors[2])
# Histogram for 'calculated_host_listings_count'
sns.histplot(data = df_airbnb_ny, x = 'calculated_host_listings_count', ax=ax[2,1], color=colors[3])
# Boxplot for 'calculated_host_listings_count'
sns.boxplot(data = df_airbnb_ny, x = 'calculated_host_listings_count', ax=ax[3,1], color=colors[3])
# Histogram for 'availability_365'
sns.histplot(data = df_airbnb_ny, x = 'availability_365', ax=ax[2,2], color=colors[4])
# Boxplot for 'availability_365'
sns.boxplot(data = df_airbnb_ny, x = 'availability_365', ax=ax[3,2], color=colors[4])

plt.show()

##### Analysis of Multivariate Variables
# Relationship between 'price' and 'minimum_night', 'number_of_reviews', 'calculated_host_listing' and 'availability' with a scatterplot and a heatmap
fig, ax = plt.subplots(4, 2, figsize=(20, 20))

# Scatterplot for 'price' and 'minimum_nights'
sns.scatterplot(data = df_airbnb_ny, x = 'price', y = 'minimum_nights', ax=ax[0,0], color=colors[0])
# Heatmap for 'price' and 'minimum_nights'
sns.heatmap(data = df_airbnb_ny[['price', 'minimum_nights']].corr(), ax=ax[1,0], annot=True, cmap='vlag')
# Scatterplot for 'price' and 'number_of_reviews'
sns.scatterplot(data = df_airbnb_ny, x = 'price', y = 'number_of_reviews', ax=ax[0,1], color=colors[1])
# Heatmap for 'price' and 'number_of_reviews'
sns.heatmap(data = df_airbnb_ny[['price', 'number_of_reviews']].corr(), ax=ax[1,1], annot=True, cmap='vlag')
# Scatterplot for 'price' and 'calculated_host_listings_count'
sns.scatterplot(data = df_airbnb_ny, x = 'price', y = 'calculated_host_listings_count', ax=ax[2,0], color=colors[2])
# Heatmap for 'price' and 'calculated_host_listings_count'
sns.heatmap(data = df_airbnb_ny[['price', 'calculated_host_listings_count']].corr(), ax=ax[3,0], annot=True, cmap='vlag')
# Scatterplot for 'price' and 'availability_365'
sns.scatterplot(data = df_airbnb_ny, x = 'price', y = 'availability_365', ax=ax[2,1], color=colors[3])
# Heatmap for 'price' and 'availability_365'
sns.heatmap(data = df_airbnb_ny[['price', 'availability_365']].corr(), ax=ax[3,1], annot=True, cmap='vlag')

plt.show()
# Relationship between 'minimum_nights' and 'number_of_reviews', 'calculated_hast_listing_count' and 'availability_365' with a scatterplot and a heatmap
fig, ax = plt.subplots(2, 3, figsize=(20, 10))

# Scatterplot for 'minimum_nights' and 'number_of_reviews'
sns.scatterplot(data = df_airbnb_ny, x = 'minimum_nights', y = 'number_of_reviews', ax=ax[0,0], color=colors[0])
# Heatmap for 'minimum_nights' and 'number_of_reviews'
sns.heatmap(data = df_airbnb_ny[['minimum_nights', 'number_of_reviews']].corr(), ax=ax[1,0], annot=True, cmap='vlag')
# Scatterplot for 'minimum_nights' and 'calculated_host_listings_count'
sns.scatterplot(data = df_airbnb_ny, x = 'minimum_nights', y = 'calculated_host_listings_count', ax=ax[0,1], color=colors[0])
# Heatmap for 'minimum_nights' and 'calculated_host_listings_count'
sns.heatmap(data = df_airbnb_ny[['minimum_nights', 'calculated_host_listings_count']].corr(), ax=ax[1,1], annot=True, cmap='vlag')
# Scatterplot for 'minimum_nights' and 'availability_365'
sns.scatterplot(data = df_airbnb_ny, x = 'minimum_nights', y = 'availability_365', ax=ax[0,2], color=colors[0])
# Heatmap for 'minimum_nights' and 'availability_365'
sns.heatmap(data = df_airbnb_ny[['minimum_nights', 'availability_365']].corr(), ax=ax[1,2], annot=True, cmap='vlag')

plt.show()
# Relationship between 'number_of_reviews' and 'calculated_host_listings_count' and 'availability_365' with a scatterplot and a heatmap
fig, ax = plt.subplots(2, 2, figsize=(20, 10))

# Scatterplot for 'number_of_reviews' and 'calculated_host_listings_count'
sns.scatterplot(data = df_airbnb_ny, x = 'number_of_reviews', y = 'calculated_host_listings_count', ax=ax[0,0], color=colors[0])
# Heatmap for 'number_of_reviews' and 'calculated_host_listings_count'
sns.heatmap(data = df_airbnb_ny[['number_of_reviews', 'calculated_host_listings_count']].corr(), ax=ax[1,0], annot=True, cmap='vlag')
# Scatterplot for 'number_of_reviews' and 'availability_365'
sns.scatterplot(data = df_airbnb_ny, x = 'number_of_reviews', y = 'availability_365', ax=ax[0,1], color=colors[0])
# Heatmap for 'number_of_reviews' and 'availability_365'
sns.heatmap(data = df_airbnb_ny[['number_of_reviews', 'availability_365']].corr(), ax=ax[1,1], annot=True, cmap='vlag')

plt.show()
# Relationship between 'calculated_host_listings_count' and 'availability_365' with a scatterplot and a heatmap
fig, ax = plt.subplots(1, 2, figsize=(20, 5))

# Scatterplot for 'calculated_host_listings_count' and 'availability_365'
sns.scatterplot(data = df_airbnb_ny, x = 'calculated_host_listings_count', y = 'availability_365', ax=ax[0], color=colors[0])
# Heatmap for 'calculated_host_listings_count' and 'availability_365'
sns.heatmap(data = df_airbnb_ny[['calculated_host_listings_count', 'availability_365']].corr(), ax=ax[1], annot=True, cmap='vlag')

plt.show()
# Convert the categorical variables to numerical variables using the factorize method
df_airbnb_ny['areas_n'] = pd.factorize(df_airbnb_ny['areas'])[0]
df_airbnb_ny['neighborhood_n'] = pd.factorize(df_airbnb_ny['neighborhood'])[0]
df_airbnb_ny['room_type_n'] = pd.factorize(df_airbnb_ny['room_type'])[0]
df_airbnb_ny.head()
# Relationship between all the variables with a heatmap
fig, ax = plt.subplots(figsize=(20, 10))

# Heatmap for all the variables
sns.heatmap(data = df_airbnb_ny.corr(numeric_only = True), annot=True, cmap='vlag')

plt.show()
# Now, We are going to analyze all the data at the same time with a pairplot
sns.pairplot(data=df_airbnb_ny[['areas_n', 'neighborhood_n', 'room_type_n', 'price', 'minimum_nights', 'number_of_reviews', 'calculated_host_listings_count', 'availability_365']], palette='vlag')
plt.show()
##### Step 3: Feature engineering
# Once we have explored the data, we can start the feature engineering
# The first step is to analyze the outliers
# We are going to use the IQR method to detect the outliers
# We are going to use the columns 'price', 'minimum_nights', 'number_of_reviews', 'calculated_host_listings_count' and 'availability_365'

# Calculate the IQR for each column
Q1_price = df_airbnb_ny['price'].quantile(0.25)
Q3_price = df_airbnb_ny['price'].quantile(0.75)
IQR_price = Q3_price - Q1_price

Q1_minimum_nights = df_airbnb_ny['minimum_nights'].quantile(0.25)
Q3_minimum_nights = df_airbnb_ny['minimum_nights'].quantile(0.75)
IQR_minimum_nights = Q3_minimum_nights - Q1_minimum_nights

Q1_number_of_reviews = df_airbnb_ny['number_of_reviews'].quantile(0.25)
Q3_number_of_reviews = df_airbnb_ny['number_of_reviews'].quantile(0.75)
IQR_number_of_reviews = Q3_number_of_reviews - Q1_number_of_reviews

Q1_calculated_host_listings_count = df_airbnb_ny['calculated_host_listings_count'].quantile(0.25)
Q3_calculated_host_listings_count = df_airbnb_ny['calculated_host_listings_count'].quantile(0.75)
IQR_calculated_host_listings_count = Q3_calculated_host_listings_count - Q1_calculated_host_listings_count

Q1_availability_365 = df_airbnb_ny['availability_365'].quantile(0.25)
Q3_availability_365 = df_airbnb_ny['availability_365'].quantile(0.75)
IQR_availability_365 = Q3_availability_365 - Q1_availability_365

# Calculate the lower and upper limits for each column
lower_limit_price = Q1_price - 1.5 * IQR_price
upper_limit_price = Q3_price + 1.5 * IQR_price

lower_limit_minimum_nights = Q1_minimum_nights - 1.5 * IQR_minimum_nights
upper_limit_minimum_nights = Q3_minimum_nights + 1.5 * IQR_minimum_nights

lower_limit_number_of_reviews = Q1_number_of_reviews - 1.5 * IQR_number_of_reviews
upper_limit_number_of_reviews = Q3_number_of_reviews + 1.5 * IQR_number_of_reviews

lower_limit_calculated_host_listings_count = Q1_calculated_host_listings_count - 1.5 * IQR_calculated_host_listings_count
upper_limit_calculated_host_listings_count = Q3_calculated_host_listings_count + 1.5 * IQR_calculated_host_listings_count

lower_limit_availability_365 = Q1_availability_365 - 1.5 * IQR_availability_365
upper_limit_availability_365 = Q3_availability_365 + 1.5 * IQR_availability_365

# Calculate the number of outliers for each column
outliers_price = df_airbnb_ny[(df_airbnb_ny['price'] < lower_limit_price) | (df_airbnb_ny['price'] > upper_limit_price)]
outliers_minimum_nights = df_airbnb_ny[(df_airbnb_ny['minimum_nights'] < lower_limit_minimum_nights) | (df_airbnb_ny['minimum_nights'] > upper_limit_minimum_nights)]
outliers_number_of_reviews = df_airbnb_ny[(df_airbnb_ny['number_of_reviews'] < lower_limit_number_of_reviews) | (df_airbnb_ny['number_of_reviews'] > upper_limit_number_of_reviews)]
outliers_calculated_host_listings_count = df_airbnb_ny[(df_airbnb_ny['calculated_host_listings_count'] < lower_limit_calculated_host_listings_count) | (df_airbnb_ny['calculated_host_listings_count'] > upper_limit_calculated_host_listings_count)]
outliers_availability_365 = df_airbnb_ny[(df_airbnb_ny['availability_365'] < lower_limit_availability_365) | (df_airbnb_ny['availability_365'] > upper_limit_availability_365)]

# Calculate the percentage of outliers for each column
percentage_outliers_price = len(outliers_price) / len(df_airbnb_ny) * 100
percentage_outliers_minimum_nights = len(outliers_minimum_nights) / len(df_airbnb_ny) * 100
percentage_outliers_number_of_reviews = len(outliers_number_of_reviews) / len(df_airbnb_ny) * 100
percentage_outliers_calculated_host_listings_count = len(outliers_calculated_host_listings_count) / len(df_airbnb_ny) * 100
percentage_outliers_availability_365 = len(outliers_availability_365) / len(df_airbnb_ny) * 100

# Print the results
print(f'Percentage of outliers in the column "price": {percentage_outliers_price}%')
print(f'Percentage of outliers in the column "minimum_nights": {percentage_outliers_minimum_nights}%')
print(f'Percentage of outliers in the column "number_of_reviews": {percentage_outliers_number_of_reviews}%')
print(f'Percentage of outliers in the column "calculated_host_listings_count": {percentage_outliers_calculated_host_listings_count}%')
print(f'Percentage of outliers in the column "availability_365": {percentage_outliers_availability_365}%')

# We can see that the percentage of outliers in the columns 'price' and 'minimum_nights' is very high, so we are going to remove them from the dataset
# We are going to remove the rows with outliers in the columns 'price' and 'minimum_nights'
df_airbnb_ny = df_airbnb_ny[(df_airbnb_ny['price'] >= lower_limit_price) & (df_airbnb_ny['price'] <= upper_limit_price)]
df_airbnb_ny = df_airbnb_ny[(df_airbnb_ny['minimum_nights'] >= lower_limit_minimum_nights) & (df_airbnb_ny['minimum_nights'] <= upper_limit_minimum_nights)]
df_airbnb_ny
# Now, we are going to check the distribution of the data again
df_airbnb_ny.describe(include='all')
# Now, we are going to check the distribution of the numerical columns with a histogram and a boxplot
fig, ax = plt.subplots(4, 3, figsize=(20, 20))

colors = sns.color_palette("vlag", 5)

# Histogram for 'host_id'
sns.histplot(data = df_airbnb_ny, x = 'host_id', ax=ax[0,0], color=colors[0])
# Boxplot for 'host_id'
sns.boxplot(data = df_airbnb_ny, x = 'host_id', ax=ax[1,0], color=colors[0])
# Histogram for 'price'
sns.histplot(data = df_airbnb_ny, x = 'price', ax=ax[0,1], color=colors[0])
# Boxplot for 'price'
sns.boxplot(data = df_airbnb_ny, x = 'price', ax=ax[1,1], color=colors[0])
# Histogram for 'minimum_nights'
sns.histplot(data = df_airbnb_ny, x = 'minimum_nights', ax=ax[0,2], color=colors[1])
# Boxplot for 'minimum_nights'
sns.boxplot(data = df_airbnb_ny, x = 'minimum_nights', ax=ax[1,2], color=colors[1])
# Histogram for 'number_of_reviews'
sns.histplot(data = df_airbnb_ny, x = 'number_of_reviews', ax=ax[2,0], color=colors[2])
# Boxplot for 'number_of_reviews'
sns.boxplot(data = df_airbnb_ny, x = 'number_of_reviews', ax=ax[3,0], color=colors[2])
# Histogram for 'calculated_host_listings_count'
sns.histplot(data = df_airbnb_ny, x = 'calculated_host_listings_count', ax=ax[2,1], color=colors[3])
# Boxplot for 'calculated_host_listings_count'
sns.boxplot(data = df_airbnb_ny, x = 'calculated_host_listings_count', ax=ax[3,1], color=colors[3])
# Histogram for 'availability_365'
sns.histplot(data = df_airbnb_ny, x = 'availability_365', ax=ax[2,2], color=colors[4])
# Boxplot for 'availability_365'
sns.boxplot(data = df_airbnb_ny, x = 'availability_365', ax=ax[3,2], color=colors[4])

plt.show()
# For the feature scaling we are going to use sklearn and the MinMaxScaler method

# Create the scaler
scaler = MinMaxScaler()

# Scale the columns 'areas_n', 'neighborhood_n', 'room_type_n', 'price', 'minimum_nights', 'number_of_reviews', 'calculated_host_listings_count' and 'availability_365'
df_airbnb_ny[['areas_n', 'neighborhood_n', 'room_type_n', 'price', 'minimum_nights', 'number_of_reviews', 'calculated_host_listings_count', 'availability_365']] = scaler.fit_transform(df_airbnb_ny[['areas_n', 'neighborhood_n', 'room_type_n', 'price', 'minimum_nights', 'number_of_reviews', 'calculated_host_listings_count', 'availability_365']])
df_airbnb_ny
##### Step 4: Feature selection
# To finish the EDA we are going to do a feature selection
# We are going to use the columns 'areas_n', 'neighborhood_n', 'room_type_n', 'price', 'minimum_nights', 'number_of_reviews', 'calculated_host_listings_count' and 'availability_365' to predict the price of the Airbnb
# We are going to use the SelectKBest method from sklearn to select the best features

# Create the selector
selector = SelectKBest(score_func=f_regression, k=3)

# Fit the selector
selector.fit(df_airbnb_ny[['areas_n', 'neighborhood_n', 'room_type_n', 'minimum_nights', 'number_of_reviews', 'calculated_host_listings_count', 'availability_365']], df_airbnb_ny['price'])

# Obtain the scores
scores = selector.scores_
print(scores)

# Obtain the p-values
pvalues = selector.pvalues_
print(pvalues)

# Create a DataFrame with the scores and the p-values
df_scores = pd.DataFrame({'scores': scores, 'pvalues': pvalues}, index=['areas_n', 'neighborhood_n', 'room_type_n', 'minimum_nights', 'number_of_reviews', 'calculated_host_listings_count', 'availability_365'])

# Print the DataFrame
print(df_scores)
# We can see that the best features to predict the price of the Airbnb are 'areas_n', 'neighborhood_n' and 'room_type_n'
# We are going to use these features to predict the price of the Airbnb

# Save the raw dataset
df_airbnb_ny.to_csv('/workspace/new_york_city_airbnb_eda/data/raw/airbnb_ny.csv', index=False, if_exists='replace')

# Save the cleaned dataset
df_airbnb_ny.to_csv('/workspace/new_york_city_airbnb_eda/data/interim/airbnb_ny.csv', index=False, if_exists='replace')

# Save the dataset with the best features
df_airbnb_ny[['areas_n', 'neighborhood_n', 'room_type_n']].to_csv('/workspace/new_york_city_airbnb_eda/data/processed/airbnb_ny_features.csv', index=False, if_exists='replace')

# Save the dataset with the best features and the target
df_airbnb_ny[['areas_n', 'neighborhood_n', 'room_type_n', 'price']].to_csv('/workspace/new_york_city_airbnb_eda/data/processed/airbnb_ny_features_target.csv', index=False, if_exists='replace')
