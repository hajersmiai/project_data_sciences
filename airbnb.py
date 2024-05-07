import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

folder= Path(__file__).parent

data = pd.read_csv(Path(folder,"Airbnb_Data.csv"))

df = pd.DataFrame(data)

#%%  distribution des prix selon le type de logement
df_app = pd.DataFrame(df[df['property_type']=='Apartment'])
print (df_app)
df_other=pd.DataFrame(df[df['property_type']=='House'])
print (df_other)
df_other=pd.DataFrame(df[(df['property_type']!='House') & (df['property_type']!='Apartment' )])
print (df_other)
#%% Histogram of log prices
sns.histplot(df['log_price'], bins=30, kde=True)
plt.title('Distribution of Logement Prices')
plt.xlabel('Logement Price')
plt.ylabel('Frequency')
plt.show()


sns.histplot(df_app['log_price'], bins=30, kde=True)
# plt.plot(df_app['log_price'] )
plt.title('Distribution of Appartement Prices')
plt.xlabel('Appartement Price')
plt.ylabel('Frequency')
plt.show()


plt.plot(df_other['log_price'] )
sns.histplot(df_other['log_price'], bins=30, kde=True)
plt.title('Distribution of House Prices')
plt.xlabel('House Price')
plt.ylabel('Frequency')
plt.show()

sns.histplot(df_other['log_price'], bins=30, kde=True)
# plt.plot(df_other['log_price'] )
plt.title('Distribution of Other Type of Logement Prices')
plt.xlabel('Other Price')
plt.ylabel('Frequency')
plt.show()
# %%
# Box plot of log prices
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['log_price'])
plt.title('Box Plot of Log Prices')
plt.xlabel('Log Price')
plt.show()
plt.figure(figsize=(10, 6))
sns.boxplot(x=df_app['log_price'])
plt.title('Box Plot of Appartment Prices')
plt.xlabel('Log Price')
plt.show()
plt.figure(figsize=(10, 6))
sns.boxplot(x=df_other['log_price'])
plt.title('Box Plot of House Prices')
plt.xlabel('Log Price')
plt.show()
plt.figure(figsize=(10, 6))
sns.boxplot(x=df_other['log_price'])
plt.title('Box Plot of Other Prices')
plt.xlabel('Log Price')
plt.show()

# %%
# Exploring the relationship between price and property features using scatter plots and correlation matrices
# Selecting numerical columns that are relevant to the property features
numerical_features = df[['log_price', 'accommodates', 'bathrooms', 'bedrooms', 'beds']]
correlation_matrix = numerical_features.corr()
# Displaying the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.05)
plt.title('Correlation Matrix of Numerical Property Features with Log Price')
plt.show()
# Scatter plots to visualize the relationship between log_price and other numerical property features
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
sns.scatterplot(ax=axes[0, 0], data=df, x='accommodates', y='log_price')
axes[0, 0].set_title('Log Price vs Accommodates')

sns.scatterplot(ax=axes[0, 1], data=df, x='bathrooms', y='log_price')
axes[0, 1].set_title('Log Price vs Bathrooms')

sns.scatterplot(ax=axes[1, 0], data=df, x='bedrooms', y='log_price')
axes[1, 0].set_title('Log Price vs Bedrooms')

sns.scatterplot(ax=axes[1, 1], data=df, x='beds', y='log_price')
axes[1, 1].set_xticks(np.arange(min(data['beds']), max(data['beds'])+1, 1))

plt.tight_layout()
plt.show()
# for apartment
numerical_features = df_app[['log_price', 'accommodates', 'bathrooms', 'bedrooms', 'beds']]
correlation_matrix = numerical_features.corr()
# Displaying the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.05)
plt.title('Correlation Matrix of Numerical Property Features with Apartment Price')
plt.show()
# Scatter plots to visualize the relationship between log_price and other numerical property features
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
sns.scatterplot(ax=axes[0, 0], data=df_app, x='accommodates', y='log_price')
axes[0, 0].set_title('Apartment Price vs Accommodates')

sns.scatterplot(ax=axes[0, 1], data=df_app, x='bathrooms', y='log_price')
axes[0, 1].set_title('Apartment Price vs Bathrooms')

sns.scatterplot(ax=axes[1, 0], data=df_app, x='bedrooms', y='log_price')
axes[1, 0].set_title('Apartment Price vs Bedrooms')

sns.scatterplot(ax=axes[1, 1], data=df_app, x='beds', y='log_price')
axes[1, 1].set_xticks(np.arange(min(data['beds']), max(data['beds'])+1, 1))

plt.tight_layout()
plt.show()
# for house
numerical_features = df_other[['log_price', 'accommodates', 'bathrooms', 'bedrooms', 'beds']]
correlation_matrix = numerical_features.corr()
# Displaying the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.05)
plt.title('Correlation Matrix of Numerical Property Features with House Price')
plt.show()
# Scatter plots to visualize the relationship between log_price and other numerical property features
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
sns.scatterplot(ax=axes[0, 0], data=df_other, x='accommodates', y='log_price')
axes[0, 0].set_title('House Price vs Accommodates')

sns.scatterplot(ax=axes[0, 1], data=df_other, x='bathrooms', y='log_price')
axes[0, 1].set_title('House Price vs Bathrooms')

sns.scatterplot(ax=axes[1, 0], data=df_other, x='bedrooms', y='log_price')
axes[1, 0].set_title('House Price vs Bedrooms')

sns.scatterplot(ax=axes[1, 1], data=df_other, x='beds', y='log_price')
axes[1, 1].set_xticks(np.arange(min(data['beds']), max(data['beds'])+1, 1))

plt.tight_layout()
plt.show()
# for other
numerical_features = df_other[['log_price', 'accommodates', 'bathrooms', 'bedrooms', 'beds']]
correlation_matrix = numerical_features.corr()
# Displaying the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.05)
plt.title('Correlation Matrix of Numerical Property Features with Other Price')
plt.show()
# Scatter plots to visualize the relationship between log_price and other numerical property features
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
sns.scatterplot(ax=axes[0, 0], data=df_other, x='accommodates', y='log_price')
axes[0, 0].set_title('Other Price vs Accommodates')

sns.scatterplot(ax=axes[0, 1], data=df_other, x='bathrooms', y='log_price')
axes[0, 1].set_title('Other Price vs Bathrooms')

sns.scatterplot(ax=axes[1, 0], data=df_other, x='bedrooms', y='log_price')
axes[1, 0].set_title('Other Price vs Bedrooms')

sns.scatterplot(ax=axes[1, 1], data=df_other, x='beds', y='log_price')
axes[1, 1].set_xticks(np.arange(min(data['beds']), max(data['beds'])+1, 1))

plt.tight_layout()
plt.show()
# %%
# Investigate price variation across different cities or neighborhoods
plt.figure(figsize=(12, 8))
sns.boxplot(x='city', y='log_price', data=df)
plt.title('Log Price Distribution Across Different Cities')
plt.xlabel('City')
plt.ylabel('Log Price')
plt.show()

# Analyzing price variation across neighborhoods for a specific city (e.g., New York City)
nyc_data = df[df['city'] == 'Boston']
top_neighbourhoods = nyc_data['neighbourhood'].value_counts().head(10).index
nyc_top_neighbourhoods_data = nyc_data[nyc_data['neighbourhood'].isin(top_neighbourhoods)]

plt.figure(figsize=(14, 8))
sns.boxplot(x='neighbourhood', y='log_price', data=nyc_top_neighbourhoods_data)
plt.title('Log Price Distribution Across Top 10 Neighborhoods in Boston')
plt.xlabel('Neighborhood')
plt.ylabel('Log Price')
plt.xticks(rotation=45)
plt.show()

# %%
# Analyzing the distribution and popularity of property types and room types across various locations

# Setting the aesthetic style of the plots
sns.set(style="whitegrid")

# Distribution of Property Types
plt.figure(figsize=(12, 6))
sns.countplot(data=df, y='property_type', order=df['property_type'].value_counts().index)
plt.title('Distribution of Property Types')
plt.xlabel('Count')
plt.ylabel('Property Type')
plt.show()

# Distribution of Room Types
plt.figure(figsize=(8, 4))
sns.countplot(data=df, x='room_type', order=df['room_type'].value_counts().index)
plt.title('Distribution of Room Types')
plt.xlabel('Room Type')
plt.ylabel('Count')
plt.show()

# Popularity of Property Types by City
plt.figure(figsize=(12, 6))
sns.countplot(data=df, y='property_type', hue='city', order=df['property_type'].value_counts().index)
plt.title('Popularity of Property Types by City')
plt.xlabel('Count')
plt.ylabel('Property Type')
plt.legend(title='City')
plt.show()

# Popularity of Room Types by City
plt.figure(figsize=(8, 4))
sns.countplot(data=df, x='room_type', hue='city', order=df['room_type'].value_counts().index)
plt.title('Popularity of Room Types by City')
plt.xlabel('Room Type')
plt.ylabel('Count')
plt.legend(title='City')
plt.show()

# Average Prices for Different Property Types
plt.figure(figsize=(12, 6))
sns.barplot(data=df, y='property_type', x='log_price', estimator=np.mean, ci=None, order=data.groupby('property_type')['log_price'].mean().sort_values(ascending=False).index)
plt.title('Average Log Prices for Different Property Types')
plt.xlabel('Average Log Price')
plt.ylabel('Property Type')
plt.show()

# Average Prices for Different Room Types
plt.figure(figsize=(8, 4))
sns.barplot(data=df, x='room_type', y='log_price', estimator=np.mean, ci=None, order=data.groupby('room_type')['log_price'].mean().sort_values(ascending=False).index)
plt.title('Average Log Prices for Different Room Types')
plt.xlabel('Room Type')
plt.ylabel('Average Log Price')
plt.show()
# %%
# Analyzing the distribution of listings across different cities and neighborhoods
plt.figure(figsize=(12, 8))
sns.countplot(data=df, y='city', order=df['city'].value_counts().index)
plt.title('Distribution of Listings Across Different Cities')
plt.xlabel('Count')
plt.ylabel('City')
plt.show()

# Analyzing the distribution of listings in neighborhoods for the top cities
top_cities = df['city'].value_counts().index
fig, axes = plt.subplots(len(top_cities), 1, figsize=(12, 20), sharex=True)

for i, city in enumerate(top_cities):
    city_data = df[df['city'] == city]
    sns.countplot(ax=axes[i], data=city_data, y='neighbourhood', order=city_data['neighbourhood'].value_counts().head(10).index)
    axes[i].set_title(f'Distribution of Listings in Top 10 Neighborhoods of {city}')
    axes[i].set_xlabel('Count')
    axes[i].set_ylabel('Neighborhood')

plt.tight_layout()
plt.show()
# %%
# Analyzing the impact of different amenities on listing prices and review scores
from sklearn.preprocessing import MultiLabelBinarizer

# Check if amenities are already processed into lists, if not, convert them
if isinstance(df['amenities'].iloc[0], str):
    df['amenities'] = df['amenities'].apply(lambda x: x.strip('{}').replace('"', '').split(','))

# Use MultiLabelBinarizer to transform the amenities into a binary format
mlb = MultiLabelBinarizer()
amenities_encoded = mlb.fit_transform(df['amenities'])
amenities_df = pd.DataFrame(amenities_encoded, columns=mlb.classes_)

# Add the encoded amenities back to the main dataframe
data_amenities = pd.concat([df.reset_index(drop=True), amenities_df], axis=1)

# Select only numeric columns for correlation analysis
numeric_cols = data_amenities.select_dtypes(include=[np.number])

# Analyze the impact of amenities on log_price
amenities_price_impact = numeric_cols.drop(columns=['id', 'log_price']).corrwith(data_amenities['log_price']).sort_values(ascending=False)
plt.figure(figsize=(10, 8))
sns.barplot(x=amenities_price_impact.values, y=amenities_price_impact.index)
plt.title('Impact of Amenities on Log Price')
plt.xlabel('Correlation with Log Price')
plt.ylabel('Amenities')
plt.show()

# Analyze the impact of amenities on review_scores_rating
amenities_review_impact = numeric_cols.drop(columns=['id', 'review_scores_rating']).corrwith(data_amenities['review_scores_rating']).sort_values(ascending=False)
plt.figure(figsize=(10, 8))
sns.barplot(x=amenities_review_impact.values, y=amenities_review_impact.index)
plt.title('Impact of Amenities on Review Scores Rating')
plt.xlabel('Correlation with Review Scores Rating')
plt.ylabel('Amenities')
plt.show()
# %%
# Analyzing the distribution of review scores and identifying factors influencing guest satisfaction

# Distribution of review scores
plt.figure(figsize=(10, 6))
sns.histplot(df['review_scores_rating'], bins=20, kde=True)
plt.title('Distribution of Review Scores')
plt.xlabel('Review Scores Rating')
plt.ylabel('Frequency')
plt.show()

# Correlation of review scores with numerical features
review_numeric_features = df[['review_scores_rating', 'log_price', 'accommodates', 'bathrooms', 'bedrooms', 'beds']]
review_correlation_matrix = review_numeric_features.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(review_correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.05)
plt.title('Correlation Matrix of Numerical Features with Review Scores')
plt.show()

# Box plots to see the impact of room type and property type on review scores
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.boxplot(ax=axes[0], data=df, x='room_type', y='review_scores_rating')
axes[0].set_title('Review Scores by Room Type')
axes[0].set_xlabel('Room Type')
axes[0].set_ylabel('Review Scores Rating')
sns.boxplot(ax=axes[1], data=df, x='property_type', y='review_scores_rating', order=df['property_type'].value_counts().index[:10])
axes[1].set_title('Review Scores by Property Type')
axes[1].set_xlabel('Property Type')
axes[1].set_ylabel('Review Scores Rating')
axes[1].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()

#%% Analyzing the impact of host characteristics on review scores

host_review_features = ['host_since', 'host_response_rate', 'host_has_profile_pic', 'host_identity_verified', 'review_scores_rating']
data['host_since'] = pd.to_datetime(data['host_since'])
data['host_duration_years'] = (pd.to_datetime('today') - data['host_since']).dt.days / 365
host_review_data = data[host_review_features + ['host_duration_years']]
plt.figure(figsize=(10, 8))
sns.heatmap(host_review_data.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.05)
plt.title('Correlation Matrix for Host Characteristics with Review Scores')
plt.show()
# %%
