import dash
from dash import dcc, html
import plotly.express as px
from dash.dependencies import Output,Input
import plotly.express as px
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("agg")
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer
from plotly.tools import mpl_to_plotly

folder= Path(__file__).parent
#%%

data = pd.read_csv(Path(folder,"Airbnb_Data.csv"))

df = pd.DataFrame(data)
# Convert host_response_rate to numeric
# df['host_response_rate'] = df['host_response_rate'].str.replace('%', '').astype(int)
#%%
def hist_graph(column="Histogramme des prix de logements"):
   
    if column == "Histogramme des prix de logements":
        fig=plt.figure(figsize=(10, 6))
        sns.histplot(df['log_price'], bins=30, kde=True)
        plt.title('Distribution des prix de logement')
        plt.xlabel('Log Price')
        plt.ylabel('Frequency')
        fig =mpl_to_plotly(fig)

        # # Box plot of log prices
    elif column =="Box plot des prix de logement":   
         fig=px.box(df,x=df['log_price'],title='Box plot des prix de logement')
    return (fig)
    #%%    
    #     # Exploring the relationship between price and property features using scatter plots and correlation matrices
    #     # Selecting numerical columns that are relevant to the property features
def corr_matrix(column="Correlation matrices"):
    if column =="Correlation matrices":
      # #     # Displaying the correlation matrix
        numerical_features = df[['log_price', 'accommodates', 'bathrooms', 'bedrooms', 'beds']]
        correlation_matrix = numerical_features.corr()
        # fig=plt.figure(figsize=(10, 8))
        # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.05)
        # plt.title('Matrice de corrélation des caractéristiques de propriété numériques avec le prix de logement')
        fig=px.imshow(correlation_matrix,text_auto=True,title='Matrice de corrélation des caractéristiques de propriété numériques avec le prix de logement')
        # fig = mpl_to_plotly(fig)
        return fig  
      #%%
    elif column =="Scatter Plot":
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # #     # Scatter plots to visualize the relationship between log_price and other numerical property features
            sns.scatterplot(ax=axes[0, 0], data=df, x='accommodates', y='log_price')
            axes[0, 0].set_title('Log Price vs Accommodates')

   
            sns.scatterplot(ax=axes[0, 1], data=df, x='bathrooms', y='log_price')
            axes[0, 1].set_title('Log Price vs Bathrooms')

   
            sns.scatterplot(ax=axes[1, 0], data=df, x='bedrooms', y='log_price')
            axes[1, 0].set_title('Log Price vs Bedrooms')

    
            sns.scatterplot(ax=axes[1, 1], data=df, x='beds', y='log_price')
            axes[1, 1].set_xticks(np.arange(min(data['beds']), max(data['beds'])+1, 1))

            plt.tight_layout()
            fig=mpl_to_plotly(fig)
    return (fig)
#%%
# Investigate price variation across different cities or neighborhoods
def price_town(column="La variation des prix dans différentes villes ou quartiers"):
   if column =="La variation des prix dans différentes villes ou quartiers":
        # fig = plt.figure(figsize=(12, 8))
        fig= px.box(df,x=df['city'], y=df['log_price'],title='Log Price Distribution Across Different Cities')
        # plt.title('Log Price Distribution Across Different Cities')
        # plt.xlabel('City')
        # plt.ylabel('Log Price')
        return (fig)
 #%%
# Analyzing price variation across neighborhoods for a specific city 
def price_neighber(column="NYC"):
    if column == 'NYC':
        nyc_data = df[df['city'] == 'NYC']
        top_neighbourhoods = nyc_data['neighbourhood'].value_counts().head(10).index
        nyc_top_neighbourhoods_data = nyc_data[nyc_data['neighbourhood'].isin(top_neighbourhoods)]

    # plt.figure(figsize=(14, 8))
        fig=px.box(nyc_top_neighbourhoods_data,x=nyc_top_neighbourhoods_data['neighbourhood'], y=nyc_top_neighbourhoods_data['log_price'],title='Log Price Distribution Across Top 10 Neighborhoods in NYC' )
    # plt.title('Log Price Distribution Across Top 10 Neighborhoods in NYC')
    # plt.xlabel('Neighborhood')
    # plt.ylabel('Log Price')
    #  plt.xticks(rotation=45)
    # plt.show()
    elif column == 'SF':
        sf_data = df[df['city'] == 'SF']
        top_neighbourhoods = sf_data['neighbourhood'].value_counts().head(10).index
        sf_top_neighbourhoods_data = sf_data[sf_data['neighbourhood'].isin(top_neighbourhoods)]
        fig=px.box(sf_top_neighbourhoods_data,x=sf_top_neighbourhoods_data['neighbourhood'], y=sf_top_neighbourhoods_data['log_price'],title='Log Price Distribution Across Top 10 Neighborhoods in SF' )
    elif column == 'DC':
        dc_data = df[df['city'] == 'DC']
        top_neighbourhoods = dc_data['neighbourhood'].value_counts().head(10).index
        dc_top_neighbourhoods_data = dc_data[dc_data['neighbourhood'].isin(top_neighbourhoods)]
        fig=px.box(dc_top_neighbourhoods_data,x=dc_top_neighbourhoods_data['neighbourhood'], y=dc_top_neighbourhoods_data['log_price'],title='Log Price Distribution Across Top 10 Neighborhoods in DC' )
    elif column == 'LA':
        la_data = df[df['city'] == 'LA']
        top_neighbourhoods = la_data['neighbourhood'].value_counts().head(10).index
        la_top_neighbourhoods_data = la_data[la_data['neighbourhood'].isin(top_neighbourhoods)]
        fig=px.box(la_top_neighbourhoods_data,x=la_top_neighbourhoods_data['neighbourhood'], y=la_top_neighbourhoods_data['log_price'],title='Log Price Distribution Across Top 10 Neighborhoods in LA' )
    elif column == 'Chicago':
        ch_data = df[df['city'] == 'Chicago']
        top_neighbourhoods = ch_data['neighbourhood'].value_counts().head(10).index
        ch_top_neighbourhoods_data = ch_data[ch_data['neighbourhood'].isin(top_neighbourhoods)]
        fig=px.box(ch_top_neighbourhoods_data,x=ch_top_neighbourhoods_data['neighbourhood'], y=ch_top_neighbourhoods_data['log_price'],title='Log Price Distribution Across Top 10 Neighborhoods in Chicago' )
    elif column == 'Boston':
        bs_data = df[df['city'] == 'Boston']
        top_neighbourhoods = bs_data['neighbourhood'].value_counts().head(10).index
        bs_top_neighbourhoods_data = bs_data[bs_data['neighbourhood'].isin(top_neighbourhoods)]
        fig=px.box(bs_top_neighbourhoods_data,x=bs_top_neighbourhoods_data['neighbourhood'], y=bs_top_neighbourhoods_data['log_price'],title='Log Price Distribution Across Top 10 Neighborhoods in Boston' )

    return (fig)

#%%
# Analyzing the distribution and popularity of property types and room types across various locations
def loc_type(column='Répartition des types de propriétés'):
# Setting the aesthetic style of the plots
    # sns.set(style="whitegrid")
    if column =='Répartition des types de propriétés':
# Distribution of Property Types
        fig=plt.figure(figsize=(12, 6))
        sns.countplot(data=df, y='property_type', order=df['property_type'].value_counts().index)
        plt.title('Distribution of Property Types')
        plt.xlabel('Count')
        plt.ylabel('Property Type')
        # plt.show()
       
       
# Distribution of Room Types
    elif column =='Répartition des types de chambres':
        fig=plt.figure(figsize=(8, 4))
        sns.countplot(data=df, x='room_type', order=df['room_type'].value_counts().index)
        plt.title('Distribution of Room Types')
        plt.xlabel('Room Type')
        plt.ylabel('Count')
        # plt.show()

# Popularity of Property Types by City
    elif column =='Popularité des types de propriétés par ville':
        fig=plt.figure(figsize=(12, 6))
        sns.countplot(data=df, y='property_type', hue='city', order=df['property_type'].value_counts().index)
        plt.title('Popularity of Property Types by City')
        plt.xlabel('Count')
        plt.ylabel('Property Type')
        plt.legend(title='City')
        # plt.show()

        # Popularity of Room Types by City
    elif column=='Popularité des types de chambres par ville':
        fig=plt.figure(figsize=(8, 4))
        sns.countplot(data=df, x='room_type', hue='city', order=df['room_type'].value_counts().index)
        plt.title('Popularity of Room Types by City')
        plt.xlabel('Room Type')
        plt.ylabel('Count')
        plt.legend(title='City')
        # plt.show()

        # Average Prices for Different Property Types
    elif column =='Prix moyens pour différents types de propriétés':
        fig=plt.figure(figsize=(12, 6))
        sns.barplot(data=df, y='property_type', x='log_price', estimator=np.mean, ci=None, order=df.groupby('property_type')['log_price'].mean().sort_values(ascending=False).index)
        plt.title('Average Log Prices for Different Property Types')
        plt.xlabel('Average Log Price')
        plt.ylabel('Property Type')
        # plt.show()

        # Average Prices for Different Room Types
    elif column =='Prix moyens pour différents types de chambres':
        fig=plt.figure(figsize=(8, 4))
        sns.barplot(data=df, x='room_type', y='log_price', estimator=np.mean, ci=None, order=df.groupby('room_type')['log_price'].mean().sort_values(ascending=False).index)
        plt.title('Average Log Prices for Different Room Types')
        plt.xlabel('Room Type')
        plt.ylabel('Average Log Price')
        # plt.show()
    fig=mpl_to_plotly(fig)
    return fig
#%%
# Analyzing the distribution of listings across different cities and neighborhoods
def list_cities(column='Analyser la répartition des annonces dans différentes villes'):
    if column =='Analyser la répartition des annonces dans différentes villes':    
        fig=plt.figure(figsize=(12, 8))
        sns.countplot(data=df, y='city', order=df['city'].value_counts().index)
        plt.title('Distribution of Listings Across Different Cities')
        plt.xlabel('Count')
        plt.ylabel('City')
        # plt.show()

        # Analyzing the distribution of listings in neighborhoods for the top cities
    elif column=='Analyser la répartition des annonces dans différentes quartiers':
        top_cities = df['city'].value_counts().index
        fig, axes = plt.subplots(len(top_cities), 1, figsize=(12, 20), sharex=True)

        for i, city in enumerate(top_cities):
            city_data = df[df['city'] == city]
            sns.countplot(ax=axes[i], data=city_data, y='neighbourhood', order=city_data['neighbourhood'].value_counts().head(10).index)
            axes[i].set_title(f'Distribution of Listings in Top 10 Neighborhoods of {city}')
            axes[i].set_xlabel('Count')
            axes[i].set_ylabel('Neighborhood')

    plt.tight_layout()
    fig=mpl_to_plotly(fig)
    return fig
        # plt.show()

#%%
     # Analyzing the impact of different amenities on listing prices and review scores
def amenities_etude(column="Analyser l'impact des commodités sur le prix de logement"):
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
        if column =="Analyser l'impact des commodités sur le prix de logement":
        # Analyze the impact of amenities on log_price
            amenities_price_impact = numeric_cols.drop(columns=['id', 'log_price']).corrwith(data_amenities['log_price']).sort_values(ascending=False)
            fig=plt.figure(figsize=(10, 8))
            sns.barplot(x=amenities_price_impact.values, y=amenities_price_impact.index)
            plt.title('Impact of Amenities on Log Price')
            plt.xlabel('Correlation with Log Price')
            plt.ylabel('Amenities')
            # plt.show()
        elif column=="Analyser l'impact des commodités sur les notes des avis":
        # Analyze the impact of amenities on review_scores_rating
            amenities_review_impact = numeric_cols.drop(columns=['id', 'review_scores_rating']).corrwith(data_amenities['review_scores_rating']).sort_values(ascending=False)
            fig=plt.figure(figsize=(10, 8))
            sns.barplot(x=amenities_review_impact.values, y=amenities_review_impact.index)
            plt.title('Impact of Amenities on Review Scores Rating')
            plt.xlabel('Correlation with Review Scores Rating')
            plt.ylabel('Amenities')
            # plt.show()  
        fig=mpl_to_plotly(fig) 
        return fig

# Analyzing host characteristics and their relationship with listing prices or review scores

# Select host-related features and numerical features for correlation analysis
# host_features = ['host_since', 'host_response_rate', 'host_has_profile_pic', 'host_identity_verified']
# numeric_features = ['log_price', 'review_scores_rating']

# # Convert 'host_since' to datetime and create a new feature 'host_duration_years' to see how long they have been a host
# data['host_since'] = pd.to_datetime(data['host_since'])
# data['host_duration_years'] = (pd.to_datetime('today') - data['host_since']).dt.days / 365

# # Convert 'host_has_profile_pic' and 'host_identity_verified' to numeric for correlation analysis
# data['host_has_profile_pic'] = data['host_has_profile_pic'].map({'t': 1, 'f': 0})
# data['host_identity_verified'] = data['host_identity_verified'].map({'t': 1, 'f': 0})

# # Select relevant columns for analysis
# host_data = data[host_features + numeric_features + ['host_duration_years']]

# # Plot correlation matrix for host characteristics and numerical features
# plt.figure(figsize=(10, 8))
# sns.heatmap(host_data.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.05)
# plt.title('Correlation Matrix for Host Characteristics')
# plt.show()

# # Visualize the impact of host characteristics on log_price and review_scores_rating
# fig, axes = plt.subplots(2, 2, figsize=(14, 12))
# sns.boxplot(ax=axes[0, 0], data=data, x='host_has_profile_pic', y='log_price')
# axes[0, 0].set_title('Log Price vs Host Has Profile Pic')
# sns.boxplot(ax=axes[0, 1], data=data, x='host_identity_verified', y='log_price')
# axes[0, 1].set_title('Log Price vs Host Identity Verified')
# sns.boxplot(ax=axes[1, 0], data=data, x='host_has_profile_pic', y='review_scores_rating')
# axes[1, 0].set_title('Review Scores Rating vs Host Has Profile Pic')
# sns.boxplot(ax=axes[1, 1], data=data, x='host_identity_verified', y='review_scores_rating')
# axes[1, 1].set_title('Review Scores Rating vs Host Identity Verified')
# plt.tight_layout()
# plt.show()

# # Scatter plot to see the relationship between how long the host has been active and log_price
# plt.figure(figsize=(10, 6))
# sns.scatterplot(data=data, x='host_duration_years', y='log_price')
# plt.title('Log Price vs Host Duration in Years')
# plt.xlabel('Host Duration (Years)')
# plt.ylabel('Log Price')
# plt.show()
#%%
app=dash.Dash(__name__)
app.layout = html.Div([ html.H1("Etude de données du site airbnb"),
            html.Div([ html.H2("Distribution des prix selon le type de logement"),
            dcc.Dropdown( ["Histogramme des prix de logements",
                   "Box plot des prix de logement" ],
                 value="Histogramme des prix de logements", 
                 id="type_etude"
                ),
             dcc.Graph(figure=hist_graph(), id="type-graph",className="box")
            ]),
            html.Div([html.H2("Explorer la relation entre le prix et les caractéristiques de la propriété à l'aide de matrices de corrélation"),
                      dcc.Dropdown(["Correlation matrices","Scatter Plot"],value ="Correlation matrices",id="corr_matrix"),
                      dcc.Graph(figure =corr_matrix(),id="fig_corr_matrix",className="box")
            ]),
            html.Div([html.H2("La variation des prix dans différentes villes ou quartiers"),
                      dcc.Dropdown(["La variation des prix dans différentes villes ou quartiers" ],value="La variation des prix dans différentes villes ou quartiers",id="price_town"),
                      dcc.Graph(figure=price_town(),id="fig_price_town",className="box")
            ]),
            html.Div([html.H2("Analyser la variation des prix entre les quartiers pour une ville donnée"),
                      dcc.Dropdown(["NYC","SF","DC","LA","Chicago","Boston"],value="NYC",id="price_neighber"),
                      dcc.Graph(figure=price_neighber(),id="fig_price_neighber",className="box")

            ]),
            html.Div([html.H2("Analyser la répartition et la popularité des types de propriétés et des types de chambres sur divers sites"),
                      dcc.Dropdown(["Répartition des types de propriétés","Répartition des types de chambres",
                                    "Popularité des types de propriétés par ville","Popularité des types de chambres par ville",
                                    "Prix moyens pour différents types de propriétés","Prix moyens pour différents types de chambres"],
                                    value="Répartition des types de propriétés",id="loc_type"),
                      dcc.Graph(figure=loc_type(),id="fig_loc_type",className="box")

            ]),
            html.Div([html.H2("Analyser la répartition des annonces dans différentes villes et quartiers"),
                      dcc.Dropdown(["Analyser la répartition des annonces dans différentes villes","Analyser la répartition des annonces dans différentes quartiers"],
                                    value="Analyser la répartition des annonces dans différentes villes",id="list_cities"),
                      dcc.Graph(figure=list_cities(),id="fig_list_cities",className="box")

            ]),
            html.Div([html.H2("Analyser l'impact de différentes commodités sur les prix des annonces et les notes des avis"),
                      dcc.Dropdown(["Analyser l'impact des commodités sur le prix de logement","Analyser l'impact des commodités sur les notes des avis"],
                                    value="Analyser l'impact des commodités sur le prix de logement",id="amenities_etude"),
                      dcc.Graph(figure=amenities_etude(),id="fig_amenities_etude",className="box")

            ])
]
)


@app.callback(
     Output("type-graph", "figure"),
     [Input("type_etude", "value")]
 )
def update_graph(x_column):
     return hist_graph(x_column)
@app.callback(
     Output("fig_corr_matrix", "figure"),
     [Input("corr_matrix", "value")]
 )
def update_graph(column):
     return corr_matrix(column)
@app.callback(
     Output("fig_price_town", "figure"),
     [Input("price_town", "value")]
 )
def update_graph(column):
     return price_town(column)

@app.callback(
     Output("fig_price_neighber", "figure"),
     [Input("price_neighber", "value")]
 )
def update_graph(column):
     return price_neighber(column)
@app.callback(
     Output("fig_loc_type", "figure"),
     [Input("loc_type", "value")]
 )
def update_graph(column):
     return loc_type(column)
@app.callback(
     Output("fig_list_cities", "figure"),
     [Input("list_cities", "value")]
 )
def update_graph(column):
     return list_cities(column)
@app.callback(
     Output("fig_amenities_etude", "figure"),
     [Input("amenities_etude", "value")]
 )
def update_graph(column):
     return amenities_etude(column)

if __name__ == "__main__":
    app.run(debug=True)
# %%
