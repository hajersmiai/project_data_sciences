from tkinter.ttk import Style
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
# Handling missing values
df['bathrooms'].fillna(df['bathrooms'].median(), inplace=True)
df['bedrooms'].fillna(df['bedrooms'].median(), inplace=True)
df['beds'].fillna(df['beds'].median(), inplace=True)
df['host_has_profile_pic']=df['host_has_profile_pic'].fillna('f')
df['host_identity_verified']=df['host_identity_verified'].fillna('f')
df['host_response_rate']=df['host_response_rate'].fillna('0%')
df['neighbourhood'].fillna('Unknown', inplace=True)
df['review_scores_rating'].fillna(df['review_scores_rating'].median(), inplace=True)
df['thumbnail_url'].fillna('No URL', inplace=True)
df['zipcode']= df['zipcode'].fillna('Unknown')
# Convert host_response_rate to numeric
df['host_response_rate'] = df['host_response_rate'].str.replace('%', '').astype(int)
# df['log_price']=np.exp(df['log_price'])

df['host_has_profile_pic'] = df['host_has_profile_pic'].map({'t': 1, 'f': 0})
df['host_identity_verified'] = df['host_identity_verified'].map({'t': 1, 'f': 0})
# print( df['log_price'])
#%%
def hist_graph(column="Histogramme des prix de logements"):
   
    if column == "Histogramme des prix de logements": #log_price c'est log e du prix
        
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
def corr_matrix(column="Matrice de corrélation"):
    if column =="Matrice de corrélation":
      # #     # Displaying the correlation matrix
        numerical_features = df[['log_price', 'accommodates', 'bathrooms', 'bedrooms', 'beds']]
        correlation_matrix = numerical_features.corr()
        fig=px.imshow(correlation_matrix,text_auto=True,title='Matrice de corrélation des caractéristiques de propriété numériques avec le prix de logement',width=1000,height=700)
        
      #%%
    elif column =="La relation entre le prix et le type de logement":      
    # #     # Scatter plots to visualize the relationship between log_price and other numerical property features
        fig = px.scatter(df,x='accommodates', y='log_price',title='Log Price vs Accomodate')
    elif column =="La relation entre le prix de logement et la salle de bain":
        fig = px.scatter(df,x='bathrooms', y='log_price',title='Log Price vs Bathrooms')
    elif column =="La relation entre le prix de logement et la chambre à coucher":
        fig = px.scatter(df,x='bedrooms', y='log_price',title='Log Price vs Bedrooms')
    elif column =="La relation entre le prix de logement et le lit":
        fig = px.scatter(df,x='beds', y='log_price',title='Log Price vs Beds')
    return (fig)
#%%
# Investigate price variation across different cities or neighborhoods
def price_town(column="La variation des prix dans différentes villes ou quartiers"):
   if column =="La variation des prix dans différentes villes ou quartiers":
        fig= px.box(df,x=df['city'], y=df['log_price'],title='Répartition des prix des logements dans les différentes villes')
        return (fig)
 #%%
# Analyzing price variation across neighborhoods for a specific city 
def price_neighber(column="NYC"):
    if column == 'NYC':
        nyc_data = df[df['city'] == 'NYC']
        top_neighbourhoods = nyc_data['neighbourhood'].value_counts().head(10).index
        nyc_top_neighbourhoods_data = nyc_data[nyc_data['neighbourhood'].isin(top_neighbourhoods)]
        fig=px.box(nyc_top_neighbourhoods_data,x=nyc_top_neighbourhoods_data['neighbourhood'], y=nyc_top_neighbourhoods_data['log_price'],title='Répartition des prix  dans les 10 principaux quartiers de NYC' )
   
    elif column == 'SF':
        sf_data = df[df['city'] == 'SF']
        top_neighbourhoods = sf_data['neighbourhood'].value_counts().head(10).index
        sf_top_neighbourhoods_data = sf_data[sf_data['neighbourhood'].isin(top_neighbourhoods)]
        fig=px.box(sf_top_neighbourhoods_data,x=sf_top_neighbourhoods_data['neighbourhood'], y=sf_top_neighbourhoods_data['log_price'],title='Répartition des prix  dans les 10 principaux quartiers de SF' )
  
    elif column == 'DC':
        dc_data = df[df['city'] == 'DC']
        top_neighbourhoods = dc_data['neighbourhood'].value_counts().head(10).index
        dc_top_neighbourhoods_data = dc_data[dc_data['neighbourhood'].isin(top_neighbourhoods)]
        fig=px.box(dc_top_neighbourhoods_data,x=dc_top_neighbourhoods_data['neighbourhood'], y=dc_top_neighbourhoods_data['log_price'],title='Répartition des prix  dans les 10 principaux quartiers de DC' )
  
    elif column == 'LA':
        la_data = df[df['city'] == 'LA']
        top_neighbourhoods = la_data['neighbourhood'].value_counts().head(10).index
        la_top_neighbourhoods_data = la_data[la_data['neighbourhood'].isin(top_neighbourhoods)]
        fig=px.box(la_top_neighbourhoods_data,x=la_top_neighbourhoods_data['neighbourhood'], y=la_top_neighbourhoods_data['log_price'],title='Répartition des prix  dans les 10 principaux quartiers de LA' )
   
    elif column == 'Chicago':
        ch_data = df[df['city'] == 'Chicago']
        top_neighbourhoods = ch_data['neighbourhood'].value_counts().head(10).index
        ch_top_neighbourhoods_data = ch_data[ch_data['neighbourhood'].isin(top_neighbourhoods)]
        fig=px.box(ch_top_neighbourhoods_data,x=ch_top_neighbourhoods_data['neighbourhood'], y=ch_top_neighbourhoods_data['log_price'],title='Répartition des prix  dans les 10 principaux quartiers de Chicago' )
   
    elif column == 'Boston':
        bs_data = df[df['city'] == 'Boston']
        top_neighbourhoods = bs_data['neighbourhood'].value_counts().head(10).index
        bs_top_neighbourhoods_data = bs_data[bs_data['neighbourhood'].isin(top_neighbourhoods)]
        fig=px.box(bs_top_neighbourhoods_data,x=bs_top_neighbourhoods_data['neighbourhood'], y=bs_top_neighbourhoods_data['log_price'],title='Répartition des prix  dans les 10 principaux quartiers de Boston' )

    return (fig)

#%%
# Analyzing the distribution and popularity of property types and room types across various locations
def loc_type(column='Répartition des types de propriétés'):
# Setting the aesthetic style of the plots
    if column =='Répartition des types de propriétés':
# Distribution of Property Types
        fig= px.histogram(df,x='property_type',text_auto=True,title='Répartition des types de propriétés')      
# Distribution of Room Types
    elif column =='Répartition des types de chambres':
        fig= px.histogram(df,x='room_type',title='Répartition des types de chambres',text_auto=True)
# Popularity of Property Types by City
    elif column =='Popularité des types de propriétés par ville':

        fig= px.density_heatmap(df,x='property_type',y='city',title='Popularité des types de propriétés par ville',text_auto=True)
        # Popularity of Room Types by City
    elif column=='Popularité des types de chambres par ville':
        fig= px.density_heatmap(df,x='room_type',y='city',title='Popularité des types de chambres par ville',text_auto=True)
        # Average Prices for Different Property Types
    elif column =='Prix moyens pour différents types de propriétés':
        fig= px.density_heatmap(df,x='log_price',y='city',title='Prix moyens pour différents types de propriétés sur différents sites',text_auto=True)
    elif column =='Prix moyens pour différents types de chambres':
        fig= px.density_heatmap(df,x='log_price',y='room_type',title='Prix moyens pour différents types de chambres',text_auto=True)
    return fig
#%%
# Analyzing the distribution of listings across different cities and neighborhoods
def list_cities(column='Analyser la répartition des annonces dans différentes villes'):
    if column =='Analyser la répartition des annonces dans différentes villes': 
        fig= px.histogram(df,y='city',text_auto=True,title='Répartition des annonces dans différentes villes')     
    # Analyzing the distribution of listings in neighborhoods for the top cities
    elif column=='Analyser la répartition des annonces dans différentes quartiers':
        top_cities = df['city'].value_counts().index
        fig, axes = plt.subplots(len(top_cities), 1, figsize=(12, 20), sharex=True)

        for i, city in enumerate(top_cities):
            city_data = df[df['city'] == city]
            # fig= px.histogram(city_data,y='neighbourhood',title=f'Répartition des annonces dans les 10 principaux quartiers de {city}',text_auto=True)
            sns.countplot(ax=axes[i], data=city_data, y='neighbourhood', order=city_data['neighbourhood'].value_counts().head(10).index)
            axes[i].set_title(f'Répartition des annonces dans les 10 principaux quartiers de {city}')
            axes[i].set_xlabel('Count')
            axes[i].set_ylabel('Neighborhood')
        
        plt.tight_layout()
        fig=mpl_to_plotly(fig)
    return fig
      
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
            sns.barplot(y=amenities_price_impact.values, x=amenities_price_impact.index)
            plt.title('Impact des commodités sur le prix de logement')
            plt.ylabel('Corrélation avec prix de logement')
            plt.xlabel('Commodités')
            # plt.show()
        elif column=="Analyser l'impact des commodités sur les notes des avis":
        # Analyze the impact of amenities on review_scores_rating
            amenities_review_impact = numeric_cols.drop(columns=['id', 'review_scores_rating']).corrwith(data_amenities['review_scores_rating']).sort_values(ascending=False)
            fig=plt.figure(figsize=(10, 8))
            sns.barplot(y=amenities_review_impact.values, x=amenities_review_impact.index)
            plt.title('Impact des commodités sur les notes des avis')
            plt.ylabel('Corrélation avec les notes des avies')
            plt.xlabel('Commodités')
            # plt.show()  
        fig=mpl_to_plotly(fig) 
        return fig
# %%
        # Analyzing host characteristics and their relationship with listing prices or review scores
def host_carac(column="La matrice de corrélation pour les caractéristiques de l'hôte et les caractéristiques numériques",co=''):
        
        if column =="La matrice de corrélation pour les caractéristiques de l'hôte et les caractéristiques numériques":            
#    # Select host-related features and numerical features for correlation analysis
           
            host_features = ['host_since', 'host_response_rate', 'host_has_profile_pic', 'host_identity_verified']
            numeric_features = ['log_price', 'review_scores_rating']

             # # Convert 'host_since' to datetime and create a new feature 'host_duration_years' to see how long they have been a host
            df['host_since'] = pd.to_datetime(df['host_since'])
            df['host_duration_years'] = (pd.to_datetime('today') - df['host_since']).dt.days / 365

            # # Convert 'host_has_profile_pic' and 'host_identity_verified' to numeric for correlation analysis
            # # Select relevant columns for analysis
            host_data = df[host_features + numeric_features + ['host_duration_years']]
             # # Plot correlation matrix for host characteristics and numerical features
            
            fig =px.imshow(host_data.corr(), text_auto=True, title='Correlation Matrix for Host Characteristics',width=1000,height=700)
             
      # # Visualize the impact of host characteristics on log_price and review_scores_rating       
        elif column=="L'impact des caractéristiques de l'hôte sur le prix des logements et les notes d'évaluation": 
            if co=="L'impact du photo de profile de l'hôte sur le prix dz logement":

                fig=px.box(df,x='host_has_profile_pic', y='log_price',title="Le prix de logement par rapport à l'hôte a une photo de profil")
            elif co=="L'impact de l'identité vérifiée de l'hôte sur le prix de logement":
                fig=px.box(df,x='host_identity_verified', y='log_price',title="Prix de logement par rapport à l'identité de l'hôte vérifiée")
            elif co =="L'impact du photo de profile de l'hôte sur le score d'évaluation":
                fig=px.box(df,x='host_has_profile_pic', y='review_scores_rating',title="Évaluation des scores par rapport à l'hôte a une photo de profil")

            else:
                fig=px.box(df,x='host_identity_verified', y='review_scores_rating',title="Évaluation des scores par rapport à l'identité de l'hôte vérifiée")

    # # Scatter plot to see the relationship between how long the host has been active and log_price
        elif column =="La relation entre la durée d'activité de l'hôte et le prix de logement":
           
           fig=px.scatter(df, x='host_duration_years', y='log_price',title="La relation entre la durée d'activité de l'hôte et le prix de logement")
        return fig
       
       # Analyzing the distribution of review scores and identifying factors influencing guest satisfaction
def review_scores_impact(column="Distribution des notes d'évaluation"):

        if column == "Distribution des notes d'évaluation":
        # Distribution of review scores
            fig = px.histogram(df,x='review_scores_rating',title="Distribution des notes d'évaluation" )
        elif column =="Corrélation des notes des évaluations avec les caractéristiques numériques":
        # Correlation of review scores with numerical features
            review_numeric_features = data[['review_scores_rating', 'log_price', 'accommodates', 'bathrooms', 'bedrooms', 'beds']]
            review_correlation_matrix = review_numeric_features.corr()
            fig = px.imshow(review_correlation_matrix,text_auto=True, title="Corrélation des notes des évaluations avec les caractéristiques numériques",width=1000,height=700)
        elif column =="L'impact du type de chambre sur les notes des avis.":
            # Box plots to see the impact of room type and property type on review scores
            fig = px.box(df, x='room_type', y='review_scores_rating',title="L'impact du type de chambre sur les notes des avis.")
        elif column =="L'impact du type de propriété sur les notes des avis" : 
             fig = px.box(df, x='property_type', y='review_scores_rating', title="L'impact du type de propriété sur les notes des avis")
        elif column =="l'impact des caractéristiques de l'hôte sur les notes des avis" :
            # Analyzing the impact of host characteristics on review scores
            host_review_features = ['host_since', 'host_response_rate', 'host_has_profile_pic', 'host_identity_verified', 'review_scores_rating']
            df['host_since'] = pd.to_datetime(df['host_since'])
            df['host_duration_years'] = (pd.to_datetime('today') - df['host_since']).dt.days / 365
            host_review_data = df[host_review_features + ['host_duration_years']]
            fig = px.imshow( host_review_data.corr(),text_auto=True, title="Corrélation des notes des évaluations avec les caractéristiques  de l'hôte",width=1000,height=700)
        else:
            fig=px.scatter_3d(df,x='review_scores_rating',y='host_has_profile_pic',z='log_price', title="La relation entre le score d'évaluation, le prix de logement et la photo de profil de l'hote",width=1000,height=700)
        return fig
#%%
app=dash.Dash(__name__)
app.layout = html.Div([ html.H1("Etude de données du site airbnb"),
            html.Div([ html.H2("Distribution des prix logement"),
            dcc.Dropdown( ["Histogramme des prix de logements",
                   "Box plot des prix de logement" ],
                 value="Histogramme des prix de logements", 
                 id="type_etude"
                ),
             dcc.Graph(figure=hist_graph(), id="type-graph",className="box")
            ]),
            html.Div([html.H2("Explorer la relation entre le prix et les caractéristiques de la propriété à l'aide de Matrices de corrélation et de Scatter Plot"),
                      dcc.Dropdown(["Matrice de corrélation","La relation entre le prix et le type de logement","La relation entre le prix de logement et la salle de bain",
                      "La relation entre le prix de logement et la chambre à coucher","La relation entre le prix de logement et le lit"],value ="Matrice de corrélation",id="corr_matrix"),
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

            ]),
            html.Div([html.H2("Analyser les caractéristiques de l'hôte et leur relation avec les prix des annonces ou les notes des avis"),
                      dcc.Dropdown(["La matrice de corrélation pour les caractéristiques de l'hôte et les caractéristiques numériques","La relation entre la durée d'activité de l'hôte et le prix de logement","L'impact des caractéristiques de l'hôte sur le prix des logements et les notes d'évaluation"],
                                   
                                    value="La matrice de corrélation pour les caractéristiques de l'hôte et les caractéristiques numériques",id="host_carac"),
                      html.Div([dcc.Dropdown( ["L'impact du photo de profile de l'hôte sur le prix dz logement",
                                    "L'impact de l'identité vérifiée de l'hôte sur le prix de logement",
                                    "L'impact du photo de profile de l'hôte sur le score d'évaluation",
                                    "L'impact du l'identité vérifiée de l'hôte sur le  le score d'évaluation"],value="L'impact du photo de profile de l'hôte sur le prix dz logement", id='host_carac2')],id='dropdown_div'),
                      dcc.Graph(figure=host_carac(),id="fig_host_carac",className="box")

            ]),
             html.Div([html.H2("Analyser la répartition des notes des avis et identifier les facteurs influençant la satisfaction des clients"),
                      dcc.Dropdown(["Distribution des notes d'évaluation","Corrélation des notes des évaluations avec les caractéristiques numériques","L'impact du type de chambre sur les notes des avis.",
                                    "L'impact du type de propriété sur les notes des avis","l'impact des caractéristiques de l'hôte sur les notes des avis","Autre"],
                                    value="Distribution des notes d'évaluation",id="review_scores_impact"),
                      dcc.Graph(figure=review_scores_impact(),id="fig_review_scores_impact",className="box")

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
@app.callback(
     Output("fig_host_carac", "figure"),
     [Input("host_carac", "value"),Input("host_carac2","value")]
 )
 
def update_graph(column,co):
     return host_carac(column,co)
@app.callback(

     Output("dropdown_div","style"),
     [Input("host_carac", "value")]
 )
 
def dropdown_visibility(co):
    if co=="L'impact des caractéristiques de l'hôte sur le prix des logements et les notes d'évaluation":
     return {}
    else :
        return {"display": "none"}

@app.callback(
     Output("fig_review_scores_impact", "figure"),
     [Input("review_scores_impact", "value")]
 )
 
def update_graph(column):
     return review_scores_impact(column)

if __name__ == "__main__":
    app.run(debug=True)
# %%
