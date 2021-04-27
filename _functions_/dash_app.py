#!/usr/bin/env python
# coding: utf-8

# # DASH APPLICATION

# In[ ]:


import plotly.express as px
from jupyter_dash import JupyterDash
import dash.dependencies as dd
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import dash
import dash_dangerously_set_inner_html

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import seaborn as sns
from IPython.display import display

import string 

from io import BytesIO

from wordcloud import WordCloud
import base64

get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
import plotly.express as px 
from plotly.subplots import make_subplots


# In[ ]:
import os
getcwd = os.getcwd()
cwd = getcwd.rsplit('\\', 1)[0]+"\\_functions_"

import sys
sys.path.insert(0, cwd)

from recommender_system import *
from functions_eda import *
from sentiment_analysis import *

# In[ ]:
# Load images
nomadland = '../images/nomadland-poster.png' # replace with your own image
encoded_nomadland = base64.b64encode(open(nomadland, 'rb').read()).decode('ascii')

promising = '../images/promising.png' # replace with your own image
encoded_promising = base64.b64encode(open(promising, 'rb').read()).decode('ascii')

hurt = '../images/hurt_locker.png' # replace with your own image
encoded_hurt = base64.b64encode(open(hurt, 'rb').read()).decode('ascii')

lafee = '../images/lafee.png' # replace with your own image
encoded_lafee = base64.b64encode(open(lafee, 'rb').read()).decode('ascii')

# Load Data
ratings = pd.read_csv('../clean_data/ratings_gender.csv')
descriptions = pd.read_csv('../clean_data/data_descriptions.csv')
sentiment = pd.read_csv('../clean_data/data_sentiment.csv')

# Drop duplicates sentiment
sentiment.drop_duplicates(inplace=True)

# tconst_gender from string to tuple
ratings['tconst_gender'] = ratings['tconst_gender'].apply(lambda x : eval(x))

# Create dictionary with movie titles
ratings['primaryTitle'] = ratings['primaryTitle'].str.lower()
ratings_title = ratings.primaryTitle.unique().tolist()
dict_ratings_title = pd.DataFrame({'label':ratings_title, 'value':ratings_title}).to_dict('records')

# Create df with only movies directed by females
ratings_female = ratings[['tconst','primaryTitle','primaryName','gender','genres','bayesian_aver_female']][ratings.gender == 'F'].drop_duplicates('tconst')
ratings_female = ratings_female.sort_values(by='bayesian_aver_female', ascending=False)
ratings_female['primaryName'] = ratings_female['primaryName'].str.lower()

# Create dictionary with the genres of the movies
split_female_genres_list = [x.split(',') for x in ratings_female.genres.tolist()]
# To flat list
flate_female_genres_list = [item for sublist in split_female_genres_list for item in sublist]
# To dataframe
df_female_genres = pd.DataFrame({'genres':flate_female_genres_list})
df_female_genres.drop_duplicates(inplace=True)
df_female_genres = df_female_genres.reset_index().drop('index',axis=1)
df_female_genres['lower'] = df_female_genres.genres.str.lower()
df_female_genres.columns = ['label', 'value']
# To dict
dict_female_genres = df_female_genres.to_dict('records')

# In[ ]:


def dash_app(run_server_mode):
    '''
    Function to run dash app.
    
    Input: server mode (inline, external)
    '''
    app = JupyterDash(__name__, external_stylesheets=[dbc.themes.MINTY],suppress_callback_exceptions=True)
    # the style arguments for the sidebar. We use position:fixed and a fixed width
    SIDEBAR_STYLE = {
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": "16rem",
        "padding": "2rem 1rem",
        "background-color": "#f8f9fa",
    }

    # the styles for the main content position it to the right of the sidebar and
    # add some padding.
    CONTENT_STYLE = {
        "margin-left": "18rem",
        "margin-right": "2rem",
        "padding": "2rem 1rem",
    }
    
    sidebar = html.Div(
    [
        html.H2("Index", className="display-4"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("Recommender", href="/page-1", active="exact"),
                dbc.NavLink("Analysis", href="/page-2", active="exact")
            ],
            vertical=True,
            pills=True
        ),
    ],
    style=SIDEBAR_STYLE
    )

    content = html.Div(id="page-content", style=CONTENT_STYLE)

    app.layout = html.Div([dcc.Location(id="url"), sidebar, content])
    
    ###############################################################################################
    home_layout = html.Div([
        dbc.Row([
                dbc.Col(html.H1("Recommender system for female directed movies"),
                        width={'size': 8, 'offset': 2},
                        ),
                ]),
        dbc.Row([
                dbc.Col(html.Div(children='''This project was created with the aim 
                of bringing films directed by women closer to the public. 
                Here you can find an analysis of the female presence behind 
                the camera made with IMDB datasets. There is also a recommendation 
                system based on the K Nearest Neighbours algorithm.''')
                       ),
                ]),
        dbc.Row(dbc.Col(html.Div(html.P(html.Br())),
                        width=True)
                ),
        dbc.Row([
                dbc.Col(html.H6("Click on the images to know some curious facts"),
                        width={'size': 8, 'offset': 0},
                        ),
                ]),
        dbc.Row(dbc.Col(html.Div(html.P(html.Br())),
                        width=True)
                ),
        dbc.Row([
                dbc.Col([dbc.Button(id='button_nomadland', 
                                   children=[
                                       html.Img(src='data:image/png;base64,{}'.format(encoded_nomadland),
                                               style={'width': 100, 'height': 150}),
                                       html.Img(src='data:image/png;base64,{}'.format(encoded_promising),
                                               style={'width': 100, 'height': 150})], n_clicks=0, color='primary'),
                        html.Div(id='container-button-nomadland')],
                        width={'size': 3, 'offset': 0},
                        ),
                dbc.Col([dbc.Button(id='button_hurt', 
                                   children=[html.Img(src='data:image/png;base64,{}'.format(encoded_hurt),
                                               style={'width': 100, 'height': 150})], 
                                   n_clicks=0, color='primary'),
                         html.Div(id='container-button-hurt')],
                        width={'size': 3, 'offset': 2},
                        ),
                dbc.Col([dbc.Button(id='button_lafee', 
                                   children=[html.Img(src='data:image/png;base64,{}'.format(encoded_lafee),
                                               style={'width': 100, 'height': 150})], 
                                   n_clicks=0, color='primary'),
                        html.Div(id='container-button-lafee')],
                        width={'size': 3, 'offset': 1},
                        ),
            
            ])
    ])

    @app.callback(Output('container-button-nomadland', 'children'),
                  Input('button_nomadland', 'n_clicks'))

    def displayClickNomadland(b1):
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        if 'button_nomadland' in changed_id:
            msg = '''2021, the 93rd edition of the Oscars, marks the first time that 2 films directed
             by women have been nominated for Best Picture. These films are Nomadland by Chloé Zhao
             and Promissing young woman by Emerald Fennell.'''
            return html.Div(msg)

    @app.callback(Output('container-button-hurt', 'children'),
                  Input('button_hurt', 'n_clicks'))

    def displayClickhurt(b2):
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        if 'button_hurt' in changed_id:
            msg = '''The Hurt Locker is a 2009 American war thriller film directed by Kathryn Bigelow. 
            This is the first film directed by a woman to win the Oscar for Best Picture.'''
            return html.Div(msg)
    
    @app.callback(Output('container-button-lafee', 'children'),
                  Input('button_lafee', 'n_clicks'))

    def displayClickLafee(b3):
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        if 'button_lafee' in changed_id:
            msg = '''The first recorded film directed by a woman is Alice
            Guy Blaché's La Fée aux Choux (1896).'''
            return html.Div(msg)
    ############################################################################################
    recommender_layout = html.Div([
        dbc.Row(dbc.Col(html.H3("Recommender system"),
                        width={'size': 8, 'offset': 4},
                        ),
                ),
        
        
        dbc.Row(
            [
                dbc.Col([
                         html.Label('Choose a movie title'),
                         dcc.Dropdown(id='movie_title', placeholder='last dropdown',
                                     options=dict_ratings_title, value = 'titanic')
                        ],
                        width={'size': 3, "offset": 4, 'order': 2}
                        ),
                dbc.Col([
                         html.Label('Choose a movie genre'),
                         dcc.Dropdown(id='filter_dropdown', placeholder='first dropdown',
                                     options= dict_female_genres, value = 'action')
                        ],
                        width={'size': 3, "offset": 1, 'order': 1}
                        )
            ], no_gutters=True
        ),
        
        dbc.Row(
            [
                dbc.Col([
                        html.Label('Select the number of movies'),
                        dcc.Input(id='filter_rows',value=4, type="number")
                        ],
                        width={'size': 3, "offset": 1, 'order': 1}
                        )
            ]
        ),
        
        dbc.Row(dbc.Col(html.Div(html.P(html.Br())),
                        width=True
                        )
                ),
        
        dbc.Row(
            [
                dbc.Col(dt.DataTable(id='table-recommender',
                 columns=[{'id': 'primaryTitle', 'name': 'Title'},
                          {'id': 'primaryName', 'name': 'Director Name'}],
                style_as_list_view=True,
                 style_cell={'textAlign': 'left', 'padding': '5px'},
                 style_header={
                     'backgroundColor': 'white',
                     'fontWeight': 'bold'
                 }),
                        width={'size': 3, "offset": 4, 'order': 2}
                        ),
                dbc.Col( dt.DataTable(id='table-container',
                 columns=[{'id': 'primaryTitle', 'name': 'Title'},
                          {'id': 'primaryName', 'name': 'Director Name'},
                          {'id': 'genres', 'name': 'Genres'}],
                 style_as_list_view=True,
                 style_cell={'textAlign': 'left', 'padding': '5px'},
                 style_header={
                     'backgroundColor': 'white',
                     'fontWeight': 'bold'
                 }),
                        width={'size': 3, "offset": 1, 'order': 1}
                        )
            ], no_gutters=True
        ),
    
        dbc.Row(dbc.Col(html.Div(html.P(html.Br())),
                        width=True
                        )
                ),
    
        dbc.Row(
            [
                dbc.Col([dbc.Button('Random movie', id='btn-nclicks-1', n_clicks=0, color='primary'),
                        html.Div(id='container-button-timestamp')],
                        width={'size': 10, 'offset': 1}
                        )
            ]
        )
    ])

    @app.callback(Output('table-recommender', 'data'),
                  Input('movie_title', 'value'))

    def recommender_system(title):
        similar_ids = find_similar_movies(title)
        data = ratings[['primaryTitle','primaryName']][ratings.tconst_gender.isin(similar_ids)].drop_duplicates()
        data.primaryTitle = data.primaryTitle.apply(lambda x : x.capitalize())
        data.primaryName = data.primaryName.apply(lambda x : string.capwords(x))
        return data.to_dict('records')

    @app.callback(Output('table-container', 'data'),
                  [Input('filter_dropdown', 'value'),
                   Input('filter_rows','value')] )
    
    def display_table(genre, row):
        dff = ratings_female[['primaryTitle', 'primaryName','genres']][ratings_female["genres"].str.lower().str.contains(genre, regex=False, na=False)]   
        dff.primaryTitle = dff.primaryTitle.apply(lambda x : x.capitalize())
        dff.primaryName = dff.primaryName.apply(lambda x : string.capwords(x))
    
        return dff.iloc[0:row,].to_dict('records')


    @app.callback(Output('container-button-timestamp', 'children'),
                  Input('btn-nclicks-1', 'n_clicks'))

    def displayClick(btn1):
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        if 'btn-nclicks-1' in changed_id:
            number = np.random.randint(0, len(ratings_female), 1)[0]
            datos = pd.DataFrame(ratings_female[['primaryTitle','primaryName','genres']].iloc[number,]).T
        
            title = datos.primaryTitle.apply(lambda x : x.capitalize()).values[0]
            genres = datos.genres.str.lower().values[0]
            name = datos.primaryName.apply(lambda x : string.capwords(x)).values[0]
        
            msg = html.P(children=[
                html.Span('I recommend you '),
                html.Strong(title),
                html.Span(', a '),
                html.Span(genres),
                html.Span(' movie by '),
                html.Strong(name)])
            return html.Div(msg)

    ##########################################
    analysis_layout = html.Div([
        dbc.Row(dbc.Col(html.H3("Analysis of the data"),
                        width={'size': 8, 'offset': 4},
                        ),
                ),
        
        
        dbc.Row(dbc.Col(dcc.Graph(id='graph-directors-movies', figure=bar_num_movies(descriptions)),
                        width=True)
        ),
        dbc.Row(dbc.Col(html.Div(html.P(html.Br())),
                        width=True)
                ),
       dbc.Row(dbc.Col(dcc.Graph(id='graph-year-distribution', figure=num_movies_year(descriptions)),
                        width=True)
                        ),
        
        dbc.Row(dbc.Col(html.Div(html.P(html.Br())),
                        width=True)
                ),
        
        dbc.Row(dbc.Col(dcc.Graph(id='graph-genre-dstribution', figure=principal_genres(descriptions)),
                        width=True)
                        ),
        
        dbc.Row(dbc.Col(dcc.Graph(id='sentiment-donut', figure=donut_chart(sentiment)),
                        width=True)
                ),
        dbc.Row(dbc.Col(html.H4("Word cloud of the descriptions"),
                        width={'size': 8, 'offset': 0},
                        ),
                ),
        dbc.Row([
            dbc.Col([
                html.Label('Male directors'),
                html.Img(id="image_wc_des_male")],
                        width={'size': 4, "offset": 0, 'order': 1},
                        ),
            dbc.Col([
                html.Label('Female directors'),
                html.Img(id="image_wc_des_female")],
                        width={'size': 4, "offset": 2, 'order': 2},
                        )
                ])
    ])

    @app.callback(dd.Output('image_wc_des_male', 'src'), [dd.Input('image_wc_des_male', 'id')])

    def make_image_male(b):
        img = BytesIO()
        word_cloud_male(descriptions, 'descriptions').save(img, format='PNG')
        return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())

    @app.callback(dd.Output('image_wc_des_female', 'src'), [dd.Input('image_wc_des_female', 'id')])
    
    def make_image_female(b):
        img = BytesIO()
        word_cloud_female(descriptions, 'descriptions').save(img, format='PNG')
        return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())

    ######################################    
    @app.callback(Output("page-content", "children"), [Input("url", "pathname")])
    
    def render_page_content(pathname):
        if pathname == "/":
            return home_layout
        elif pathname == "/page-1":
            return recommender_layout
        elif pathname == "/page-2":
            return analysis_layout
        # If the user tries to reach a different page, return a 404 message
        return dbc.Jumbotron(
            [
                html.H1("404: Not found", className="text-danger"),
                html.Hr(),
                html.P(f"The pathname {pathname} was not recognised..."),
            ]
        )


    # Run the app
    app.run_server(mode=run_server_mode)

