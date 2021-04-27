#!/usr/bin/env python
# coding: utf-8

# # FUNCIONES EDA

# In[2]:


#import libraries
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import seaborn as sns
from IPython.display import display

get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
import plotly.express as px 
from plotly.subplots import make_subplots

from wordcloud import WordCloud

# text processing libraries
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

py.init_notebook_mode(connected=True)

import warnings
warnings.filterwarnings('ignore')


# In[3]:


#CREAR COLOR PALETTES
#Midsommar palette
colors_male = ['#5A6C70', '#003F2E', '#D4AB72', '#FF7F50', '#D99800']
colors_female = ['#DAA190', '#9F5F9D', '#9F9DAA', '#3F5D8F', '#B50102']
male_cmap = ListedColormap(colors_male, name='my_colormap_name_male')
female_cmap = ListedColormap(colors_female, name='my_colormap_name_female')
stop = set(stopwords.words("english"))

# In[ ]:


# NUMBER OF DIRECTORS BY GENDER
def bar_num_directors(df):
    '''
    Create a bar plot with the number of female directors and male directors.
    
    Input: dataframe with a column "gender" with the gender of the movie director, and the column "primaryName"
    '''
    df_barplot = pd.DataFrame({'gender':['Male', 'Female'], 
                           'count':[df.primaryName[df.gender=='M'].nunique(),df.primaryName[df.gender=='F'].nunique()]})
    fig = px.bar(df_barplot, 
            x='gender',
            y='count',
            color='gender',
            title = 'Number of male directors vs number of female directors',
            color_discrete_sequence =[colors_male[0], colors_female[0]],
                template='simple_white')
    
    fig['layout']['yaxis']['title']='Number of movie directors'
    fig['layout']['xaxis']['title']=''
    fig['layout']['legend']['title'] = 'Gender'
    
    #fig.show()
    return fig

# In[24]:


#NUMBER OF MOVIES PER GENDER
def bar_num_movies(df):
    '''
    Create a bar plot with the number of movies directed by a woman and the number of movies directed by a man.
    
    Input: dataframe with a column "gender" with the gender of the movie director and the column "primaryTitle"
    '''
    df_barplot = pd.DataFrame({'gender':['Male', 'Female'], 
                           'count':[df.primaryTitle[df.gender=='M'].nunique(),df.primaryTitle[df.gender=='F'].nunique()]})
    fig = px.bar(df_barplot, 
            x='gender',
            y='count',
            color='gender',
            title = 'Number of movies with male directors vs number of movies with female directors',
            color_discrete_sequence =[colors_male[0], colors_female[0]],
                template='simple_white')
    
    fig['layout']['yaxis']['title']='Number of movies'
    fig['layout']['xaxis']['title']=''
    fig['layout']['legend']['title'] = 'Gender'
    
    #fig.show()
    return fig


# In[26]:


# DISTRIBUTION OF RATINGS
def dist_ratings(df):
    '''
    Create a boxplot with the distributions of ratings
    of movies directed by women and the number of movies directed by men.
    
    Input: dataframe with a column "gender" with the gender of the movie director
    and the column "averageRating" with the rating for each movie
    '''
    data = df.drop_duplicates(subset='tconst')
    
    fig = go.Figure()
    
    fig.add_trace(go.Box(
        y= data.averageRating[data.gender=='M'],
        name='Male',
        marker_color=colors_male[0],
        boxmean=True # represent mean
    ))
    
    fig.add_trace(go.Box(
        y= data.averageRating[data.gender=='F'],
        name='Female',
        marker_color=colors_female[0],
        boxmean=True # represent mean
    ))
    
    fig.update_layout(template='simple_white')
    fig['layout']['yaxis']['title'] = 'Average ratings'
    fig['layout']['title'] = 'Distribution of average ratings by gender'
    fig['layout']['legend']['title'] = 'Gender'
    
    #fig.show()
    return fig


# In[ ]:


# DISTRIBUTION OF NUMBER OF VOTES
def dist_num_votes(df):
    '''
    Create a boxplot with the distributions of the number of votes
    of movies directed by women and the movies directed by men.
    
    Input: dataframe with a column "gender" with the gender of the movie director
    and the column "numVotes" with the number of votes for each movie
    '''
    data = df.drop_duplicates(subset='tconst')
    
    fig = go.Figure()
    
    fig.add_trace(go.Box(
        y= data.numVotes[data.gender=='M'],
        name='Male',
        marker_color=colors_male[0],
        boxmean=True # represent mean
    ))
    
    fig.add_trace(go.Box(
        y= data.numVotes[data.gender=='F'],
        name='Female',
        marker_color=colors_female[0],
        boxmean=True # represent mean
    ))
    
    fig.update_layout(template='simple_white')
    fig['layout']['yaxis']['title'] = 'Number of votes'
    fig['layout']['title'] = 'Distribution of number of votes by gender'
    fig['layout']['legend']['title'] = 'Gender'
    
    #fig.show()
    return fig


# In[30]:


# BAYESIAN AVERAGE RATING
def bayesian_avg_rating(df, gender='both'):
    '''
    Function to calculate the average rating of each movie
    
    Input: 
        df = dataframe with gender, numVotes and averageRating
        gender = string ('both', 'male', 'female') to calculate
        bayesian average for both male and female directors or only
        for one gender
    Output:
        dataframe with a new column with the bayesian rating
    '''
    
    if gender == 'both':
        # DF with both genders
        df_total = df.drop_duplicates(['tconst', 'gender'])
        
        # C constant
        C = df_total.averageRating.mean()
        # m constant
        m = df_total.numVotes.mean()
        # average
        bayesian_avg_rating = [(df_total.averageRating[df_total.tconst==t].values[0]*df_total.numVotes[df_total.tconst==t].values[0]-m*C)/(df_total.numVotes[df_total.tconst==t].values[0]+m) 
                       for t in df_total.tconst.tolist()]
        # dataframe
        df_bayesian_avg_rating = pd.DataFrame({'tconst':df_total.tconst.tolist(),'bayesian_aver':bayesian_avg_rating})
        
        #return
        return df.merge(df_bayesian_avg_rating, on='tconst', how='left',copy=False)
    
    elif gender == 'male':
        # DF for male
        df_male = df[['tconst','primaryTitle','averageRating','numVotes']][df.gender == 'M'].drop_duplicates('tconst')
        
        # C constant
        C = df_male.averageRating.mean()
        # m constant
        m = df_male.numVotes.mean()
        # average
        bayesian_avg_rating = [(df_male.averageRating[df_male.tconst==t].values[0]*df_male.numVotes[df_male.tconst==t].values[0]-m*C)/(df_male.numVotes[df_male.tconst==t].values[0]+m) 
                       for t in df_male.tconst.tolist()]
        # dataframe
        df_bayesian_avg_rating = pd.DataFrame({'tconst':df_male.tconst.tolist(),'bayesian_aver_male':bayesian_avg_rating})
        
        #return
        return df.merge(df_bayesian_avg_rating, on='tconst', how='left',copy=False)
    
    elif gender == 'female':
        
        # DF for female
        df_female = df[['tconst','primaryTitle','averageRating','numVotes']][df.gender == 'F'].drop_duplicates('tconst')
    
        # C constant
        C = df_female.averageRating.mean()
        # m constant
        m = df_female.numVotes.mean()
        # average
        bayesian_avg_rating = [(df_female.averageRating[df_female.tconst==t].values[0]*df_female.numVotes[df_female.tconst==t].values[0]-m*C)/(df_female.numVotes[df_female.tconst==t].values[0]+m) 
                       for t in df_female.tconst.tolist()]
        # dataframe
        df_bayesian_avg_rating = pd.DataFrame({'tconst':df_female.tconst.tolist(),'bayesian_aver_female':bayesian_avg_rating})
        
        #return
        return df.merge(df_bayesian_avg_rating, on='tconst', how='left', copy=False)
    else:
        print('The posible values for gender are "female", "male" or "both"')
    


# In[ ]:


# PRINCIPALES GENEROS
def principal_genres(df):
    '''
    Pie plot with the 5 principal movies genre for each director gender.
    
    Input: dataframe
    Output: pieplot
    '''
    
    colores = [ '#5A6C70', '#D4AB72', '#885D4C', '#D99800', '#DAA190', '#9F5F9D', '#9F9DAA', '#3F5D8F', '#B50102','#377856',]
    
    ################## male ########################
    # genres to list
    male_genres_list = df.drop_duplicates('tconst').genres[df.gender=='M'].tolist()
    # split when there are more than 1 genre
    split_male_genres_list = [x.split(',') for x in male_genres_list]
    #flat list
    flate_male_genres_list = [item for sublist in split_male_genres_list for item in sublist]
    # to dataframe
    df_male_genres = pd.DataFrame({'genres':flate_male_genres_list})
    # value counts
    male_pie = df_male_genres.genres.value_counts().reset_index().rename(columns={'index':'genre','genres':'count'}).iloc[0:5,:]
    
    ######################### female #################
    # genres to list
    female_genres_list = df.drop_duplicates('tconst').genres[df.gender=='F'].tolist()
    # split when there are more than 1 genre
    split_female_genres_list = [x.split(',') for x in female_genres_list]
    #flat list
    flate_female_genres_list = [item for sublist in split_female_genres_list for item in sublist]
    # to dataframe
    df_female_genres = pd.DataFrame({'genres':flate_female_genres_list})
    # value counts
    female_pie = df_female_genres.genres.value_counts().reset_index().rename(columns={'index':'genre','genres':'count'}).iloc[0:5,:]
    
    ################### pie plots ##################
    fig = make_subplots(rows=1, cols=2, 
                        specs=[[{'type':'domain'}, {'type':'domain'}]],
                       subplot_titles=['Male', 'Female'])
    
    fig.add_trace(go.Pie(
        labels = male_pie.genre,
        values = male_pie['count'],
        name = 'Male'
    ),1,1)
    
    fig.add_trace(go.Pie(
        labels = female_pie.genre,
        values= female_pie['count'],
        name = 'Female' 
    ),1,2)
    
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(template='simple_white', colorway = colores )
    
    fig['layout']['title'] = 'Most popular genres by gender'
    fig['layout']['legend']['title'] = 'Genre of movie'
    
    fig = go.Figure(fig)
    #fig.show()
    return fig


# In[ ]:


# FIRST MOVIE BY GENDER
def first_movie(df, gender):
    '''
    Function to know the movie and the name director of the first movie in the dataset directed by a man or a woman.
    
    Input:
        df = dataframe with the movies
        gender = string('M', 'F')
    Output:
        dataframe or series with all the values for a movie
    '''
    year = min(df.startYear[df.gender==gender])
    return df[['primaryTitle','primaryName','startYear']][(df.startYear == year) & (df.gender == gender)]


# In[ ]:


# DISTRIBUCION POR AÑOS
def num_movies_year(df):
    '''
    Temporal series plot with the number of movies for each director gender by years.
    
    Input: dataframe tih the movies
    
    '''
    
    datos = df.drop_duplicates(['tconst','gender'])
    datos_group = datos.groupby(['startYear','gender'])['tconst'].count().reset_index().rename(columns={'tconst':'number'})
    
    ####################################################################
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=datos_group.startYear[datos_group.gender=='M'], 
                         y=datos_group.number[datos_group.gender=='M'],
                         mode='lines+markers',
                         name='Male'))
    
    fig.add_trace(go.Scatter(x=datos_group.startYear[datos_group.gender=='F'], 
                         y=datos_group.number[datos_group.gender=='F'],
                         mode='lines+markers',
                         name = 'female'))
    
    fig.update_layout(template='simple_white', colorway = [colors_male[0], colors_female[0]])
    fig['layout']['legend']['title'] = 'Gender'
    fig['layout']['title'] = 'Number of movies by year and gender'
    fig['layout']['xaxis']['title'] = 'Year'
    fig['layout']['yaxis']['title'] = 'Number of movies'
    
    #fig.show()
    
    return fig

# In[ ]:


# DISTRIBUCION POR AÑOS DOBLE AXIS
def num_movies_year2(df):
    '''
    Temporal series plot with the number of movies for each director gender by years with 2 axis.
    
    Input: dataframe tih the movies
    
    '''
    datos = df.drop_duplicates(['tconst','gender'])
    datos_group = datos.groupby(['startYear','gender'])['tconst'].count().reset_index().rename(columns={'tconst':'number'})
    
    ####################################################################
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(go.Scatter(x=datos_group.startYear[datos_group.gender=='M'], 
                         y=datos_group.number[datos_group.gender=='M'],
                         mode='lines+markers',
                         name='Male'),secondary_y=False)
    
    fig.add_trace(go.Scatter(x=datos_group.startYear[datos_group.gender=='F'], 
                         y=datos_group.number[datos_group.gender=='F'],
                         mode='lines+markers',
                         name = 'Female'),secondary_y=True)
    
    fig.update_layout(template='simple_white', colorway = [colors_male[0], colors_female[0]])
    fig['layout']['legend']['title'] = 'Gender'
    fig['layout']['title'] = 'Number of movies by year and gender'
    fig['layout']['xaxis']['title'] = 'Year'
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Male", secondary_y=False)
    fig.update_yaxes(title_text="Female", secondary_y=True)
    
    return fig


# In[ ]:


# WORDCLOUD
def word_cloud_male(df, column):
    '''
    Wordcloud with the principal words for the movies directed by men.
    
    Input:
        df = dataframe
        column = string('descriptions', 'reviews')
    '''
    wordcloud1 = WordCloud(background_color='white',
                           max_words=100,
                           colormap = male_cmap,
                           stopwords = stop,
                        width=300,
                        height=200).generate(" ".join(df[column][df.gender=='M'].drop_duplicates()))
    return wordcloud1

# In[ ]:

def word_cloud_female(df, column):
    '''
    Wordcloud with the principal words for the movies directed by women.
    
    Input:
        df = dataframe
        column = string('descriptions', 'reviews')
     '''
    wordcloud2 = WordCloud(background_color='white',
                           max_words=100,
                           colormap = female_cmap,
                           stopwords=stop,
                        width=300,
                        height=200).generate(" ".join(df[column][df.gender=='F'].drop_duplicates()))
    
    return wordcloud2




# In[ ]:
def word_cloud_no_gender(df, column):
    '''
    Wordcloud with the principal words for the movies directed by men and women.
    
    Input:
        df = dataframe
        column = string('descriptions', 'reviews')
    '''
    wordcloud = WordCloud(background_color='white',
                           max_words=100,
                           colormap = female_cmap,
                           stopwords=set(stopwords.words("english")),
                        width=480,
                        height=360).generate(" ".join(df[column].drop_duplicates()))
    
    return wordcloud



