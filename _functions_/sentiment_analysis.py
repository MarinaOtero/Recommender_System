#!/usr/bin/env python
# coding: utf-8

# # Sentiment analysis

# In[ ]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('seaborn')
import seaborn as sns

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import warnings
warnings.filterwarnings("ignore")


# # Histograms

# In[ ]:


def sent_analysis(data, gender):
    if gender == 'male':
        fig = make_subplots(rows=2, cols=2, 
                    subplot_titles=('Negative Sentiment Score', 'Neutral Sentiment Score',
                                    'Positive Sentiment Score', 'Compound'),
                   x_title = '<b>Sentiment Score</b>', y_title='<b>Number of Reviews</b>')
        trace0 = go.Histogram(x=data.sentiment_neg[data.gender=='M'], 
                      nbinsx=25, marker_color='#DAA190', autobinx = False,
                     name = 'Negative Sentiment Score', showlegend=False)

        trace1 = go.Histogram(x=data.sentiment_neu[data.gender=='M'], 
                      nbinsx=25, marker_color='#AA2366', autobinx = False,
                     name = 'Neutral Sentiment Score', showlegend=False)

        trace2 = go.Histogram(x=data.sentiment_pos[data.gender=='M'], 
                      nbinsx=25, marker_color='#9F9DAA', autobinx = False,
                     name = 'Positive Sentiment Score', showlegend=False)

        trace3 = go.Histogram(x=data.sentiment_compound[data.gender=='M'], 
                      nbinsx=25, marker_color='#3F5D8F', autobinx = False,
                     name = 'Compound', showlegend=False)

        fig.append_trace(trace0, 1, 1)
        fig.append_trace(trace1, 1, 2)
        fig.append_trace(trace2, 2, 1)
        fig.append_trace(trace3, 2, 2)

        fig.update_layout(
            template='simple_white',
            title={
                'text': '<b>Sentiment Analysis of Reviews for Male Directors</b>',
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 18, 'family':'Arial'}
            },
            xaxis_title={
                'font': {'size': 14, 'family':'Arial'}
            },
            yaxis_title={
                'font': {'size': 14, 'family':'Arial'}
            })

        fig.update_annotations(font = {'size': 12, 'family':'Arial'})

        fig.show()
        
    elif gender == 'female':
        fig = make_subplots(rows=2, cols=2, 
                    subplot_titles=('Negative Sentiment Score', 'Neutral Sentiment Score',
                                    'Positive Sentiment Score', 'Compound'),
                   x_title = '<b>Sentiment Score</b>', y_title='<b>Number of Reviews</b>')
        trace0 = go.Histogram(x=data.sentiment_neg[data.gender=='F'], 
                      nbinsx=25, marker_color='#DAA190', autobinx = False,
                     name = 'Negative Sentiment Score', showlegend=False)

        trace1 = go.Histogram(x=data.sentiment_neu[data.gender=='F'], 
                      nbinsx=25, marker_color='#AA2366', autobinx = False,
                     name = 'Neutral Sentiment Score', showlegend=False)

        trace2 = go.Histogram(x=data.sentiment_pos[data.gender=='F'], 
                      nbinsx=25, marker_color='#9F9DAA', autobinx = False,
                     name = 'Positive Sentiment Score', showlegend=False)

        trace3 = go.Histogram(x=data.sentiment_compound[data.gender=='F'], 
                      nbinsx=25, marker_color='#3F5D8F', autobinx = False,
                     name = 'Compound', showlegend=False)

        fig.append_trace(trace0, 1, 1)
        fig.append_trace(trace1, 1, 2)
        fig.append_trace(trace2, 2, 1)
        fig.append_trace(trace3, 2, 2)

        fig.update_layout(
            template='simple_white',
            title={
                'text': '<b>Sentiment Analysis of Reviews for Female Directors</b>',
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 18, 'family':'Arial'}
            },
            xaxis_title={
                'font': {'size': 14, 'family':'Arial'}
            },
            yaxis_title={
                'font': {'size': 14, 'family':'Arial'}
            })

        fig.update_annotations(font = {'size': 12, 'family':'Arial'})

        fig.show()
        


# # Donut chart

# In[ ]:


def donut_chart(data):
    percentiles_male = data.sentiment_compound[data.gender=='M'].describe(percentiles=[.05, .1, .2, .3, .4, .5, .6, .7, .8, .9])
    percentiles_female = data.sentiment_compound[data.gender=='F'].describe(percentiles=[.05, .1, .2, .3, .4, .5, .6, .7, .8, .9])

    names = ['Negative Comments', 'Neutral Comments','Positive Comments']
    colores = ['#3EB489', '#D4AB72', '#FF7F50']

    neg_male = percentiles_male['30%']
    mid_male = percentiles_male['40%']
    pos_male = percentiles_male['max']
    size_male = [neg_male, mid_male, pos_male]

    neg_female = percentiles_female['30%']
    mid_female = percentiles_female['40%']
    pos_female = percentiles_female['max']

    size_female = [neg_female, mid_female, pos_female]


    fig = make_subplots(rows=1, cols=2, 
                        specs=[[{'type':'domain'}, {'type':'domain'}]],
                       subplot_titles=['Male', 'Female'])
    
    fig.add_trace(go.Pie(
        labels = names,
        values = size_male,
        name = 'Male',
        hole = .3,
        showlegend=False
        ),1,1)
    
    fig.add_trace(go.Pie(
        labels = names,
        values= size_female,
        name = 'Female',
        hole = .3,
        showlegend = False
        ),1,2)
    
    fig.update_traces(textinfo='percent+label')

    fig.update_layout(template='simple_white', colorway = colores,
                  annotations=[dict(text='Male', x=0.225, y=0.495, font_size=16, showarrow=False),
                 dict(text='Female', x=0.775, y=0.495, font_size=16, showarrow=False)],
                 title={
                'text': 'Sentiment analysis of reviews by gender',
                'y':0.9,
                'x':0.5,
                'xanchor': 'right',
                'yanchor': 'top'#,
                #'font': {'size': 12, 'family':'Arial'}
            })
    

    
    fig = go.Figure(fig)
    #fig.show()
    return fig


# # Text length

# In[ ]:


def length_text(df_pos_male,df_neg_male,df_pos_female,df_neg_female):
    
    sns.set_style("white")
    
    fig, axes = plt.subplots(1,2, figsize=(24,8))

    fig.suptitle('Distribution Plot for Length of Comments',  fontsize=18, fontweight='bold')

    sns.distplot(df_pos_male['text_length'], kde=True, bins=50, color='#DAA190',ax = axes[0])
    sns.distplot(df_neg_male['text_length'], kde=True, bins=50, color='#3F5D8F',ax = axes[0])
    axes[0].set_title('Male')
    axes[0].legend(['Positive Comments', 'Negative Comments'])
    axes[0].set(ylabel='', xlabel='')

    sns.distplot(df_pos_female['text_length'], kde=True, bins=50, color='#DAA190',ax = axes[1])
    sns.distplot(df_neg_female['text_length'], kde=True, bins=50, color='#3F5D8F',ax = axes[1])
    axes[1].set_title('Female')
    axes[1].legend(['Positive Comments', 'Negative Comments'])
    axes[1].set(ylabel='', xlabel='')

    fig.text(0.5, 0.04, 'Text Length', ha='center',fontsize=16, fontweight='bold')
    fig.text(0.04, 0.5, 'Percentage of comments', fontweight='bold',fontsize = 16, va='center', rotation='vertical');


# In[ ]:




