#!/usr/bin/env python
# coding: utf-8

# # Clean different texts

# # Descriptions

# In[ ]:


def clean_descriptions(text):
    '''
    Function to clean the descriptions of the dataset.
    
    Make text lowercase, remove text in square brackets,
    remove links,remove punctuation
    and remove words containing numbers.
    
    Input: string
    Output: clean string
    '''
    
    text = re.sub(r'\n.*?\n', '',text).strip()
    text = text.lower()
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    
    return text
    


# # Reviews

# In[ ]:


def clean_reviews(text):
    '''
    Function to clean reviews of the dataset.
    
    Make text lowercase, remove text in square brackets,
    remove links,remove punctuation
    and remove words containing numbers.
    
    Input: string
    Output: clean string
    '''
   
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = text.strip()
    return text

