# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 11:40:31 2019

@author: adraj
"""

import pandas as pd
import os
import numpy as np
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
from textblob import TextBlob 

#######vader on property_description##########
def clean_description(desc): 
        '''removing links, special characters  '''
        clean_description = desc
        try:
            clean_description = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", desc).split()) 
        except TypeError:
            pass
        return clean_description

def sentiment_scores_pos(sentence): 
    # Create a SentimentIntensityAnalyzer object. 
    sid_obj = SentimentIntensityAnalyzer()     
    sentiment_dict = sid_obj.polarity_scores(sentence) 
    print("Overall sentiment dictionary is : ", sentiment_dict)   
    return (sentiment_dict['pos']*100)


# Point to data
path = os.getcwd() +'\\Airbnb\\'
train_data = pd.read_csv(path + 'train_airbnb_2905.csv', header = 0, encoding='latin1') 
test_data = pd.read_csv(path + 'test_airbnb_2905.csv', header = 0, encoding='latin1') 
train_data = train_data[(train_data['log_price'] != 0)]
sentiment_data = pd.read_csv(path + 'description_sentiment_all_data.csv', header = 0, encoding='latin1')

data = pd.concat([train_data, test_data],sort=False, ignore_index=True)

sentimentinfo =data[['id','description']]

sentimentinfo['cleandescription'] = sentimentinfo['description'].apply(lambda x:clean_description(x))
sentimentinfo['vader_pos']=sentimentinfo['cleandescription'].apply(lambda x: sentiment_scores_pos(str(x))) 


pd.DataFrame(sentimentinfo.to_csv('description_sentiment_all_data.csv'))

###################### textblob on property_name########  

name_polarity = data[['id','name']]

def clean_name(name): 
        ''' 
        Utility function to clean tweet text by removing links, special characters 
        using simple regex statements. 
        '''
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", name).split()) 
  
def get_name_sentiment(name): 
        ''' 
        Utility function to classify sentiment of passed tweet 
        using textblob's sentiment method 
        '''
        # create TextBlob object of passed tweet text 
        analysis = TextBlob(clean_name(name)) 
        # set sentiment 
        if analysis.sentiment.polarity > 0: 
            return '1'
        elif analysis.sentiment.polarity == 0: 
            return '0'
        else: 
            return '-1'   


name_polarity['polarity_textblob'] = name_polarity['name'].apply(lambda x: get_name_sentiment(str(x)))
name_polarity['log_price']= data['log_price']

pd.DataFrame(name_polarity.to_csv('name_sentiment_all_data.csv'))
