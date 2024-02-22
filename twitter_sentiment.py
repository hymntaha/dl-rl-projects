import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import os
for dirname, _, filenames in os.walk('./twitter_data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import re
import nltk
from nltk import word_tokenize
nltk.download('stopwords')


val = pd.read_csv('./twitter_data/twitter_validation.csv')
train = pd.read_csv('./twitter_data/twitter_training.csv')

train.columns = ['id', 'information', 'type', 'text']
print(train.head())

val.columns=['id','information','type','text']
val.head()

train_data=train
train_data

val_data=val
val_data

#Text transformation
train_data["lower"]=train_data.text.str.lower() #lowercase
train_data["lower"]=[str(data) for data in train_data.lower] #converting all to string
train_data["lower"]=train_data.lower.apply(lambda x: re.sub('[^A-Za-z0-9 ]+', ' ', x)) #regex
val_data["lower"]=val_data.text.str.lower() #lowercase
val_data["lower"]=[str(data) for data in val_data.lower] #converting all to string
val_data["lower"]=val_data.lower.apply(lambda x: re.sub('[^A-Za-z0-9 ]+', ' ', x)) #regex

train_data.head()
 