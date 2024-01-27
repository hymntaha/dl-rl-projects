import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import math
import folium
import numpy as np
from IPython.display import Markdown
from sklearn.preprocessing import LabelEncoder

def bold(string):
    display(Markdown(string))

import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('/kaggle/input/new-york-housing-market/NY-House-Dataset.csv')
df.head()


df = df.drop(['LONGITUDE', 'LATITUDE', 'FORMATTED_ADDRESS', 'LONG_NAME', 'STREET_NAME', 'ADMINISTRATIVE_AREA_LEVEL_2', 'MAIN_ADDRESS', 'STATE', 'ADDRESS', 'BROKERTITLE'], axis=1)
df.head()

df.rename(columns = {'PRICE':'price', 'BEDS':'beds', 
                     'BATH':'bath', 'PROPERTYSQFT':'area', 'LOCALITY': 'place', 
                     'SUBLOCALITY':'sublocality', 'TYPE': "type"}, inplace = True) 

df.head()

df = df.drop(df[df['price'] == 2147483647].index)
df = df.drop(df[df['price'] == 195000000].index)

df.head()
