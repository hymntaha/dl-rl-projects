import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import missingno as msno
from scipy import stats
from colorama import Fore, init
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')


df = pd.read_csv('./apple_data/apple_quality.csv')
print(df.head())
print(df.info())
msno.bar(df, color='b')
plt.show()

print(df.describe().T.style.background_gradient(axis=0, cmap='cubehelix'))

print(df.duplicated().sum())

for column in df.columns:
    num_distinc_values = len(df[column].unique())
    print(f"{column} : {num_distinc_values} distinct values")


print(df.duplicated().sum())

df[df.isnull().any(axis=1)]

def clean_data(df):
    
    df = df.drop(columns=['A_id'])
    
    df = df.dropna()
    
    df = df.astype({'Acidity': 'float64'})
    
    def label(Quality):
        """
        Transform based on the following examples:
        Quality    Output
        "good"  => 0
        "bad"   => 1
        """
        if Quality == "good":
            return 0
    
        if Quality == "bad":
            return 1
    
        return None
    
    df['Label'] = df['Quality'].apply(label)
    
    df = df.drop(columns=['Quality'])
    
    df = df.astype({'Label': 'int64'})
    
    return df

df_clean = clean_data(df.copy())
df_clean.head()