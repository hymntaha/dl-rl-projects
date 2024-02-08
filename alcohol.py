import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as st
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict
from xgboost import XGBRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.dummy import DummyRegressor

df = pd.read_csv('./data/student-mat.csv')
print(df.head())
df.info()

df.describe(include='all').T
pd.concat([df['school'].value_counts().to_frame(), df['school'].value_counts(normalize=True).to_frame()], axis=1)
sns.histplot(df.age, bins=8, discrete=True)
plt.title('Age distribtuion')
plt.show()

sns.histplot(df.sex)
plt.title('Gender distribtuion')
plt.show()

sns.histplot(df.Dalc, discrete=True)
plt.title('Workday alcohol consumption distribtuion')
plt.show()

df['Dalc'].value_counts(normalize=True)

sns.histplot(df.Walc, discrete=True)
plt.title('Weekend alcohol consumption distribtuion')
plt.show()

df['Walc'].value_counts(normalize=True)
df[df['Dalc'] == 1]['Walc'].value_counts(normalize=True)

sns.histplot(df['absences'], bins=30)
plt.title('Absences distribtuion')
plt.show()

sns.histplot(df['G1'], discrete=True)
plt.title('G1 distribtuion')
plt.show()

sns.histplot(df['G2'], discrete=True)
plt.title('G2 distribtuion')
plt.show()

sns.histplot(df['G3'], discrete=True)
plt.title('G3 distribtuion')
plt.show()

df.query('G3 == 0')
