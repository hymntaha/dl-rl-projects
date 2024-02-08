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

fig, axes = plt.subplots(1, 2, figsize=(14,5))
axes[0].scatter(data=df, x='G1', y='G3')
axes[0].set_title('G3 versus G1')
axes[0].set_xlabel('G1')
axes[0].set_ylabel('G3')
axes[1].scatter(data=df, x='G2', y='G3')
axes[1].set_title('G3 versus G2')
axes[1].set_xlabel('G2')
axes[1].set_ylabel('G3')
plt.show()

sns.histplot(data=df, x="Dalc", hue="sex", element="step",  stat="density")
plt.title('Influence of gender on the alcohol consumption on the workdays')
plt.show()

sns.histplot(data=df, x="Walc", hue="sex", element="step",  stat="density")
plt.title('Influence of gender on the alcohol consumption on the weekend')
plt.show()

df.groupby('sex')[['Dalc', 'Walc']].mean()

df.groupby('age')[['Dalc', 'Walc']].mean().plot(kind='bar')
plt.ylabel('Alcohol consumption')
plt.xticks(rotation=0)
plt.title('Mean alcohol consumption over age')
plt.show()

df.groupby('age')[['Dalc', 'Walc']].agg(['mean', 'count'])
df.groupby('address')[['Dalc', 'Walc']].agg(['mean', 'count'])

for group, data in df.groupby('address'):
    data['Dalc'].hist(alpha=0.5, density=True, label=group, grid=False)
plt.legend()
plt.title('Workday alcohol consumption depending of the type of living area')
plt.show()

for group, data in df.groupby('address'):
    data['Walc'].hist(alpha=0.5, density=True, label=group, grid=False)
plt.legend()
plt.title('Weekend alcohol consumption depending of the type of living area')
plt.show()

sample_R = df.query('address == "R"')['Walc']
sample_U = df.query('address == "U"')['Walc']

st.ttest_ind(sample_R, sample_U).pvalue
df.groupby('Pstatus')[['Dalc', 'Walc']].agg(['mean', 'count'])

for group, data in df.groupby('Pstatus'):
    data['Dalc'].hist(alpha=0.5, density=True, label=group, grid=False)
plt.legend()
plt.title('Workday alcohol consumption depending of the parent\'s cohabitation status')
plt.show()

for group, data in df.groupby('Pstatus'):
    data['Walc'].hist(alpha=0.5, density=True, label=group, grid=False)
plt.legend()
plt.title('Weekend alcohol consumption depending of the parent\'s cohabitation status')
plt.show()

plt.figure(figsize=(15,6))
for age, grouped_data in df.groupby('age'):
    if age <= 19:
        sns.kdeplot(grouped_data['G1'], label=age)
plt.legend()
plt.title('Grade distribtuion depending on age of students')
plt.show()