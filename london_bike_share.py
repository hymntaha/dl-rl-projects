import numpy as np 
import pandas as pd
import os
import pandas_profiling as pp 
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("./bike-data/london_merged.csv")
df.head()

profile = pp.ProfileReport(df)
profile

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp")

df.head()

df["hour"] = df.index.hour
df["day_of_month"] = df.index.day
df["day_of_week"]  = df.index.dayofweek
df["month"] = df.index.month
df.head()

### Exploratory Data Analysis ###
plt.figure(figsize=(15, 7))
ax = sns.lineplot(x=df.index, y=df.cnt,data=df)
ax.set_title("Amount of bike shares vs date", fontsize=25)
ax.set_xlabel("Date", fontsize=20)
ax.set_ylabel('Amount of bike shares', fontsize=20)
plt.show()

df_by_week = df.resample("D").sum()
plt.figure(figsize=(16,6))
ax = sns.lineplot(data=df_by_week,x=df_by_week.index,y=df_by_week.cnt)
ax.set_title("Amount of bike shares per day", fontsize=25)
ax.set_xlabel("Time", fontsize=20)
ax.set_ylabel('Amount of bike shares', fontsize=20)
plt.show()

plt.figure(figsize=(16,6))
ax = sns.pointplot(data=df,hue=df.season,y=df.cnt,x=df.month)
ax.set_title("Amount of bike shares per season", fontsize=25)
ax.set_xlabel("Month", fontsize=20)
ax.set_ylabel('Amount of bike shares', fontsize=20)
plt.show()

plt.figure(figsize=(16, 6))
ax = sns.pointplot(x='day_of_week', y='cnt',data=df)
ax.set_title("Amount of bike shares in a week", fontsize=25)
ax.set_xlabel("Day of the week", fontsize=20)
ax.set_ylabel('Amount of bike shares', fontsize=20)
plt.show()

plt.figure(figsize=(16, 6))
ax = sns.pointplot(x='hour', y='cnt',data=df[df["is_weekend"]==1])
ax.set_title("Amount of bike shares per hour on a weekend day", fontsize=25)
ax.set_xlabel("Hour of they day", fontsize=20)
ax.set_ylabel('Amount of bike shares', fontsize=20)
plt.show()

plt.figure(figsize=(20,10))

ax = sns.pointplot(x='t1', y='cnt',data=df)
ax.set_title("Amount of bike shares vs real temperature", fontsize=25)
ax.set_xlabel("Real temperature (°C)", fontsize=20)
ax.set_ylabel('Amount of bike shares', fontsize=20)
plt.locator_params(axis='x', nbins=10)

plt.figure(figsize=(20,10))

ax = sns.pointplot(x='t2', y='cnt',data=df)
ax.set_title("Amount of bike shares vs feeling temperature", fontsize=25)
ax.set_xlabel("Feeling temperature (°C)", fontsize=20)
ax.set_ylabel('Amount of bike shares', fontsize=20)
plt.locator_params(axis='x', nbins=10)
plt.show()

plt.figure(figsize=(20,10))

ax = sns.pointplot(x='hum', y='cnt',data=df)
ax.set_title("Amount of bike shares vs humidity", fontsize=25)
ax.set_xlabel("Humidity (%)", fontsize=20)
ax.set_ylabel('Amount of bike shares', fontsize=20)
plt.locator_params(axis='x', nbins=10)
plt.show()

print("Temperature and humidity have a weak negative correlation:")
df["t1"].corr(df["hum"], method = "pearson")

plt.figure(figsize=(20,10))

ax = sns.pointplot(x='wind_speed', y='cnt',data=df)
ax.set_title("Amount of bike shares vs windspeed", fontsize=25)
ax.set_xlabel("Windspeed (km/h)", fontsize=20)
ax.set_ylabel('Amount of bike shares', fontsize=20)
plt.locator_params(axis='x', nbins=10)
plt.show()

plt.figure(figsize=(16,6))
ax = sns.histplot(data=df,y=df.cnt,x=df.weather_code)
ax.set_title("Amount of bike shares vs the weather", fontsize=25)
ax.set_xlabel("Weather code", fontsize=20)
ax.set_ylabel('Amount of bike shares', fontsize=20)
plt.show()

plt.figure(figsize=(16,6))
sns.heatmap(df.corr(),cmap="YlGnBu",square=True,linewidths=.5,center=0)