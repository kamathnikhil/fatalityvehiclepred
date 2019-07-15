
# coding: utf-8

# # Severity of fatality prediction through vehicle accidents

# Road accidents are very critical and a major problem in todays world. WHO (World Health Organization) had recorded around 1.25 million death in USA alone by motor vehicle. It is necessary to understand the severity of the crashes and anlayze the factors affecting it.This will help us to prevent some crashes in future. Can machine learning help us to predict factors affecting these crashes?
# 
# I have taken crash datset of New Zealand from crash analysis system which is easily available in different format and APIs. It has various different variables to consider:
# Traffic data
# Vehicle data
# Road data
# Crash data
# Weather data
# 
# Url: https://opendata-nzta.opendata.arcgis.com/datasets/crash-analysis-system-cas-data/geoservice
# 
# I have used Geojson file, instead of the usual CSV file, so that we can perform geographic data analysis without creating geometries from latitude and longitude and deal with coordinate reference systems and projections.
# 
# In New Zealand, the total death in car crash accidents since the year 2000, up to 2018 is 6991.
# While the total number of serious injuries and minor injuries in car accidents reach 45604, 208623 respectively.
# 
# Fatality count is not very high in the data. Mostly consist of major and minor serious injusries.

# #### Loading necessary libraries

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import requests
import copy


# In[387]:


get_ipython().system('pip install folium ')
import folium


# In[7]:


# Get the data from url and request it as json file
url = 'https://opendata.arcgis.com/datasets/a163c5addf2c4b7f9079f08751bd2e1a_0.geojson'
geojson = requests.get(url).json()


# In[8]:


# Read the data as GeodataFrame in Geopandas
crs = {'init': 'epsg:3851'} # Coordinate reference system (CRS) for Newzealand
gdf = gpd.GeoDataFrame.from_features(geojson['features'], crs=crs)


# In[32]:


df = copy.deepcopy(gdf)


# Used deep copy function as I did not want any modification on the original data

# In[400]:


from accidentfunc import missing_values, modeling


# ### Handling Missing Values

# In[12]:


missing_values(gdf)


# ## Data Preprocessing

# Prepared and cleaned the data to use it for further analysis

# In[54]:


df['crashDirec']= df['crashDirec'].fillna('Unknown')
df['crashRPDir'] = df['crashRPDir'].fillna(0)
df['speedLimit'] = df['speedLimit'].fillna(999)
df['cornerRoad'] = df['cornerRoad'].fillna(0)
df['roadLane'] = df['roadLane'].fillna(999)
df['crashRPSH'] = df['crashRPSH'].fillna(0)
df['crashRPNew'] = df['crashRPNew'].fillna(0)


# In[96]:


df['speedLimit']=df['speedLimit'].astype(str).astype(int)
df = df[(df['speedLimit'] != 999.0)]


# ### Exploratory Data Analysis

# In[64]:


df.info()


# ### Crash reports of fatal counts, serious injuries and minor injuries over the years

# In[36]:


df.rename(columns = {'seriousInj':'seriousInjuryCount','minorInjur':'minorInjuryCount'}, inplace = True)


# In[122]:


fig, ax = plt.subplots(1, 3, figsize = (30, 5));
sns.set_style("dark")
sns.lineplot(x="crashYear", y="fatalCount",data=df, ax=ax[0]);
sns.lineplot(x="crashYear", y="seriousInjuryCount",data=df, ax=ax[1]);
sns.lineplot(x="crashYear", y="minorInjuryCount",data=df, ax=ax[2]);


# In[99]:


year_wise_fatalCounts = df.groupby(['crashYear']).agg({'fatalCount':['sum'],'speedLimit':['min','max']})
year_wise_fatalCounts


# ### Fatality count by Roads

# In[150]:


fig, ax = plt.subplots(nrows = 3, ncols = 3, figsize = (20, 10));

sns.barplot(x='roadCurvat',y="fatalCount",data=df,palette="Blues_d", ax=ax[0,0])
sns.barplot(x="junctionTy",y="fatalCount",data=df,palette="Reds_d",  ax=ax[0,1])
sns.barplot(x='roadLane',y="fatalCount",data=df,palette="BuGn_r", ax=ax[0,2])
sns.barplot(x='numberOfLa',y="fatalCount",data=df,palette="Blues_d", ax=ax[1,0])
sns.barplot(x='roadWet',y="fatalCount",data=df,palette="BuGn_r", ax=ax[1,1])
sns.barplot(x='roadMarkin',y="fatalCount",data=df, palette="Reds_d", ax=ax[1,2])
sns.barplot(x='roadSurfac',y="fatalCount",data=df,palette="Blues_d", ax=ax[2,0])
sns.barplot(x='darkLight',y="fatalCount",data=df,palette="BuGn_r", ax=ax[2,1])
sns.barplot(x='intersecti',y="fatalCount",data=df,palette="Reds_d", ax=ax[2,2])
plt.tight_layout()


# I have genrated graph in such a way that it could give us clear indications on how diffrent important road parameters can affect accidents. In road lanes, highest is 2 wrt to fatal counts. There is 999 in data which is basically unknown values replaced.

# ### Fatalities count wrt traffic

# In[271]:


plt.figure(figsize=(8,6))
sns.barplot(x="speedLimit", y="fatalCount",  data=df,palette="Blues_d")


# In[273]:


plt.figure(figsize=(8,6))
sns.catplot(x="advisorySp", y="crashSever",data=df, kind='bar',palette="Reds_d",)


# Possible values are 'F' (fatal), 'S' (serious), 'M' (minor), 'N' (non-injury).
# FatalCount is more wrt speed

# ### Fatality wrt Weather

# In[276]:


plt.figure(figsize=(8,6))
sns.barplot(x="fatalCount", y="weatherA",data=gdf, palette="Greens_d")


# In[278]:


plt.figure(figsize=(8,6))
sns.barplot(x = "fatalCount", y="weatherB",data=gdf, palette="Blues_d")


# ### Geographic data

# In[160]:


df.plot(markersize=0.01, edgecolor='orange',figsize=(12,12));
plt.axis('off');


# #### Let us have a look at crashes aggregated in cluster map in Auckland.
# 

# In[142]:


from folium.plugins import MarkerCluster
df_sample = df.sample(5000)
lons = df_sample.geometry.x
lats = df_sample.geometry.y

m = folium.Map(
    location=[np.mean(lats), np.mean(lons)],
    tiles='Cartodb Positron',
    zoom_start=6
)

#FastMarkerCluster(data=list(zip(lats, lons))).add_to(m)
MarkerCluster(list(zip(lats, lons))).add_to(m)

folium.LayerControl().add_to(m)
m


# In[237]:


df_sample = df.sample(5000)
lons = df_sample.geometry.x
lats = df_sample.geometry.y
heat_cols = list(zip(lats, lons))
from folium.plugins import HeatMap

m = folium.Map([np.mean(lats), np.mean(lons)], 
               tiles='CartoDB dark_matter', 
               zoom_start=6)

HeatMap(heat_cols).add_to(m)
m


# #### The map gives us clear indication on most fatal roads and accidents prone areas which is mostly in cities.

# ## Machine Learning Model

# I have implemented three different model to compare and check the accuracy. Lets see if weather and rest of the parameters affects the results.

# We need to convert categorical variable to numerical for regression. I have one hot encoding to do so.

# In[288]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[250]:


# Since GeoDataframe and Geometry Column will not work in our analysis therefore, change our data to data frame.
newdf = pd.DataFrame(df.drop(['geometry','OBJECTID','OBJECTID_1'], axis = 1))
newdf.head()


# In[251]:


#This converts all string / object type columns to categorical. Then applies codes to each type of category.
for x in newdf:
    if (newdf[x].dtype == 'object'):
        newdf[x]= newdf[x].astype('category')
        newdf[x] = newdf[x].cat.codes


# In[388]:


newdf = pd.get_dummies(newdf)
newdf.head(5)


# ### Lets see if the fatalCounts differ wrt to weather conditions

# In[259]:


newdf1 = newdf.drop(columns=['weatherA', 'weatherB'])
#we will need to replace NA values here with 999 as the classifier will not accept any NA values
newdf1['speedLimit'] = newdf1['speedLimit'].fillna(999)
newdf['speedLimit'] = newdf['speedLimit'].fillna(999)


# In[283]:


Y = newdf.fatalCount.values
Y1 = newdf1.fatalCount.values


# In[260]:


X = newdf.loc[:, newdf.columns != 'fatalCount']
X1 = newdf1.loc[:, newdf1.columns != 'fatalCount']
X.columns;


# In[432]:


# Split our data into training and testing sets
X_train1, X_test1,Y_train1,Y_test1 = train_test_split(X1, Y1, test_size=0.33, random_state=99)
#Without weather


# In[433]:


# Random Forest

rf = RandomForestClassifier(n_estimators=50)
rf.fit(X_train1, Y_train1)
Y_pred = rf.predict(X_test1)
rf.score(X_train1, Y_train1)
acc_rf1 = round(rf.score(X_test1, Y_test1) * 100, 2)
acc_rf1


# ### Function called to check the accuracy of the model

# In[382]:


modeling(X,newdf.fatalCount.values,'logistic')


# Logistic Regression performs almost the same as RF.

# ### There is not much on removing the weather conditions. The classification is successful and the accuracy of the model is more or less 99.91% when investigated on multiple metrics. 

# In[436]:


f_imp = pd.DataFrame(data={'importance': rf.feature_importances_, 'features': X_train1.columns}).set_index('features')
f_imp = f_imp.sort_values('importance', ascending=False)
f_imp.head(10)


# Here are the important factors wrt accidents that we should consider. The results indicated that adding weather-related features to a machine learning algorithm in predicting severity of an accident did not change the accuracy of the model. When adding three features of light condition, weather condition, and the condition of the road surface, the measures of recall, precision, and f1-score remained unchanged.
