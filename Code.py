#https://www.kaggle.com/code/kushagranull/crop-yield-prediction/notebook

import numpy as np
import pandas as pd
df_yield = pd.read_csv('yield.csv')
print(df_yield.shape)
print(df_yield.head())
print(df_yield.tail(10))

# rename columns.
df_yield = df_yield.rename(index=str, columns={"Value": "hg/ha_yield"})
df_yield.head()

# drop unwanted columns.
df_yield = df_yield.drop(['Year Code','Element Code','Element','Year Code','Area Code','Domain Code','Domain','Unit','Item Code'], axis=1)
df_yield.head()

print(df_yield.describe())

print(df_yield.info())

df_rain = pd.read_csv('rainfall.csv')
print(df_rain.head())

print(df_rain.tail())

df_rain = df_rain.rename(index=str, columns={" Area": 'Area'})

print(df_rain.info())

df_rain['average_rain_fall_mm_per_year'] = pd.to_numeric(df_rain['average_rain_fall_mm_per_year'],errors = 'coerce')
print(df_rain.info())


df_rain = df_rain.dropna()

print(df_rain.describe())

yield_df = pd.merge(df_yield, df_rain, on=['Year','Area'])
print(yield_df.head())

print(yield_df.describe())


df_pes = pd.read_csv('pesticides.csv')
print(df_pes.head())

df_pes = df_pes.rename(index=str, columns={"Value": "pesticides_tonnes"})
df_pes = df_pes.drop(['Element','Domain','Unit','Item'], axis=1)
print(df_pes.head())

print(df_pes.describe())

print(df_pes.info())

yield_df = pd.merge(yield_df, df_pes, on=['Year','Area'])
print(yield_df.shape)

print(yield_df.head())


avg_temp=  pd.read_csv('temp.csv')
print(avg_temp.head())

print(avg_temp.describe())

avg_temp = avg_temp.rename(index=str, columns={"year": "Year", "country":'Area'})
print(avg_temp.head())

yield_df = pd.merge(yield_df,avg_temp, on=['Area','Year'])
print(yield_df.head())

print(yield_df.shape)

print(yield_df.describe())

print(yield_df.isnull().sum())

print(yield_df.groupby('Item').count())

print(yield_df.describe())

print(yield_df['Area'].nunique())

print(yield_df.groupby(['Area'],sort=True)['hg/ha_yield'].sum().nlargest(10))

print(yield_df.groupby(['Item','Area'],sort=True)['hg/ha_yield'].sum().nlargest(10))


import sklearn
import seaborn as sns
import matplotlib.pyplot as plt


correlation_data=yield_df.select_dtypes(include=[np.number]).corr()

mask = np.zeros_like(correlation_data, dtype=np.bool_)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.palette="vlag"

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(correlation_data, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5});

plt.show()

print(yield_df.head())

from sklearn.preprocessing import OneHotEncoder

yield_df_onehot = pd.get_dummies(yield_df, columns=['Area',"Item"], prefix = ['Country',"Item"])
features=yield_df_onehot.loc[:, yield_df_onehot.columns != 'hg/ha_yield']
label=yield_df['hg/ha_yield']
print(features.head())

features = features.drop(['Year'], axis=1)
print(features.info())

print(features.head())


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
features=scaler.fit_transform(features)

print(features)


from sklearn.model_selection import train_test_split
train_data, test_data, train_labels, test_labels = train_test_split(features, label, test_size=0.2, random_state=42)

#write final df to csv file
yield_df.to_csv('Yield_df.csv')

from sklearn.model_selection import train_test_split
train_data, test_data, train_labels, test_labels = train_test_split(features, label, test_size=0.2, random_state=42)




yield_df_onehot = yield_df_onehot.drop(['Year'], axis=1)

print(yield_df_onehot.head())


#setting test data to columns from dataframe and excluding 'hg/ha_yield' values where ML model should be predicting

test_df=pd.DataFrame(test_data,columns=yield_df_onehot.loc[:, yield_df_onehot.columns != 'hg/ha_yield'].columns)

# using stack function to return a reshaped DataFrame by pivoting the columns of the current dataframe

cntry=test_df[[col for col in test_df.columns if 'Country' in col]].stack()[test_df[[col for col in test_df.columns if 'Country' in col]].stack()>0]
cntrylist=list(pd.DataFrame(cntry).index.get_level_values(1))
countries=[i.split("_")[1] for i in cntrylist]
itm=test_df[[col for col in test_df.columns if 'Item' in col]].stack()[test_df[[col for col in test_df.columns if 'Item' in col]].stack()>0]
itmlist=list(pd.DataFrame(itm).index.get_level_values(1))
items=[i.split("_")[1] for i in itmlist]

print(test_df.head())

test_df.drop([col for col in test_df.columns if 'Item' in col],axis=1,inplace=True)
test_df.drop([col for col in test_df.columns if 'Country' in col],axis=1,inplace=True)
print(test_df.head())

test_df['Country']=countries
test_df['Item']=items
print(test_df.head())

from sklearn.tree import DecisionTreeRegressor
clf=DecisionTreeRegressor()
model=clf.fit(train_data,train_labels)

test_df["yield_predicted"]= model.predict(test_data)
test_df["yield_actual"]=pd.DataFrame(test_labels)["hg/ha_yield"].tolist()
test_group=test_df.groupby("Item")
# test_group.apply(lambda x: r2_score(x.yield_actual,x.yield_predicted))

# So let's run the model actual values against the predicted ones

fig, ax = plt.subplots()

ax.scatter(test_df["yield_actual"], test_df["yield_predicted"],edgecolors=(0, 0, 0))

ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title("Actual vs Predicted")
plt.show()

varimp= {'imp':model.feature_importances_,'names':yield_df_onehot.columns[yield_df_onehot.columns!="hg/ha_yield"]}

a4_dims = (8.27,16.7)
fig, ax = plt.subplots(figsize=a4_dims)
df=pd.DataFrame.from_dict(varimp)
df.sort_values(ascending=False,by=["imp"],inplace=True)
df=df.dropna()
sns.barplot(x="imp",y="names",palette="vlag",data=df,orient="h",ax=ax);

plt.show()


#7 most important factors that affect crops
a4_dims = (16.7, 8.27)

fig, ax = plt.subplots(figsize=a4_dims)
df=pd.DataFrame.from_dict(varimp)
df.sort_values(ascending=False,by=["imp"],inplace=True)
df=df.dropna()
df=df.nlargest(7, 'imp')
sns.barplot(x="imp",y="names",palette="vlag",data=df,orient="h",ax=ax);

plt.show()

#Boxplot that shows yield for each item
a4_dims = (16.7, 8.27)

fig, ax = plt.subplots(figsize=a4_dims)
sns.boxplot(x="Item",y="hg/ha_yield",palette="vlag",data=yield_df,ax=ax);

plt.show()







