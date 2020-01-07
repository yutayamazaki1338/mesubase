#!/usr/bin/env python
# coding: utf-8

# ## 予測アルゴリズムの完成を目指す
# - ①変数の特徴を見る
# - ②特徴量を考える
# - ③アルゴリズムの検討をする。

# In[30]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LinearRegression as LR
import seaborn as sns

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample = pd.read_csv("sample.csv")

train.index = pd.to_datetime(train["datetime"])

train.head()


# In[31]:


train.describe()


# In[32]:


train.describe(include="O")


# In[33]:


train["precipitation"]


# ### データ改編と特徴量の作成
# - NaNを補完する
# - datetimeから日時情報を取り出したい

# In[34]:


train["remarks"] = train["remarks"].fillna("なし")
train["event"] = train["event"].fillna("なし")
train["payday"] = train["payday"].fillna(0)
train["precipitation"] =train["precipitation"].apply(lambda x : -1 if x == "--" else float(x))
train["month"] = train["datetime"].apply(lambda x : int(x.split("-")[1]))


# In[35]:


train["y"].plot(figsize=(15,4))


# #### ここまででわかること
# - 日付が進むことによって、全体的には減少傾向にあること
# - ところどころ何かの要因で上に跳ね上がっている場所が存在する。
# - 説明変数とyの関係性を見ていくべき

# ### 相関関係を見てみる
# - 数字情報を持つものは散布図で確認
# - ステータスで与えられるものは箱ひげ図で確認する

# In[1]:


fig,ax = plt.subplots(3,2,figsize=(12,7))

train.plot.scatter(x="soldout",y="y",ax=ax[0][0],c=1)
train.plot.scatter(x="kcal",y="y",ax=ax[0][1],c=1)
train.plot.scatter(x="payday",y="y",ax=ax[1][0],c=1)
train.plot.scatter(x="temperature",y="y",ax=ax[1][1],c=1)
train.plot.scatter(x="precipitation",y="y",ax=ax[2][0],c=1)
train.plot.scatter(x="month",y="y",ax=ax[2][1],c=1)


# In[37]:


fig, ax = plt.subplots(2,3,figsize=(9,6))
train.plot.scatter(x="soldout", y="y", ax=ax[0][0])
train.plot.scatter(x="kcal", y="y", ax=ax[0][1])
train.plot.scatter(x="precipitation", y="y", ax=ax[0][2])
train.plot.scatter(x="payday", y="y", ax=ax[1][0])
train.plot.scatter(x="temperature", y="y", ax=ax[1][1])
train.plot.scatter(x="month", y="y", ax=ax[1][2])


# - kicalとtemperatureがやや顕著に見える

# In[38]:


sns.set(font="IPAexGothic",style="white")#日本語の指定
fig, ax = plt.subplots(3,2,figsize=(12,12))

sns.boxplot(x="week",y="y",data=train,ax=ax[0][0])
sns.boxplot(x="event",y="y",data=train,ax=ax[0][1])
sns.boxplot(x="remarks",y="y",data=train,ax=ax[1][0])
ax[1][0].set_xticklabels(ax[1][0].get_xticklabels(),rotation=30)
sns.boxplot(x="event",y="y",data=train,ax=ax[1][1])
sns.boxplot(x="payday",y="y",data=train,ax=ax[2][0])
plt.tight_layout()


# In[39]:


train[train["remarks"]!= "お楽しみメニュー"]["y"].plot(figsize=(15,4))


# In[40]:


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as MSE
from sklearn.linear_model import LinearRegression as LR
from sklearn.ensemble import RandomForestRegressor as RF


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample = pd.read_csv("sample.csv",header=None)


# In[41]:


train["t"] = 1
test["t"] = 0
train.head()


# In[42]:


dat1 = pd.concat([train,test],sort=True).reset_index()
dat1


# In[43]:


dat = pd.concat([train,test],sort=True).reset_index(drop=True)
dat


# In[44]:


dat.index = pd.to_datetime(dat["datetime"])
dat


# In[45]:


dat = dat["2014-05-01":]
dat


# In[46]:


dat = dat.reset_index(drop=True)
dat


# In[47]:


dat["days"] = dat.index


dat


# In[48]:


dat["precipitation"] = dat["precipitation"].apply(lambda x : -1 if x=="--" else x).astype(np.float)
dat["fun"] = dat["remarks"].apply(lambda x: 1 if x=="お楽しみメニュー" else 0)
dat["curry"] = dat["name"].apply(lambda x : 1 if x.find("カレー")>=0 else 0)

cols = ["precipitation","weather","days","fun","curry","y"]


# In[49]:


def learning(trainX,y_train):
    model1 = LR()
    model2 = RF(n_estimators=100,max_depth=4,random_state=777)
    
    model1.fit(trainX["days"].values.reshape(-1,1),y_train)
    pred = model1.predict(trainX["days"].values.reshape(-1,1))
    
    pred_sub = y_train - pred
    model2.fit(trainX.iloc[:, ~trainX.columns.str.match("y")],pred_sub)
    return model1, model2


# In[53]:


train.iloc[:,~train.columns.str.match("weather")]


# In[51]:


train


# In[23]:


kf = KFold(n_splits=5,random_state=777)
tr = dat[dat["t"]==1][cols]
tr.head()


# In[24]:


kf = KFold(n_splits=5,random_state=777)
tr = dat[dat["t"]==1][cols]

trains = []
tests = []
for train_index, test_index in kf.split(tr):
    tr.loc[train_index,"tt"] = 1
    tr.loc[test_index,"tt"] = 0
    tr["tt"] = tr["tt"].astype(np.int)
    tmp = pd.get_dummies(tr)
    
    trainX = tmp[tmp["tt"]==1]
    del trainX["tt"]
    testX = tmp[tmp["tt"]==0]
    del testX["tt"]
    y_train = tmp[tmp["tt"]==1]["y"]
    y_test = tmp[tmp["tt"]==0]["y"]
    
    model1, model2 = learning(trainX, y_train)
    
    pred_train = model1.predict(trainX["days"].values.reshape(-1,1)) + model2.predict(trainX.iloc[:, ~trainX.columns.str.match("y")])
    pred_test = model1.predict(testX["days"].values.reshape(-1,1)) + model2.predict(testX.iloc[:, ~testX.columns.str.match("y")])
    print(pred_train)
    print("TRAIN:",MSE(y_train,pred_train)**0.5, "VARIDATE",MSE(y_test, pred_test)**0.5)
    trains.append(MSE(y_train,pred_train)**0.5)
    tests.append(MSE(y_test, pred_test)**0.5)
print("AVG")
print(np.array(trains).mean(), np.array(tests).mean())


# In[25]:


kf.split(tr)


# In[ ]:





# In[ ]:





# In[ ]:




