
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd


# ## Load Dataset

# In[2]:

train = pd.read_csv("data/train.csv", parse_dates=["datetime"])

train.drop("casual", axis=1, inplace=True)
train.drop("registered", axis=1, inplace=True)

print(train.shape)
train.head()


# In[3]:

train["yyyy"] = train["datetime"].dt.year
train["mm"] = train["datetime"].dt.month
train["hh"] = train["datetime"].dt.hour
train["weekday"] = train["datetime"].dt.dayofweek


# In[4]:

def group_week(day):
    if day >= 4 and day <= 6: 
        return 0
    else:
        return 1
    
train["group_week"] = train["weekday"].apply(group_week)

train.head()


# In[5]:

test = pd.read_csv("data/test.csv", parse_dates=["datetime"])

test["yyyy"] = test["datetime"].dt.year
test["mm"] = test["datetime"].dt.month
test["hh"] = test["datetime"].dt.hour
test["weekday"] = test["datetime"].dt.dayofweek

print(test.shape)
test.head()


# In[6]:

test["group_week"] = test["weekday"].apply(group_week)

test.head()


# ## Score

# In[7]:

feature_names = ["season", "holiday", "workingday", "weather", 
                 "temp", "atemp", "humidity","windspeed", 
                 "yyyy", "mm", "hh", "group_week"]

X_train = train[feature_names]

print(X_train.shape)
X_train.head()


# In[8]:

label_name ="count"

y_train = train[label_name]

print(y_train.shape)
y_train.head()


# In[9]:

from sklearn.tree import DecisionTreeRegressor

seed=37
model = DecisionTreeRegressor(random_state=seed)

model


# ## RMSLE 
# 
# $$ \sqrt{\frac{1}{n} \sum_{i=1}^n (\log(p_i + 1) - \log(a_i+1))^2 } $$

# In[10]:

## implement RMSLE function

from sklearn.metrics import make_scorer

def rmsle(predict, actual):
    predict = np.array(predict)
    actual = np.array(actual)
    
    log_predict = np.log(predict + 1)
    log_actual = np.log(actual + 1)
    
    difference = log_predict - log_actual
    square_difference = np.square(difference)
    mean_square_difference = np.mean(square_difference)
    
    score = np.sqrt(mean_square_difference)
    
    return score 

rmsle_score = make_scorer(rmsle)
rmsle_score
    


# In[11]:

from sklearn.cross_validation import cross_val_score

score = cross_val_score(model, X_train, y_train, cv=20, scoring=rmsle_score).mean()

print("Score = {0:.5f}".format(score))


# ## Submission

# In[12]:

X_test = test[feature_names]

print(X_test.shape)
X_test.head()


# In[13]:

model.fit(X_train, y_train)

prediction = model.predict(X_test)

print(prediction.shape)
prediction[0:5]


# In[14]:

submission = pd.read_csv("data/sampleSubmission.csv")

submission["count"] = prediction

print(submission.shape)
submission.head()


# In[15]:

submission.to_csv("baseline-script.csv", index=False)

