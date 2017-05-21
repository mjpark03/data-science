
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd


# # Load Dataset

# In[2]:

train = pd.read_csv("../data/train.csv", index_col = "PassengerId")

print(train.shape)
train.head(1)


# In[3]:

test = pd.read_csv("../data/test.csv", index_col = "PassengerId")

print(test.shape)
test.head(1)


# # Preprocessing

# In[4]:

# merge train and test dataset

combi = pd.concat([train, test])

print(combi.shape)
combi.head(1)


# In[5]:

# encode Sex

combi["Sex_encode"] = (combi["Sex"] == "male").astype(int)

print(combi.shape)
combi[["Sex", "Sex_encode"]].head(2)


# In[6]:

# encode Embarked

embarked = pd.get_dummies(combi["Embarked"], prefix="Embarked").astype(np.bool)

combi = pd.concat([combi, embarked], axis=1)

print(combi.shape)
combi[["Embarked", "Embarked_C", "Embarked_Q", "Embarked_S"]].head()


# In[7]:

combi["Family"] = combi["SibSp"] + combi["Parch"]

print(combi.shape)
combi.head(2)


# In[8]:

# fill out NaN fare

mean_fare = train["Fare"].mean()

print("mean fare = ${mean_fare:.3f}".format(mean_fare=mean_fare))


# In[9]:

combi["Fare_fillout"] = combi["Fare"]

combi.loc[pd.isnull(combi["Fare"]), "Fare_fillout"] = mean_fare

missing_fare = combi[pd.isnull(combi["Fare"])]


# In[10]:

# split dataset into train

train = combi[pd.notnull(combi["Survived"])]

train.head(1)


# In[11]:

# split dataset into test

test = combi[pd.isnull(combi["Survived"])]

test.drop("Survived", axis=1, inplace=True)

test.head(1)


# # Score

# In[12]:

# make prediction model through decision tree using train.csv
# predict score using test.csv
# note: cross validation

feature_names = ["Pclass", "Sex_encode", "Fare_fillout", "Embarked_C", "Embarked_Q", "Embarked_S", "Family"]

X_train = train[feature_names]

print(X_train.shape)
X_train.head()


# In[13]:

label_name = "Survived"

y_train = train[label_name]

print(y_train.shape)
y_train.head()


# In[14]:

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=5,
                               random_state=37)
model


# In[15]:

from sklearn.cross_validation import cross_val_score

score = cross_val_score(model, X_train, y_train, cv=100).mean()
print("Score = {score:.5f}".format(score=score))


# # Predict

# In[16]:

X_test = test[feature_names]

print(X_test.shape)
X_test.head(1)


# In[17]:

# mean_fare = train["Fare"].mean()
# X_test.loc[pd.isnull(X_test["Fare"]), "Fare"] = mean_fare

model.fit(X_train, y_train)

prediction = model.predict(X_test)

print(prediction.shape)
prediction[:20]


# # Submit

# In[18]:

submission = pd.read_csv("../data/gender_submission.csv", index_col = "PassengerId")

submission["Survived"] = prediction.astype(np.int32)

print(submission.shape)
submission.head()


# In[19]:

submission.to_csv("../data/kaggle-titanic-submission")


# In[ ]:



