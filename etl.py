import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rnd

from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgm

data_path = 'C:\Pycharm/titanic/data'

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

###Load data

data_train = pd.read_csv(data_path + '/' + 'train.csv')
data_test = pd.read_csv(data_path + '/' + 'test.csv')
data_gender = pd.read_csv(data_path + '/' + 'gender_submission.csv')

###ETL

clean_train = data_train.drop(['Name', 'Cabin','Ticket'],axis=1)
dummy_embarked = pd.get_dummies(clean_train.Embarked, prefix='embarked')
clean_train_dummy = pd.merge(clean_train,dummy_embarked,left_index=True, right_index=True)
clean_train_dummy.drop(['Embarked'],axis=1,inplace=True)
dummy_sex = pd.get_dummies(clean_train_dummy.Sex, prefix='sex')
clean_train_dummy = pd.merge(clean_train_dummy,dummy_sex,left_index=True, right_index=True)
clean_train_dummy.drop(['Sex'],axis=1,inplace=True)
dummy_class = pd.get_dummies(clean_train_dummy.Pclass, prefix='class')
clean_train_dummy = pd.merge(clean_train_dummy,dummy_class,left_index=True, right_index=True)
clean_train_dummy.drop(['Pclass'],axis=1,inplace=True)
clean_train_dummy.dropna(axis=1,inplace=True)
###Data Visualization

#plt.boxplot(clean_train_dummy.Fare)
#plt.boxplot(clean_train_dummy.Age)

#plt.hist(clean_train_dummy.Age)
#plt.hist(clean_train_dummy.Fare)

###Logistic Regression

#clean_train_dummy['data_istrain'] = np.random.uniform(0,1,len(clean_train_dummy.Survived)) < 0.75

#train, test = clean_train_dummy[clean_train_dummy.data_istrain == True], clean_train_dummy[clean_train_dummy.data_istrain == False]

#train.drop(['data_istrain'],axis=1,inplace=True)
#test.drop(['data_istrain'],axis=1,inplace=True)

x_columns = [v for v in clean_train_dummy.columns if 'Survived' not in v]
y_columns = ['Survived']

X = clean_train_dummy[x_columns]
Y = clean_train_dummy[y_columns]

logit_model = linear_model.LogisticRegression()
logit_model.fit(X,Y)

#logit_model.score(X,Y)
#pd.DataFrame(list(zip(X.columns,np.transpose(logit_model.coef_))))

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=0)
lm = linear_model.LogisticRegression()
lm.fit(X_train,Y_train)
#lm.score(X,Y)
prediction = lm.predict(X_test)
metrics.accuracy_score(Y_test,prediction)

###Random Forest
forest = RandomForestClassifier(n_jobs=2, oob_score=True, n_estimators=1000)
forest.fit(X_train,Y_train)
prediction = forest.predict(X_test)
prediction = np.transpose(prediction)
forest.oob_score_

###Lightgbm

X_final = pd.merge(X_train,Y_train,how='left',left_index=True, right_index=True)

parameters_binary = {"boosting_type" : "gbdt",
                     "objective" : "binary",
                     "seed" : 9520,
                     "alpha" : 10,
                     "learning_rate" : 0.01,
                     "metric" : ["binary_logloss"]}

# parameters_binary = {"boosting_type" : "gbdt",
#                      "objective" : "binary",
#                      "seed" : 9520,
#                      "alpha" : 10,
#                      "learning_rate" : 0.01,
#                      "metric" : ["binary_logloss"],
#                      "feature_fraction": 0.4,
#                      "bagging_fraction": 0.4,
#                      "bagging_freq": 1,
#                      "num_leaves": 7,
#                      "num_threads": 21}

train_data = lgm.Dataset(X_final[x_columns], label=X_final[y_columns].values)

eval_results = {}
cb_evaluation = lgm.record_evaluation(eval_results)

model = lgm.train(parameters_binary,
                  train_data,
                  verbose_eval=10)
# model = lgm.train(parameters_binary,
#                   train_data,
#                   verbose_eval=10,
#                   early_stopping_rounds=5,
#                   num_boost_round=9999,
#                   callbacks=[cb_evaluation])

prediction = model.predict(X_test)


