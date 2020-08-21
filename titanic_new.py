# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 18:49:22 2020

@author: ANNAM SHIVA KRISHNA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
#

p = print
titanic = pd.read_csv("F://Analytics path notes//titanic_train.csv")
p(titanic.head())
p("\n" + "no. of passengers" ,titanic.shape[0])
dtypes =  titanic.dtypes

# 
# Existing columns
print("No. columns:", titanic.shape[1], "\n")
p(titanic.columns)

# Change to desired column names
new_col_names = ['Passenger_ID', 'Survived', 'Class', 'Name', 'Sex', 'Age',
       'Siblings_spouses_aboard', 'Parents_children_aboard', 'Ticket', 'Fare', 'Cabin_num', 'Port_of_Embarkation']
titanic.columns = new_col_names
p(titanic.columns)
                                                                           
# changing Survived and class and 
titanic.Survived = titanic.Survived.map({0:'Died',1:'Survived'})
titanic.Class = titanic.Class.map({1:'first class',2:'second class',3:'third class'}) 

# information of nans
nans = titanic.isnull().sum()

# sex vs survived
count = 0
coun1 = 0
sex_survived = titanic[['Sex','Survived']]
for x in sex_survived['Survived']       :
    if (x == 'Survived'):
        coun1 = coun1 + 1
        #print('green',coun1)
    else:
        count = count + 1
        #print("red",count)
    
countplot0 = sns.countplot(x=titanic.Sex,hue=sex_survived.Survived,palette=['red','green'])


for p in countplot0.patches:
    countplot0.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.show()
male_deaths = len(sex_survived[((sex_survived['Sex'] == 'male') & (sex_survived['Survived'] == 'Died'))])
female_deaths = len(sex_survived[((sex_survived['Sex'] == 'female') & (sex_survived['Survived'] == 'Died'))])
print("male deaths percentage",male_deaths/len(sex_survived[((sex_survived['Sex'] == 'male'))])*100)
print("female deaths percentage",female_deaths/len(sex_survived[((sex_survived['Sex'] == 'female'))])*100)

obs_sex = pd.DataFrame(['Non-survivors, male']*male_deaths + ['Survivors, male']*(len(sex_survived[((sex_survived['Sex'] == 'male'))]) - male_deaths) +\
                        ['Non-survivors, female']*female_deaths + ['Survivors, female']*(len(sex_survived[((sex_survived['Sex'] == 'female'))]) - female_deaths))
obs_sex
obs_sex_table = pd.crosstab(index=obs_sex[0], columns="counts")
print(obs_sex_table)

class_survived = titanic[['Class','Survived']]
first_class_survived = len(class_survived[((class_survived["Class"] == "first class") & (class_survived["Survived"] == "Survived"))])
Second_class_survived = len(class_survived[((class_survived["Class"] == "second class") & (class_survived["Survived"] == "Survived"))])
third_class_survived = len(class_survived[((class_survived["Class"] == "third class") & (class_survived["Survived"] == "Survived"))])
first_class_death = len(class_survived[((class_survived["Class"] == "first class") & (class_survived["Survived"] == "Died"))])
Second_class_death = len(class_survived[((class_survived["Class"] == "second class") & (class_survived["Survived"] == "Died"))])
third_class_death = len(class_survived[((class_survived["Class"] == "third class") & (class_survived["Survived"] == "Died"))])

g = sns.countplot(x=class_survived.Class ,hue=class_survived.Survived,palette=["red","green"])

for p in g.patches:
    g.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')




print("first class death per",(first_class_death/len(class_survived[(class_survived["Class"] == "first class")]))*100)
print("second class death",(Second_class_death/len(class_survived[(class_survived["Class"] == "second class")]))*100)
print("third class death",(third_class_death/len(class_survived[(class_survived["Class"] == "third class")]))*100)

fig, axes = plt.subplots(1,1)
sns.boxplot(x = titanic.Survived,y=titanic.Fare,hue=titanic.Class)
plt.show()
fig.set_size_inches(5,5)
sns.boxplot(titanic.Survived,y=titanic.Fare,hue=titanic.Port_of_Embarkation,saturation=0.75)
#sns.swarmplot(titanic.Survived,y=titanic.Fare, color=".25")
plt.show()

outliers_free = titanic[titanic['Fare']<170].reset_index(drop=True)

# ramoving na values 
outliers_free1 = outliers_free.dropna(axis=0)

dummy1 = pd.get_dummies(outliers_free1['Passenger_ID'],prefix='Passenger_ID',drop_first=True)
outliers_free1 = pd.concat([outliers_free1,dummy1],axis=1)

dummy2 = pd.get_dummies(outliers_free1['Class'],prefix='Class',drop_first=True)
outliers_free1 = pd.concat([outliers_free1,dummy2],axis=1)

dummy3 = pd.get_dummies(outliers_free1['Name'],prefix='Name',drop_first=True)
outliers_free1 = pd.concat([outliers_free1,dummy3],axis=1)

dummy4 = pd.get_dummies(outliers_free1['Sex'],prefix='Sex',drop_first=True)
outliers_free1 = pd.concat([outliers_free1,dummy4],axis=1)


dummy5 = pd.get_dummies(outliers_free1['Siblings_spouses_aboard'],prefix='Siblings_spouses_aboard',drop_first=True)
outliers_free1 = pd.concat([outliers_free1,dummy5],axis=1)

dummy6 = pd.get_dummies(outliers_free1['Parents_children_aboard'],prefix='Parents_children_aboard',drop_first=True)
outliers_free1 = pd.concat([outliers_free1,dummy6],axis=1)


dummy7 = pd.get_dummies(outliers_free1['Ticket'],prefix='Ticket',drop_first=True)
outliers_free1 = pd.concat([outliers_free1,dummy7],axis=1)


dummy8 = pd.get_dummies(outliers_free1['Cabin_num'],prefix='Cabin_num',drop_first=True)
outliers_free1 = pd.concat([outliers_free1,dummy8],axis=1)

dummy9 = pd.get_dummies(outliers_free1['Port_of_Embarkation'],prefix='Port_of_Embarkation',drop_first=True)
outliers_free1 = pd.concat([outliers_free1,dummy9],axis=1)

outliers_free1 = outliers_free.drop(['Passenger_ID','Class', 'Name','Siblings_spouses_aboard', 'Parents_children_aboard', 'Ticket', 'Cabin_num'],1)
#outliers_free1 = outliers_free.drop(['Passenger_ID','Class', 'Name', 'Sex','Siblings_spouses_aboard', 'Parents_children_aboard', 'Ticket', 'Cabin_num', 'Port_of_Embarkation'],1)
outliers_free1 = outliers_free1.dropna(how='any')
outliers_free1.Survived = outliers_free1.Survived.map({'Died':0,'Survived':1})
outliers_free1.Sex = outliers_free1.Sex.map({'female':0,'male':1})
outliers_free1.Port_of_Embarkation = outliers_free1.Port_of_Embarkation.map({'C':0,'S':1,'Q':2})


train_data = outliers_free1.drop(["Survived","Age"],axis=1)
target_data = outliers_free1.Survived
logit_model=sm.Logit(target_data,train_data)
result=logit_model.fit()
print(result.summary2())

# train and test split

X_train, X_test, y_train, y_test = train_test_split(train_data,target_data, test_size=0.2, random_state=10)

# building a model
#sm = SMOTE(random_state = 2)
#X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE

"""
num_folds = 3
seed = 7
kfold = KFold(n_splits=num_folds, random_state=seed,shuffle=True)
logsk = LogisticRegressionCV(solver='liblinear', cv=5, random_state=0)
results = cross_val_score(logsk, train_data, target_data, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)",results.mean()*100.0, results.std()*100.0)"""

logsk = LogisticRegressionCV(solver='liblinear', cv=5, random_state=0)
rfe = RFE(logsk, n_features_to_select=2)
fit = rfe.fit(X_train, y_train)
support = fit.support_
ranking = fit.ranking_

fit1 = logsk.fit(X_train, y_train)
class1 = fit1.classes_
coef = fit1.coef_
intercept = fit1.intercept_
n_iter = fit1.n_iter_

y_pred = logsk.predict(X_test)
y_pred_proba = logsk.predict_proba(X_test)[:, 1]
new_score=r2_score(y_test,y_pred_proba)

acc = accuracy_score(y_test,y_pred)
conf_m = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)



"""roc = roc_auc_score(y_test, y_pred)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % roc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
"""

from sklearn.metrics import RocCurveDisplay
y_score = logsk.decision_function(X_test)

fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=logsk.classes_[1])
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()


print((y_pred == 1).sum())
print((y_test == 1).sum())
print((y_pred == 0).sum())
print((y_test == 0).sum())

values = np.where((y_pred_proba > 0.5))
values1 = np.where((y_test == 1))
print(len(values[0]),len(values1[0]))

sur_port_group = outliers_free.groupby(["Survived","Class",'Port_of_Embarkation'])
print(sur_port_group.head(10))

for t in sur_port_group.groups:
    print(t)

print(sur_port_group.Fare.describe(percentiles=[0.5]))