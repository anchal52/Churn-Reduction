
#Load libraries
import os
import pandas as pd
import numpy as np
from fancyimpute import KNN   
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

#Set working directory
os.chdir("/Users/anchalsharma/Documents/Edwisor/Project 1")
#Load data
Churn_train = pd.read_csv("Train_data.csv")
Churn_test = pd.read_csv("Test_data.csv")
data=Churn_train.append(Churn_test)


# ## Missing Value Analysis

#Creating a dataframe with missing percentage
missing_val = pd.DataFrame(data.isnull().sum())
missing_val

data['state']=data['state'].astype('object')


#Assigning levels to the categories
lis = []
for i in range(0,data.shape[1]):
    if(data.iloc[:,i].dtypes=='object'):
        data.iloc[:,i]=pd.Categorical(data.iloc[:,i])
        data.iloc[:,i]=data.iloc[:,i].cat.codes 
        data.iloc[:,i]=data.iloc[:,i].astype('object')
        lis.append(data.columns[i])      


# ## Outlier Analysis

# #Plot boxplot to visualize Outliers
#%matplotlib inline  
#plt.boxplot(data['total eve minutes'], notch=True)

#save numeric names
cnames=["number vmail messages","total day minutes",
        "total day calls","total day charge",
        "total eve minutes","total eve calls",
        "total eve charge","total night minutes",
        "total night calls","total night charge",
        "total intl minutes","total intl calls","total intl charge",
        "number customer service calls"]


# ## Feature Selection

data = data.drop(['state','area code','phone number','account length'], axis=1)

##Correlation analysis
#Correlation plot
df = data.loc[:,cnames]

#Generating correlation matrix
corr = df.corr()

#Plot using seaborn library
import seaborn as sns
sns.heatmap(corr,vmin=-1,vmax=1)

#Chisquare test of independence
#Save categorical variables
cat_names = ["international plan", "voice mail plan"]

#loop for chi square values
for i in cat_names:
    print(i)
    chi2,p,dof,ex=chi2_contingency(pd.crosstab(data['Churn'],data[i]))
    print(p)


## Feature Scaling

#dt = data.copy()
#data = dt.copy()

#Normality check
get_ipython().run_line_magic('matplotlib', 'inline')
for i in cnames:
    plt.hist(data[i], bins='auto')

#Nomalisation
for i in cnames:
    print(i)
    data[i]=(data[i]-min(data[i]))/(max(data[i])-min(data[i]))


# ## Model Development

# ## Decision Tree

#Import Library for decision tree
from sklearn import tree

data.head()


# In[ ]:


#replace target categories with True and False
data['Churn']=data['Churn'].replace(0,'False.')
data['Churn']=data['Churn'].replace(1,'True.')

#Divide the data into Train and Test
X_train=data.iloc[0:3333,0:16]
X_test=data.iloc[3333:,0:16]
y_train=data.iloc[0:3333,16]
y_test=data.iloc[3333:,16]

#Decision Tree
C50m = tree.DecisionTreeClassifier(criterion='entropy').fit(X_train, y_train)

#predict test cases
C50_Pred=C50m.predict(X_test)

#build confusion matrix
CM = pd.crosstab(y_test, C50_Pred)

#Assign TP, TN, FP, FN
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]

#check accuracy of model:92
((TP+TN)*100)/(TP+TN+FP+FN)

#False Negative rate:30
#(FN*100)/(FN+TP)

#Results
#Accuracy: 92
#FNR: 30

(FN*100)/(FN+TP)

## Random Forest

#replace target categories with True and False
data['Churn']=data['Churn'].replace(0,'False.')
data['Churn']=data['Churn'].replace(1,'True.')


X_train=data.iloc[0:3333,0:16]
X_test=data.iloc[3333:,0:16]
y_train=data.iloc[0:3333,16]
y_test=data.iloc[3333:,16]

#Random Forest
from sklearn.ensemble import RandomForestClassifier

RF_model = RandomForestClassifier(n_estimators=500).fit(X_train, y_train)

RF_Predictions = RF_model.predict(X_test)

#build confusion matrix
# from sklearn.metrics import confusion_matrix 
# CM = confusion_matrix(y_test, y_pred)
CM = pd.crosstab(y_test, RF_Predictions)

#let us save TP, TN, FP, FN
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]

#check accuracy of model
#accuracy_score(y_test, y_pred)*100
((TP+TN)*100)/(TP+TN+FP+FN)

#False Negative rate :25
#(FN*100)/(FN+TP)

#Accuracy: 95(100 trees)96 (500 trees)
#FNR: 28(100 trees) 25(500 Trees)

(FN*100)/(FN+TP)


## KNN Implementation

data['Churn'] = data['Churn'].replace( 'False.',1)
data['Churn'] = data['Churn'].replace( 'True.',2)

X_train=data.iloc[0:3333,0:16]
X_test=data.iloc[3333:,0:16]
y_train=data.iloc[0:3333,16]
y_test=data.iloc[3333:,16]

from sklearn.neighbors import KNeighborsClassifier

KNN_model=KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train)

#predict test cases
KNN_Predictions=KNN_model.predict(X_test)

#build confusion matrix
CM=pd.crosstab(y_test, KNN_Predictions)

#let us save TP, TN, FP, FN
TN=CM.iloc[0,0]
FN=CM.iloc[1,0]
TP=CM.iloc[1,1]
FP=CM.iloc[0,1]

#check accuracy of model
((TP+TN)*100)/(TP+TN+FP+FN)

#False Negative rate 
#(FN*100)/(FN+TP)

#Accuracy: 88/90/91/91 n_Neighbours_value:(1/3/5/7)
#FNR: 49/51/52/56


## Naive Bayes

data['Churn']=data['Churn'].replace('False.',1)
data['Churn']=data['Churn'].replace('True.',2)

X_train=data.iloc[0:3333,0:16]
X_test=data.iloc[3333:,0:16]
y_train=data.iloc[0:3333,16]
y_test=data.iloc[3333:,16]

from sklearn.naive_bayes import GaussianNB

#Naive Bayes implementation
NB_m=GaussianNB().fit(X_train,y_train)

#predict test cases
NB_Pred=NB_m.predict(X_test)

#Build confusion matrix
CM=pd.crosstab(y_test,NB_Pred)

#let us save TP, TN, FP, FN
TN=CM.iloc[0,0]
FN=CM.iloc[1,0]
TP=CM.iloc[1,1]
FP=CM.iloc[0,1]

#To calclulate accuracy of model
((TP+TN)*100)/(TP+TN+FP+FN)

#False Negative rate 
#(FN*100)/(FN+TP)

#Accuracy: 87
#FNR: 47

