rm(list=ls())
setwd("/Users/divyasharma/Documents/Edwisor/Project 1/")
getwd()
churn_train=read.csv("Train_data.csv", header = T)
churn_test=read.csv("Test_data.csv", header = T)

#Merge two dataframe to perform data pre-processing
data=rbind(churn_train,churn_test)


#load all the libraries
x = c("ggplot2", "corrgram","randomForest",  "C50","caret","DataCombine")

install.packages(x)
rm(x)
#Missing value analysis:

tot=data.frame(sapply(data,function(x){sum(is.na(x))}))
tot
#There are no missing values in our dataset

##Data Manupulation; convert string categories into factor numeric
for(i in 1:ncol(data)){
  
  if(class(data[,i]) == 'factor'){
    
    data[,i] = factor(data[,i], labels=(1:length(levels(factor(data[,i])))))
  }
}

##Outlier Analysis
cnames = c("number.vmail.messages","total.day.minutes",
           "total.day.calls",
           "total.day.charge","total.eve.minutes",
           "total.eve.calls","total.eve.charge","total.night.minutes",
           "total.night.calls","total.night.charge","total.intl.minutes",
           "total.intl.calls","total.intl.charge","number.customer.service.calls")
 

 
for (i in cnames)
{
boxplot(data[,i] ~ Churn, data = data,
        notch = TRUE, col = "blue",
        xlab="Churn", ylab= i)
}


## Correlation Plot 
 corrgram(data[,numeric_index], order = F,
          upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")
 
## Chi-squared Test of Independence
 factor_index = sapply(data,is.factor)
 factor_data = data[,factor_index]
 
 for (i in 1:4)
 {
   print(names(factor_data)[i])
   print(chisq.test(table(factor_data$Churn,factor_data[,i])))
 }



 ## Dimension Reduction
 data = subset(data, select = -c(phone.number,account.length,state,area.code))
 
 rmExcept('data')
 
 ##################################Feature Scaling################################################
 #Normality check
 
 #hist(data$number.vmail.messages, col='yellow')
 
 #####Normalization
 
 for(i in cnames){
   print(i)
   data[,i]=(data[,i]-min(data[,i]))/(max(data[,i]-min(data[,i])))
 }
####Partitioning dataset into train and test 
train=data[1:3333,]
test=data[3334:5000,]

######Model Development#############
###Decision Tree#########
C50m=C5.0(Churn~.,train,trials=100,rules=TRUE)

#Summary of Decision Tree model
summary(C50m)

#Predictions for test case
C50_Pred=predict(C50m,test[,-17],type="class")

##Confusion Matrix for D-Tree
ConfM_C50=table(test$Churn,C50_Pred)
confusionMatrix(ConfM_C50)

#Accuracy:96.4%
#FNR:23.66% 


##KNN Implementation
#Import class library to access knn method
library(class)

#Predict test data
KNN_Predict=knn(train[,1:16],test[,1:16],train$Churn,k=7)

#Confusion matrix
ConfM=table(KNN_Predict,test$Churn)
ConfM=table(KNN_Predict)

#Accuracy : 90.16%
sum(diag(ConfM))/nrow(test)

#False Negative rate : 15.11%
#FNR = FN/FN+TP 

######################Random Forest
Rf=randomForest(Churn~.,train,importance=TRUE,ntree=500)

#Extract rules fromn random forest
#transform rf object to an inTrees' format
treeList=RF2List(Rf)
# 
# #Extract rules
 exec=extractRules(treeList,train[,-17])  # R-executable conditions
# 
# #Visualize some rules
 exec[1:2,]
# 
# #Make rules more readable:
 readableRules=presentRules(exec,colnames(train))
 readableRules[1:2,]
# 
# #Get rule metrics
 ruleMetric=getRuleMetric(exec,train[,-17],train$Churn)  # get rule metrics
# 
# #evaulate few rules
 ruleMetric[1:2,]

#Predict test data using random forest model
RF_Predictions=predict(RF_model,test[,-17])

##Evaluate the performance of classification model
ConfMatrix_RF=table(test$Churn,RF_Predictions)
confusionMatrix(ConfMatrix_RF)
confusionMatrix
#False Negative rate
#FNR = FN/FN+TP 

#Accuracy = 96.16%
#FNR = 26.33%

