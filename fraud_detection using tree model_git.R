install.packages("caret")
install.packages("MASS")
install.packages("forecast", dependencies = TRUE)
install.packages("MLmetrics")
install.packages("leaps")
install.packages("DALEX")
install.packages("InformationValue")
install.packages("ROCR")
install.packages("gains")
install.packages("neuralnet")
install.packages("rpart.plot") 
install.packages("randomForest")
install.packages("gbm")  
install.packages("adabag")
install.packages("ipred") 
library(adabag)
library(ipred) 
library(gbm)
library(randomForest)
library(neuralnet)
library(nnet)
library(ROCR)
library(InformationValue)
library(DALEX)
library(leaps)
library(MLmetrics)
library(forecast)
library(caret)
library(readxl)
Fraud_Mix <- read_excel("~/Desktop/myproject/Fraud_Mix.xlsx")
View(Fraud_Mix)
summary(Fraud_Mix)
mean(Fraud_Mix$Sales)
Fraud_Mix$new_y <- ifelse(Fraud_Mix$Sales > 697.1303, 0, 1)

Fraud_nn<- Fraud_Mix[-c(1,2)]
Fraud_nn

# create detect outlier function
detect_outlier <- function(x) {
  Quantile1 <- quantile(x, probs=.25)
  Quantile3 <- quantile(x, probs=.75)
  IQR = Quantile3-Quantile1
  x > Quantile3 + (IQR*1.5) | x < Quantile1 - (IQR*1.5)
}

remove_outlier <- function(dataframe,
                           columns=names(dataframe)) {
  for (col in columns) {
    dataframe <- dataframe[!detect_outlier(dataframe[[col]]), ]
  }
  
  # return dataframe
  print("Remove outliers")
  print(dataframe)
}
Fraud_nn_outlier <- remove_outlier(Fraud_nn, c(6))
nrow(Fraud_nn_outlier)



##### Data preprocessing using standardization
Fraud_nn_outlier$one <- Fraud_nn_outlier$new_y ==1
Fraud_nn_outlier$zero <- Fraud_nn_outlier$new_y ==0

set.seed(1)
sample_data <- sample(c(1:647), 457)
Fraud_train_data <- Fraud_nn_outlier[sample_data, ]
Fraud_test_data <- Fraud_nn_outlier[-sample_data, ]

# The Classification Tree Model
library(rpart)
library(rpart.plot)
#-------------------------------------------------------------------------------------------------------------------------------
#first Tree
set.seed(1)
Frd.rpart.1 <- rpart(new_y ~ No_transaction + frequency + Average_trans + Time + location + IP_address
                      + V1 + V2, data = Fraud_train_data, method = "class")

prp(Frd.rpart.1, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10)
prp(Frd.rpart.1)

# prediction 
Frd.rpart.1.pred <- predict(Frd.rpart.1, Fraud_train_data, type = "class") 
# Confusion Matrix
Frd.rpart.Metics <- table("Predicted" = Frd.rpart.1.pred, "Actual" = Fraud_train_data$new_y)

# Perfromance Metrics
Tree_1 <- t(data.frame("Accuracy" = sum(diag(Frd.rpart.Metics))/sum(Frd.rpart.Metics),
                       "Error" = 1 - (sum(diag(Frd.rpart.Metics))/sum(Frd.rpart.Metics)),
                       "Sensitivity" = Frd.rpart.Metics[2,2] / colSums(Frd.rpart.Metics)[2],
                       "Specificity" = Frd.rpart.Metics[1,1] / colSums(Frd.rpart.Metics)[1],
                       "Precision" = Frd.rpart.Metics[2,2] / rowSums(Frd.rpart.Metics)[2],
                       "F1_Score" = 2 * ((Frd.rpart.Metics[2,2] / colSums(Frd.rpart.Metics)[2])*(Frd.rpart.Metics[2,2] / rowSums(Frd.rpart.Metics)[2]))/
                         ((Frd.rpart.Metics[2,2] / colSums(Frd.rpart.Metics)[2])+(Frd.rpart.Metics[2,2] / rowSums(Frd.rpart.Metics)[2])),
                       "Success_Class" = 1
))



##deeper Tree
set.seed(1)
Frd.rpart.2 <- rpart(new_y ~ No_transaction + frequency + Average_trans + Time + location + IP_address
                     + V1 + V2, data = Fraud_train_data, method = "class", cp = 0, minsplit = 1)

prp(Frd.rpart.2, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10)
prp(Frd.rpart.2)

# prediction on training data
Frd.rpart.2.pred <- predict(Frd.rpart.2, Fraud_train_data, type = "class") 
# Confusion Matrix
Frd.rpart.Metics.2 <- table("Predicted" = Frd.rpart.1.pred, "Actual" = Fraud_train_data$new_y)

# Perfromance Metrics
Tree_2 <- t(data.frame("Accuracy" = sum(diag(Frd.rpart.Metics.2))/sum(Frd.rpart.Metics.2),
                       "Error" = 1 - (sum(diag(Frd.rpart.Metics.2))/sum(Frd.rpart.Metics.2)),
                       "Sensitivity" = Frd.rpart.Metics.2[2,2] / colSums(Frd.rpart.Metics.2)[2],
                       "Specificity" = Frd.rpart.Metics.2[1,1] / colSums(Frd.rpart.Metics.2)[1],
                       "Precision" = Frd.rpart.Metics.2[2,2] / rowSums(Frd.rpart.Metics.2)[2],
                       "F1_Score" = 2 * ((Frd.rpart.Metics.2[2,2] / colSums(Frd.rpart.Metics.2)[2])*(Frd.rpart.Metics[2,2] / rowSums(Frd.rpart.Metics.2)[2]))/
                         ((Frd.rpart.Metics.2[2,2] / colSums(Frd.rpart.Metics.2)[2])+(Frd.rpart.Metics.2[2,2] / rowSums(Frd.rpart.Metics.2)[2])),
                       "Success_Class" = 1
))

#Validation on the test dataset
Frd.rpart.2.test <- predict(Frd.rpart.2, Fraud_test_data, type = "class") 
length(Frd.rpart.2.test)
length(Fraud_test_data$new_y)
# Confusion Matrix
Frd.rpart.Metics.3 <- table(Frd.rpart.2.test, Fraud_test_data$new_y)

Tree_3 <- t(data.frame("Accuracy" = sum(diag(Frd.rpart.Metics.3))/sum(Frd.rpart.Metics.3),
                       "Error" = 1 - (sum(diag(Frd.rpart.Metics.3))/sum(Frd.rpart.Metics.3)),
                       "Sensitivity" = Frd.rpart.Metics.3[2,2] / colSums(Frd.rpart.Metics.3)[2],
                       "Specificity" = Frd.rpart.Metics.3[1,1] / colSums(Frd.rpart.Metics.3)[1],
                       "Precision" = Frd.rpart.Metics.3[2,2] / rowSums(Frd.rpart.Metics.3)[2],
                       "F1_Score" = 2 * ((Frd.rpart.Metics.3[2,2] / colSums(Frd.rpart.Metics.3)[2])*(Frd.rpart.Metics.3[2,2] / rowSums(Frd.rpart.Metics.3)[2]))/
                         ((Frd.rpart.Metics.3[2,2] / colSums(Frd.rpart.Metics.3)[2])+(Frd.rpart.Metics.3[2,2] / rowSums(Frd.rpart.Metics.3)[2])),
                       "Success_Class" = 1
))

#____________________________________________________________________________________________________
# RANDOM FOREST
rf.Frd.1 <- randomForest(as.factor(new_y) ~ No_transaction + frequency + Average_trans + Time + location + IP_address
                       + V1 + V2, data = Fraud_train_data,ntree = 500, 
                       mtry = 4, nodesize = 5, importance = TRUE) 

## variable importance plot
varImpPlot(rf.Frd.1, type = 1, main = "Variable Importance Plot")


# confusion matrix (Training data)
rf.Frd.1.pred <- predict(rf.Frd.1, Fraud_train_data)

rf.Frd.tab1 <- table("Predicted" =rf.Frd.1.pred, "Actual" = Fraud_train_data$new_y)
rf.Frd.tab1

###validation dataset
rf.Frd.2.pred <- predict(rf.Frd.1, Fraud_test_data)

rf.Frd.tab2 <- table("Predicted" =rf.Frd.2.pred, "Actual" = Fraud_test_data$new_y)
rf.Frd.tab2

RF_perform <- t(data.frame("Accuracy" = sum(diag(rf.Frd.tab2))/sum(rf.Frd.tab2),
                       "Error" = 1 - (sum(diag(rf.Frd.tab2))/sum(rf.Frd.tab2)),
                       "Sensitivity" = rf.Frd.tab2[2,2] / colSums(rf.Frd.tab2)[2],
                       "Specificity" = rf.Frd.tab2[1,1] / colSums(rf.Frd.tab2)[1],
                       "Precision" = rf.Frd.tab2[2,2] / rowSums(rf.Frd.tab2)[2],
                       "F1_Score" = 2 * ((rf.Frd.tab2[2,2] / colSums(rf.Frd.tab2)[2])*(rf.Frd.tab2[2,2] / rowSums(rf.Frd.tab2)[2]))/
                         ((rf.Frd.tab2[2,2] / colSums(rf.Frd.tab2)[2])+(rf.Frd.tab2[2,2] / rowSums(rf.Frd.tab2)[2])),
                       "Success_Class" = 1
))


#_____________________________________________________________________________________________________
# Boosting Method
install.packages("gbm")     # gradient boosting classification  
library(gbm)

set.seed(1)
Frd.gbm.1 <- gbm(new_y ~ No_transaction + frequency + Average_trans + Time + location + IP_address
                   + V1 + V2, data = Fraud_train_data)
# Predictions (Training Data)
gbm.pred <- predict(Frd.gbm.1 , Fraud_train_data, type="response")
pred.gbm.1 = ifelse(data.frame(gbm.pred) > 0.6, 1, 0)

# Confusion Matrix training 
gbm.table.1 <- table("Predicted" = pred.gbm.1 , "Actual" = Fraud_train_data$new_y)
gbm.table.1


 ##validation with test data
gbm.pred.test <- predict(Frd.gbm.1 , Fraud_test_data, type="response")
pred.gbm.2 = ifelse(data.frame(gbm.pred.test ) > 0.6, 1, 0)

# Confusion Matrix training 
gbm.table.2 <- table("Predicted" = pred.gbm.2 , "Actual" = Fraud_test_data$new_y)
gbm.table.2

GBM_perform <- t(data.frame("Accuracy" = sum(diag(gbm.table.2))/sum(gbm.table.2),
                           "Error" = 1 - (sum(diag(gbm.table.2))/sum(gbm.table.2)),
                           "Sensitivity" = gbm.table.2[2,2] / colSums(gbm.table.2)[2],
                           "Specificity" = gbm.table.2[1,1] / colSums(gbm.table.2)[1],
                           "Precision" = gbm.table.2[2,2] / rowSums(gbm.table.2)[2],
                           "F1_Score" = 2 * ((gbm.table.2[2,2] / colSums(gbm.table.2)[2])*(gbm.table.2[2,2] / rowSums(gbm.table.2)[2]))/
                             ((gbm.table.2[2,2] / colSums(gbm.table.2)[2])+(gbm.table.2[2,2] / rowSums(gbm.table.2)[2])),
                           "Success_Class" = 1
))


#____________________________________________________________________________________________________
# Bagging Method

set.seed(1)
Frd.bb.1 <- bagging(new_y ~ No_transaction + frequency + Average_trans + Time + location + IP_address
                 + V1 + V2, data = Fraud_train_data)
# Predictions (Training Data)
bbm.pred <- predict(Frd.bb.1 , Fraud_train_data, type="response")
pred.bbm.1 = ifelse(data.frame(bbm.pred) > 0.6, 1, 0)

# Confusion Matrix training 
bbm.table.1 <- table("Predicted" = pred.bbm.1 , "Actual" = Fraud_train_data$new_y)
bbm.table.1

##validation with test data
bbm.pred.test <- predict(Frd.bb.1  , Fraud_test_data, type="response")
pred.bbm.2 = ifelse(data.frame(bbm.pred.test ) > 0.6, 1, 0)

# Confusion Matrix training 
bbm.table.2 <- table("Predicted" = pred.bbm.2 , "Actual" = Fraud_test_data$new_y)
bbm.table.2

BBM_perform <- t(data.frame("Accuracy" = sum(diag(bbm.table.2))/sum(bbm.table.2),
                            "Error" = 1 - (sum(diag(bbm.table.2))/sum(bbm.table.2)),
                            "Sensitivity" = bbm.table.2[2,2] / colSums(bbm.table.2)[2],
                            "Specificity" = bbm.table.2[1,1] / colSums(bbm.table.2)[1],
                            "Precision" = bbm.table.2[2,2] / rowSums(bbm.table.2)[2],
                            "F1_Score" = 2 * ((bbm.table.2[2,2] / colSums(bbm.table.2)[2])*(bbm.table.2[2,2] / rowSums(bbm.table.2)[2]))/
                              ((bbm.table.2[2,2] / colSums(bbm.table.2)[2])+(bbm.table.2[2,2] / rowSums(bbm.table.2)[2])),
                            "Success_Class" = 1
))

