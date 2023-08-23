install.packages("caret")
install.packages("MASS")
install.packages("forecast", dependencies = TRUE)
install.packages("MLmetrics")
install.packages("leaps")
install.packages("DALEX")
install.packages("InformationValue")
install.packages("ROCR")
install.packages("gains")
library(ROCR)
library(InformationValue)
library(DALEX)
library(leaps)
library(MLmetrics)
library(forecast)
library(caret)
library(readxl)
fraud_Mix <- read_excel("~/Desktop/myproject/FRAUD_Mix.xlsx")
View(fraud_Mix)
summary(fraud_Mix)
mean(fraud_Mix$Sales)
fraud_Mix$new_y <- ifelse(fraud_Mix$Sales > 697.1303, 0, 1)

fraud_logreg<- fraud_Mix[-c(1,2)]
fraud_logreg

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
fraud_log_outlier <- remove_outlier(fraud_logreg, c(6))
nrow(fraud_log_outlier)

##dividing data into training and validation set
set.seed(1)
sample_data <- sample(c(1:647), 457)
fraud_train_data <- fraud_log_outlier[sample_data, ]
fraud_test_data <- fraud_log_outlier[-sample_data, ]
options (scipen=999)
logit_reg <- glm(new_y~., data = fraud_train_data, family = "binomial")
summary(logit_reg)
varImp(logit_reg)

log_pred <- predict(logit_reg, fraud_test_data, type = "response")

tt<-table(ifelse(log_pred> 0.5, 1, 0), fraud_test_data$new_y)

sum(diag(tt))/sum(tt)

t_value <- data.frame(facebook = 3.987, TikTok = 3.498, Google = 0.734, Twiiter = 2.698
           , TTD = 1.9872, Openslate = 0.781, Youtube = 7.122, connected_tv = 0.2442)


tab <- matrix(c(3.987,3.498, 0.734,2.698, 1.9872,0.781,7.122,0.2442), ncol=8, byrow=TRUE)
colnames(tab) <- c("no_transaction" + "frequency" + "average_trans" + "time" + "location" + "address"
                   + "v1" + "v2")

t_value <- as.table(tab)
barplot(t_value, col = 'blue', main = "t_statistics Chart with all independent variables", las = 2)

tab2 <- matrix(c(0.000067,0.000467, 0.432,0.0469,0.006973,0.4348,0.0000000000000106,0.80709), ncol=8, byrow=TRUE)
colnames(tab2) <- c("no_transaction" + "frequency" + "average_trans" + "time" + "location" + "address"
                    + "v1")

p_value <- as.table(tab2)
barplot(p_value, col = 'red', main = "P_Value Chart with all indepedent variables", las = 2)

plotROC(fraud_test_data$new_y, log_pred)

confusionMatrix(fraud_test_data$new_y, log_pred, threshold = 0.5)

#######Removing the weakest independent variable(V2)
logit_reg2 <- glm(new_y ~ no_transaction + frequency data = fraud_train_data, family = "binomial")
summary(logit_reg2)
varImp(logit_reg2)

log_pred2 <- predict(logit_reg2, fraud_test_data, type = "response")

tt2<-table(ifelse(log_pred2> 0.5, 1, 0), fraud_test_data$new_y)

sum(diag(tt2))/sum(tt2)

plotROC(fraud_test_data$new_y, log_pred2)

confusionMatrix(fraud_test_data$new_y, log_pred2, threshold = 0.5)

tab3 <- matrix(c(0.0000598,0.00038, 0.47,0.041 ,0.006873,0.451,0.0000000000000106), ncol=7, byrow=TRUE)
colnames(tab3) <- c("no_transaction" + "frequency" + "average_trans" + "time" + "location" + "address"
                    + "v1")

p_value <- as.table(tab3)
barplot(p_value, col = 'red', main = "P_Value Chart with all indepedent variables", las = 2)
#######Removing the weakest independent variable(Average_trans and v1)
logit_reg3 <- glm(new_y~.-connected_tv-Google, data = fraud_train_data, family = "binomial")
summary(logit_reg3)
varImp(logit_reg3)

log_pred3 <- predict(logit_reg3, fraud_test_data, type = "response")

tt3<-table(ifelse(log_pred3> 0.6, 1, 0), fraud_test_data$new_y)

sum(diag(tt3))/sum(tt3)

plotROC(fraud_test_data$new_y, log_pred3)

confusionMatrix(fraud_test_data$new_y, log_pred3, threshold = 0.5)
tab4 <- matrix(c(0.0000208,0.00030, 0.045 ,0.00781,0.78,0.0000000000000106), ncol=6, byrow=TRUE)
colnames(tab4) <- c("no_transaction" + "frequency" +  "time" + "location" + "address"
                  )

p_value <- as.table(tab4)
barplot(p_value, col = 'red', main = "P_Value Chart with all indepedent variables", las = 2)
#######Removing the weakest independent variable(Average_trans, v2 and v1)
logit_reg4 <- glm(new_y~.-Average_trans-v2-v1, data = fraud_train_data, family = "binomial")
summary(logit_reg4)
varImp(logit_reg4)

log_pred4 <- predict(logit_reg4, fraud_test_data, type = "response")

tt4<-table(ifelse(log_pred4> 0.6, 1, 0), fraud_test_data$new_y)

sum(diag(tt4))/sum(tt4)

plotROC(fraud_test_data$new_y, log_pred4)

confusionMatrix(fraud_test_data$new_y, log_pred4, threshold = 0.6)

####Gain chart
library(gains)
gain <- gains(actual=fraud_test_data$new_y, predicted=log_pred4, groups = 10)
plot(c(0, gain$cume.pct.of.total* sum(fraud_test_data$new_y))~c(0,gain$cume.obs),
      xlab=" # cases",ylab="cumulative", main="Gain Chart(Performance) of a logistic Model",type = "l", col = 'red')
lines (c(0, sum(fraud_test_data$new_y))~c(0, dim(fraud_test_data)[1]), lty = 2)

