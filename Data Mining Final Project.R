library(glmnet)
library(caret)
library(e1071)

# Set working directory

# Reading in the data
churn <- read.csv("Churn_Modelling.csv")

# Clean the data
churn <- churn[,-c(1,2,3,5,6)]

# Holdout method - splitting the data into 80% train and 20% test, only training the model once
set.seed(721)

train_ind <- sample(seq_len(nrow(churn)), size = 8000)
train <- churn[train_ind, ]
test <- churn[-train_ind, ]

# Taking out response variable of testing data
X.test <- test[,1:8]

# Running holdout method on logistic regression model
holdout <- glm(Exited ~ ., data = train, family = "binomial")
holdout_predictions <- as.data.frame(predict(holdout, newdata = X.test, type = "response"))
names(holdout_predictions) <- 'p'

# Classifying predictions into 0 or 1 using threshold of 0.5
holdout_predictions$Exited <- 0
for (i in 1:2000){
  if(holdout_predictions$p[i] >= 0.5){
    holdout_predictions$Exited[i] <- 1
  }
}

# Confusion matrix
confusionMatrix(data = as.factor(holdout_predictions$Exited), reference = as.factor(test$Exited))

# K- fold cross validation
X.train <- as.matrix(train[,1:8])
Y.train <- as.matrix(train[,9])
tenfold <- cv.glmnet(X.train,Y.train, family = "binomial", nfolds = 10)
tenfold_predictions <- as.data.frame(predict(tenfold, newx = as.matrix(X.test), type = "response"))
names(tenfold_predictions) <- 'p'

# Classifying predictions into 0 or 1 using threshold of 0.5
tenfold_predictions$Exited <- 0
for (i in 1:2000){
  if(tenfold_predictions$p[i] >= 0.5){
    tenfold_predictions$Exited[i] <- 1
  }
}

# Confusion matrix
confusionMatrix(data = as.factor(tenfold_predictions$Exited), reference = as.factor(test$Exited))

# Leave-one-out cross validation
loocv <- cv.glmnet(X.train,Y.train, family = "binomial", nfolds = 8000)
loocv_predictions <- as.data.frame(predict(loocv, newx = as.matrix(X.test), type = "response"))
names(loocv_predictions) <- 'p'

# Classifying predictions into 0 or 1 using threshold of 0.5
loocv_predictions$Exited <- 0
for (i in 1:2000){
  if(loocv_predictions$p[i] >= 0.5){
    loocv_predictions$Exited[i] <- 1
  }
}

# Confusion matrix
confusionMatrix(data = as.factor(loocv_predictions$Exited), reference = as.factor(test$Exited))

# Stratified cross validation
# Caret package includes function to create stratified folds
folds <- createFolds(as.factor(train$Exited), k = 10, list = FALSE)
stratified <- cv.glmnet(X.train ,Y.train, family = "binomial", foldid = folds)
stratified_predictions <- as.data.frame(predict(stratified, newx = as.matrix(X.test), type = "response"))
names(stratified_predictions) <- 'p'

# Classifying predictions into 0 or 1 using threshold of 0.5
stratified_predictions$Exited <- 0
for (i in 1:2000){
  if(stratified_predictions$p[i] >= 0.5){
    stratified_predictions$Exited[i] <- 1
  }
}

# Confusion matrix
confusionMatrix(data = as.factor(stratified_predictions$Exited), reference = as.factor(test$Exited))



# Bootstrapping

# Code for bootstrapping from modified from https://rpubs.com/vadimus/bootstrap
library(boot) 

# set seed for reproducibility
set.seed(300)

# Creating logistic regression model to predict customer churn
model <- glm(Exited ~ ., data = churn, family = binomial(link = "logit"))

# number of bootstrap samples
N = 10

# sample size which is equal to the number of rows in the dataframe 
n = nrow(churn) 

# Create a dataframe to hold the coefficients from each fitted model
# Find the number of coefficients the model has
k = length(coef(model))

# Dataframe needs N rows and k cols for each of the coefficients
B = data.frame(matrix(nrow = N, ncol = k))

# Loop through and take samples with replacment N times and fit the model to the bootstrapped sample
for(i in 1:N){

# Get the indicies of the samples
b = sample(x = 1:n, size = n, replace = TRUE)

# Use those indicies to select rows from the churn dataframe
boot_sample = churn[b,]

# Fit the model to the new boot_sample dataframe
boot_model = glm(model, data = boot_sample, family = binomial)

# Save the coefficients
B[i,]= coef(boot_model)
}

# Removing intercept term
B <- B[,-1]

colnames(B) <- c("CreditScore", "Age","Tenure","Balance","NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary")

# Plotting boxplots of regression coefficients
library(ggplot2)
data <- stack(as.data.frame(B))
ggplot(data) + 
  geom_boxplot(aes(x = ind, y = values)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Subsetting the data to visualize each coefficient
data1 <- subset(data, ind == "CreditScore")
credit_score <- ggplot(data1) + 
  geom_boxplot(aes(x = ind, y = values)) 

data2 <- subset(data, ind == "Age")
age <- ggplot(data2) + 
  geom_boxplot(aes(x = ind, y = values)) 

data3 <- subset(data, ind == "Tenure")
tenure <- ggplot(data3) + 
  geom_boxplot(aes(x = ind, y = values))

data4 <- subset(data, ind == "Balance")
balance <- ggplot(data4) + 
  geom_boxplot(aes(x = ind, y = values)) 

data5 <- subset(data, ind == "NumOfProducts")
num_of_products <- ggplot(data5) + 
  geom_boxplot(aes(x = ind, y = values)) 

data6 <- subset(data, ind == "HasCrCard")
has_cr_card <- ggplot(data6) + 
  geom_boxplot(aes(x = ind, y = values)) 

data7 <- subset(data, ind == "IsActiveMember")
is_active_member <- ggplot(data7) + 
  geom_boxplot(aes(x = ind, y = values)) 

data8 <- subset(data, ind == "EstimatedSalary")
estimated_salary <- ggplot(data8) + 
  geom_boxplot(aes(x = ind, y = values)) 

# Jackknife
coefs <- sapply(1:N, function(i)
  coef(glm(Exited ~ ., data=churn[-i, ], family='binomial'))
)

# Make output matrix into a dataframe
coefs <- as.data.frame(coefs)

# Transpose and remove the intercept column
coefs <- t(coefs)
coefs_dat <- coefs[,-1]

# Make boxplots of the coefficients
data <- stack(as.data.frame(coefs_dat))
ggplot(data) + 
  geom_boxplot(aes(x = ind, y = values)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Subsetting the data to visualize each coefficient
data1 <- subset(data, ind == "CreditScore")
credit_score <- ggplot(data1) + 
  geom_boxplot(aes(x = ind, y = values))

data2 <- subset(data, ind == "Age")
age <- ggplot(data2) + 
  geom_boxplot(aes(x = ind, y = values))

data3 <- subset(data, ind == "Tenure")
tenure <- ggplot(data3) + 
  geom_boxplot(aes(x = ind, y = values))

data4 <- subset(data, ind == "Balance")
balance <- ggplot(data4) + 
  geom_boxplot(aes(x = ind, y = values))

data5 <- subset(data, ind == "NumOfProducts")
num_of_products <- ggplot(data5) + 
  geom_boxplot(aes(x = ind, y = values))

data6 <- subset(data, ind == "HasCrCard")
has_cr_card <- ggplot(data6) + 
  geom_boxplot(aes(x = ind, y = values))

data7 <- subset(data, ind == "IsActiveMember")
is_active_member <- ggplot(data7) + 
  geom_boxplot(aes(x = ind, y = values))

data8 <- subset(data, ind == "EstimatedSalary")
estimated_salary <- ggplot(data8) + 
  geom_boxplot(aes(x = ind, y = values))

# Arranging the individual plots in one larger plot
library(gridExtra)
grid.arrange(credit_score, age, tenure, balance, num_of_products, has_cr_card, is_active_member, estimated_salary, nrow=4, ncol=2)

