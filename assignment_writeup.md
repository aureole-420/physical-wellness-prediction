## Practical Machine Learning: Prediction Assignment Writeup

The wide application of smart devices such as Jawbone Up, Nike FuelBand and Fitbit makes it possible to 
a large amount of data about personal activity relatively inexpensively. In this project, people were asked to perform barbell lifts correctly and incorrectly in 5 different ways. The goal will 
be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants, to quantify how well they do it. 

* Data used for the assignment is kindly provided by http://groupware.les.inf.puc-rio.br/har
* data for training https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
* data for testing https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

### Data Cleaning
* Load caret and ggplot2 package for training and ploting. 
```R
library(caret)
library(ggplot2)
```
* Data loading
```R
## ============ Read training and testing data ==========================
training_raw <- read.csv("pml-training.csv")
testing_raw <- read.csv("pml-testing.csv")
```
* There are too many predictors, so removing columns with space, NA. Also remove the first 4 irrelevant columns.
```
training_data <- training_raw[,colSums(is.na(training_raw)) == 0]
training_data <- training_data[,sapply(training_data,is.numeric)] 
training_data <- training_data[,-c(1:4)]
```
*  First convert all elements to numerical value then conduct the correlation analysis. Highly correlated columns should be dumped keep less predictors for the prediction in the next part. 
```R
# Dump highly correlated variables
CorMat <- cor(training_data)
remove <- findCorrelation(CorMat, cutoff = 0.9)
training_data <- training_data[,-remove]
training_data[["classe"]] <- training_raw$classe
```

### Data Partition and prediction trials
* Partition the data into training set (70%) and testing set (30%)
```R
# partition for training and testing set
set.seed(201604)
inTrain <- createDataPartition(y = training_data$classe, p =  0.6, list = FALSE)
training_set <- training_data[inTrain,] 
testing_set <- training_data[-inTrain,]
```
* Training: multiple methods in caret packages are used including: rpart(decision tree), gbm(boosting with trees), treebag (bagging), rf(random forest)
```R
print("=============decision tree=========================")
mod_rpart <- train(classe ~., method = 'rpart',data = training_set)
pre <- predict(mod_rpart, newdata = testing_set)
Imp <- varImp(mod_rpart, scale = FALSE)
ggplot(Imp, top = 20)
performance <- confusionMatrix(pre,testing_set$classe)
print(performance)
print(performance$overall[1])
print("=============boosting with trees ===================")
mod_gbm <- train(classe ~., method = 'gbm',data = training_set)
pre <- predict(mod_gbm, newdata = testing_set)
Imp <- varImp(mod_gbm, scale = FALSE)
ggplot(Imp,top = 20)
performance <- confusionMatrix(pre,testing_set$classe)
print(performance)
print(performance$overall[1])
print("=============bagging ==============================")
mod_treebag <- train(classe ~., method = 'treebag',data = training_set)
pre <- predict(mod_treebag, newdata = testing_set)
Imp <- varImp(mod_treebag, scale = FALSE)
ggplot(Imp,top = 20)
performance <- confusionMatrix(pre,testing_set$classe)
print(performance)
print(performance$overall[1])
print("=============random forest==========================")
fit_control <- trainControl(method = "cv",number = 3, allowParallel =T, verbose = T)
mod_rf <- train(classe ~ ., method = 'rf', data = training_set, trControl = fit_control, verbose = T)
pre <- predict(mod_rf, newdata = testing_set)
Imp <- varImp(mod_rf, scale = FALSE)
ggplot(Imp,top = 20)
performance <- confusionMatrix(pre,testing_set$classe)
print(performance)
print(performance$overall[1])
```
