# Quantitatively predicting physical wellness using Exercise-activity data collecting by smart devices.

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
##### Performance of each methods
* **decision tree**: the accuracy is 0.5173 which is too low.

```
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 2014  580  637  513  301
         B   51  517   54   28  243
         C  132  328  538  126  309
         D   34   92  139  519  118
         E    1    1    0  100  471

Overall Statistics
                                          
               Accuracy : 0.5173          
                 95% CI : (0.5062, 0.5284)
    No Information Rate : 0.2845          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.3709          
 Mcnemar's Test P-Value : < 2.2e-16       

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.9023  0.34058  0.39327  0.40358  0.32663
Specificity            0.6382  0.94058  0.86184  0.94162  0.98407
Pos Pred Value         0.4979  0.57895  0.37544  0.57539  0.82199
Neg Pred Value         0.9426  0.85603  0.87058  0.88954  0.86649
Prevalence             0.2845  0.19347  0.17436  0.16391  0.18379
Detection Rate         0.2567  0.06589  0.06857  0.06615  0.06003
Detection Prevalence   0.5155  0.11382  0.18264  0.11496  0.07303
Balanced Accuracy      0.7703  0.64058  0.62756  0.67260  0.65535
```
* **boosting with trees**:
```
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 2196   51    0    1    2
         B   21 1408   52    8   21
         C    9   53 1294   45   10
         D    3    2   21 1216   18
         E    3    4    1   16 1391

Overall Statistics
                                          
               Accuracy : 0.9565          
                 95% CI : (0.9518, 0.9609)
    No Information Rate : 0.2845          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.945           
 Mcnemar's Test P-Value : 4.692e-08       

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.9839   0.9275   0.9459   0.9456   0.9646
Specificity            0.9904   0.9839   0.9819   0.9933   0.9963
Pos Pred Value         0.9760   0.9325   0.9171   0.9651   0.9830
Neg Pred Value         0.9936   0.9826   0.9885   0.9894   0.9921
Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
Detection Rate         0.2799   0.1795   0.1649   0.1550   0.1773
Detection Prevalence   0.2868   0.1925   0.1798   0.1606   0.1803
Balanced Accuracy      0.9871   0.9557   0.9639   0.9694   0.9804
```
* **bagging** 
```
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 2219   18    0    0    0
         B    8 1468   13    0    1
         C    3   23 1345   20    5
         D    1    7   10 1263    2
         E    1    2    0    3 1434

Overall Statistics
                                          
               Accuracy : 0.9851          
                 95% CI : (0.9822, 0.9877)
    No Information Rate : 0.2845          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.9811          
 Mcnemar's Test P-Value : 0.002177        

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.9942   0.9671   0.9832   0.9821   0.9945
Specificity            0.9968   0.9965   0.9921   0.9970   0.9991
Pos Pred Value         0.9920   0.9852   0.9635   0.9844   0.9958
Neg Pred Value         0.9977   0.9921   0.9964   0.9965   0.9988
Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
Detection Rate         0.2828   0.1871   0.1714   0.1610   0.1828
Detection Prevalence   0.2851   0.1899   0.1779   0.1635   0.1835
Balanced Accuracy      0.9955   0.9818   0.9877   0.9895   0.9968
```



* **random forest**
```
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 2229    1    0    0    0
         B    2 1515    4    0    0
         C    0    2 1360    5    2
         D    0    0    4 1281    1
         E    1    0    0    0 1439

Overall Statistics
                                          
               Accuracy : 0.9972          
                 95% CI : (0.9958, 0.9982)
    No Information Rate : 0.2845          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.9965          
 Mcnemar's Test P-Value : NA              

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.9987   0.9980   0.9942   0.9961   0.9979
Specificity            0.9998   0.9991   0.9986   0.9992   0.9998
Pos Pred Value         0.9996   0.9961   0.9934   0.9961   0.9993
Neg Pred Value         0.9995   0.9995   0.9988   0.9992   0.9995
Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
Detection Rate         0.2841   0.1931   0.1733   0.1633   0.1834
Detection Prevalence   0.2842   0.1939   0.1745   0.1639   0.1835
Balanced Accuracy      0.9992   0.9985   0.9964   0.9977   0.9989
```
#### Choosing prediction model
The accuracy of prediction made by random forest model (0.9972) is the highest of all models.
Below is the importance of variable of the random forest
![](https://github.com/aureole-420/practical_machine_learning_assignment/blob/master/rf.png)


## Out of sample accuracy
The analysis above shows the random forest with the best performance in prediction, so it will be used for prediction for testing set.
```
oospre <- predict(mod_rf, newdata = testing_raw)
print("random forest")
print(oospre)
```
The results is displayed below which one can check in the following quiz to be all correct.
```
[1] "random forest"
> print(oospre)
 [1] B A B A A E D B A A B C B A E E A B B B
Levels: A B C D E
```

