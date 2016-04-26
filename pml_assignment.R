rm(list = ls())
library(caret)
library(ggplot2)
## Set working directory
#setwd("~/OneDirve/Mooc")

## ============ Read training and testing data ==========================
training_raw <- read.csv("pml-training.csv")
testing_raw <- read.csv("pml-testing.csv")

## ================Data cleaning=========================================

# Filter space, NA, convert all to numerical value

training_data <- training_raw[,colSums(is.na(training_raw)) == 0]
training_data <- training_data[,sapply(training_data,is.numeric)] 
training_data <- training_data[,-c(1:4)]

# Dump highly correlated variables
CorMat <- cor(training_data)
remove <- findCorrelation(CorMat, cutoff = 0.9)
training_data <- training_data[,-remove]
training_data[["classe"]] <- training_raw$classe

## ======================== Training ====================================
# partition for training and testing set
set.seed(201604)
inTrain <- createDataPartition(y = training_data$classe, p =  0.6, list = FALSE)
training_set <- training_data[inTrain,] 
testing_set <- training_data[-inTrain,]


# ===========choose different method to feed into the models============

cat("=================================\n")
cat("=================================\n")
cat("Start Modeling\n")
cat("=================================\n")
cat("=================================\n")

if (TRUE){
sink('rpart_output.txt')

# ------------ decision tree -------------------------------------------
cat("=================================\n")
cat("decision tree\n")
cat("=================================\n")

sink("rpart_output.txt")
mod_rpart <- train(classe ~., method = 'rpart',data = training_set)
save(mod_rpart, file = "mod_rpart.RData")

pre <- predict(mod_rpart, newdata = testing_set)

cat('---------------------------------\n')
cat('Variable importance\n')
cat('---------------------------------\n')
Imp <- varImp(mod_rpart, scale = FALSE)
print(Imp)
ggplot(Imp, top = 20)
ggsave("rpart.png")
cat('---------------------------------\n')
cat('Confusion matrix\n')
cat('---------------------------------\n')
performance <- confusionMatrix(pre,testing_set$classe)
print(performance)

cat('---------------------------------\n')
cat('Accuracy\n')
cat('---------------------------------\n')
print(performance$overall[1])

sink()
}

if (FALSE) {
sink('gbm_output.txt')
# ------------boosting with trees ---------------------------------------
cat("=================================\n")
cat("boosting with trees (gbm)\n")
cat("=================================\n")

mod_gbm <- train(classe ~., method = 'gbm',data = training_set)
save(mod_gbm, file = "mod_gbm.RData")

pre <- predict(mod_gbm, newdata = testing_set)

cat('---------------------------------\n')
cat('Variable importance\n')
cat('---------------------------------\n')
Imp <- varImp(mod_gbm, scale = FALSE)
print(Imp)
ggplot(Imp, top = 20)
ggsave("gbm.png")

cat('---------------------------------\n')
cat('Confusion matrix\n')
cat('---------------------------------\n')
performance <- confusionMatrix(pre,testing_set$classe)
print(performance)

cat('---------------------------------\n')
cat('Accuracy\n')
cat('---------------------------------\n')
print(performance$overall[1])
sink()
}


if (TRUE){
sink('treebag_output.txt')
# ------------bagging (treebag) ---------------------------------------
cat("=================================\n")
cat("bagging (treebag)\n")
cat("=================================\n")

mod_treebag <- train(classe ~., method = 'treebag',data = training_set)
save(mod_treebag, file = "mod_treebag.RData")

pre <- predict(mod_treebag, newdata = testing_set)

cat('---------------------------------\n')
cat('Variable importance\n')
cat('---------------------------------\n')
Imp <- varImp(mod_treebag, scale = FALSE)
print(Imp)
ggplot(Imp,top = 20)
ggsave("treebag.png")
cat('---------------------------------\n')
cat('Confusion matrix\n')
cat('---------------------------------\n')
performance <- confusionMatrix(pre,testing_set$classe)
print(performance)

cat('---------------------------------\n')
cat('Accuracy\n')
cat('---------------------------------\n')
print(performance$overall[1])

sink()
}



if (TRUE){
sink("rf_output.txt")

# ------------ random forest -------------------------------------------
cat("=================================\n")
cat("random forest\n")
cat("=================================\n")
fit_control <- trainControl(method = "cv",number = 3, allowParallel =T, verbose = T)
mod_rf <- train(classe ~ ., method = 'rf', data = training_set, trControl = fit_control, verbose = T)
save(mod_rf,file = "mod_rf.RData")
pre <- predict(mod_rf, newdata = testing_set)

cat('---------------------------------\n')
cat('Variable importance\n')
cat('---------------------------------\n')
Imp <- varImp(mod_rf, scale = FALSE)
print(Imp)
ggplot(Imp,top = 20)
ggsave("rf.png")

cat('---------------------------------\n')
cat('Confusion matrix\n')
cat('---------------------------------\n')
performance <- confusionMatrix(pre,testing_set$classe)
print(performance)

cat('---------------------------------\n')
cat('Accuracy\n')
cat('---------------------------------\n')
print(performance$overall[1])


cat("=================================\n")
cat("=================================\n")
cat("Out of sample prediction\n")
cat("=================================\n")
cat("=================================\n")

sink()
}



oospre <- predict(mod_rpart, newdata = testing_raw)
print("decision tree")
print(oospre)

oospre <- predict(mod_gbm, newdata = testing_raw)
print("gbm")
print(oospre)

oospre <- predict(mod_treebag, newdata = testing_raw)
print("treebag")
print(oospre)

oospre <- predict(mod_rf, newdata = testing_raw)
print("random forest")
print(oospre)


