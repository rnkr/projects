# Titanic Kaggle problem

# 1. Prepare Problem
# a) Load libraries
library(caret)
library(neuralnet)

# b) Load dataset
filename <- 'train.csv'
datasetorig <- read.csv(filename, header=TRUE, stringsAsFactors = FALSE)
filename <- 'test.csv'
datasettest <- read.csv(filename, header=TRUE, stringsAsFactors = FALSE)

str(datasetorig)

table(datasetorig$Pclass)

datasetorig$Survived <- factor(datasetorig$Survived)
datasetorig$Embarked <- factor(datasetorig$Embarked)
datasetorig$Pclass <- factor(datasetorig$Pclass)
datasetorig$Sex <- factor(datasetorig$Sex ) 


datasettest$Survived <- factor(datasettest$Survived)
datasettest$Embarked <- factor(datasettest$Embarked)
datasettest$Pclass <- factor(datasettest$Pclass)
datasettest$Sex <- factor(datasettest$Sex ) 



## add feature
## add column
datasetorig$famly <- datasetorig$SibSp + datasetorig$Parch
datasettest$famly <- datasettest$SibSp + datasettest$Parch
## scale 
 ###datasetorig$Fare <- scale(datasetorig$Fare)

complete_cases <- complete.cases(datasetorig)

# Imputation ---- Address NA values in Age
  datasetorig$Age[is.na(datasetorig$Age)] <- mean(datasetorig$Age, na.rm = TRUE )
  datasettest$Age[is.na(datasettest$Age)] <- mean(datasettest$Age, na.rm = TRUE ) 
  datasettest$Fare[is.na(datasettest$Fare)] <- mean(datasettest$Fare, na.rm = TRUE ) 
  
  
dataset <- datasetorig

# c) Split-out validation dataset
cutoffindex <- createDataPartition(dataset$Survived, p = 0.8, list = FALSE)
validate_model_dataset <- dataset[-cutoffindex, ]
dataset <- dataset[cutoffindex, ]


# 2. Summarize Data
# a) Descriptive statistics
table(datasetorig$Survived, datasetorig$Sex)

dim(dataset)
sapply(dataset, class)
head(dataset)
levels(dataset$Pclass)
percentage <- prop.table(table(dataset$Survived)) * 100
cbind(freq = table(dataset$Survived), percentage)
summary(dataset)


# b) Data visualizations
hist(dataset$Age)
barplot(table(datasetorig$Sex, datasetorig$Survived ), legend.text=unique(datasetorig$Sex))

# 3. Prepare Data
# a) Data Cleaning
# b) Feature Selection
   #remove columns - Passenger iD 1, Name 4, ticket 9 , cabn 11
dataset <- dataset[, -c(1,4,9,11)]


# c) Data Transforms

# 4. Evaluate Algorithms
# a) Test options and evaluation metric
set.seed(17)
trainControl <- trainControl(method = "repeatedcv", number =10, repeats = 6 )
metric <- "Accuracy"

# b) Spot Check Algorithms

#LDA Linear Discriminant Analysis (LDA)
set.seed(17)
fit.lda <- train( Survived ~., data = dataset, method = "lda", metric = metric, trControl = trainControl)

#SVM Support Vector Machines
set.seed(17)
fit.svm <- train(Survived ~., data = dataset, method = "svmRadial", metric = metric, trControl = trainControl)

#Random Forest
set.seed(17)
fit.rf <- train(Survived ~., data = dataset, method = "rf", metric = metric,trControl = trainControl )

#Neural Networks
set.seed(17)
fit.nn <- train(Survived~., data=dataset, method = 'nnet', metric = metric, hidden = 1, trControl = trainControl )

#Model Tree
#set.seed(17)
#fit.mt <- train(Survived~., data=dataset, method='M5', metric = metric, trControl = trainControl, na.action = na.exclude )



# c) Compare Algorithms
results <- resamples(list(lda=fit.lda, svm = fit.svm, rf=fit.rf, nn= fit.nn ))
summary(results)

dotplot(results)
print(results)

# 5. Improve Accuracy
# a) Algorithm Tuning
 ## set.seed(17)
 ## finalModel <- randomForest(Survived~., dataset, mtry=2, ntree=2000, na.action = na.exclude)

# b) Ensembles

# 6. Finalize Model

# a) Predictions on validation dataset

predictions <- predict( fit.rf, validate_model_dataset )
confusionMatrix(predictions, validate_model_dataset$Survived)


#confusionMatrix(predictions, datasettest$Survived)

# run prdictions on full dataset
predictionfull <- predict (fit.rf, datasetorig)
confusionMatrix(predictionfull, datasetorig$Survived)


# test dataset
predictionsontest <- predict( fit.rf, datasettest )
submit <- data.frame(PassengerId = datasettest$PassengerId, Survived = predictionsontest)
write.csv(submit, file = "TitanicP1.csv", row.names = FALSE)







