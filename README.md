---
title: "Coursera Project - Machine Learning"
author: "Nagababu"
date: "September 27, 2015"
output: html_document
---

## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behaviour, or because they are tech geeks.

One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, our goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

More information is available from the website [here:](http://groupware.les.inf.puc-rio.br/har) (see the section on the Weight Lifting Exercise Dataset). 


## Weight Lifting Exercise
![body](images/body.png)  

Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).

Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes. Participants were supervised by an experienced weight lifter to make sure the execution complied to the manner they were supposed to simulate. The exercises were performed by six male participants aged between 20-28 years, with little weight lifting experience. We made sure that all participants could easily simulate the mistakes in a safe and controlled manner by using a relatively light dumbbell (1.25kg).

## Data

The training data for this project are available here: 
[Training Data](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)

The goal of our project is to predict the manner in which they did the exercise. There is a "classe" variable in the training set. It contains Factor data with Levels A,B,C,D,E that indicates how the participants performed the exercise.

The test data are available here: 
[Testing Data](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)

## Goal
Two models will be tested using decision tree and random forest algorithms. The model with the highest accuracy will be chosen as our final model.

## Cross-validation

We have a large sample size with N= 19622 in the Training data set. This allow us to divide our Training sample into subTraining and subTesting to allow cross-validation. Cross-validation will be performed by subsampling our training data set randomly without replacement into 2 subsamples: 

1. train.trn data (75% of the original Training data set) and  
2. train.tst data (25%). 

Our models will be fitted on the train.trn data set, and tested on the train.tst data. Once the most accurate model is choosen, it will be tested on the original Testing data set.

```{r}
#library(dplyr);
library(caret);
library(rpart);
library(randomForest);
library(rpart.plot);
library(e1071);

set.seed(1234);
path <- "D:\\CoursEra2014\\Data Science\\Course08 Machine Learning\\Project Submission";
setwd(path);
```

## Data Gathering and Cleaning
Loading data into R and converting invalid/missing data values into NA values
```{r}
train <- read.csv("pml-training.csv", na.strings=c("NA", "#DIV/0!", "", " "));
test  <- read.csv("pml-testing.csv", na.strings=c("NA", "#DIV/0!", "", " "));
```

1st 7 columns of Training data, Testing data and Last column of Training data are not relevant to our analysis
```{r}
train <- train[-c(1:7)];         # originally 160 variables
test  <- test[-c(1:7,160)];      # originally 160 variables
```

Originally each dataset contains 160 columns. Unnecessary variables are eliminated.
```{r}
dim(train);     # now 153 variables
dim(test);      # now 152 variables
```

Removing the columns which contain at least on NA value using a User Defined Function
```{r}
# function to verify existence of atleast one NA value in a given column
is.na.column = function(col){
   any(is.na(col))
}

cols <- names(train);
non.empty.columns <- which(!sapply(train,is.na.column));
train <- train[,non.empty.columns]

cols <- names(test);
non.empty.columns <- which(!sapply(test,is.na.column));
test <- test[,non.empty.columns]
```

Finally, Training and Testing Data sets contains 53 columns each
```{r}
dim(train);     # 53 variables
dim(test);      # 53 variables
```

## Subsetting Training data for Cross-Validation
The training data set contains 53 variables and 19622 obs.  
The testing data set contains 53 variables and 20 obs.  
In order to perform cross-validation, the training data set is partitioned into 2 sets: subTraining (75%) and subTest (25%).  
This will be performed using random subsampling without replacement.
```{r}
train.indexes <- createDataPartition(y=train$classe, p=0.75, list=FALSE);
train.trn <- train[train.indexes,];
train.tst <- train[-train.indexes,]; 
dim(train.trn);
dim(train.tst);
```
We see there are **14716 in the train.trn** group, and **4906 in the train.tst** group.

## Charting the Predicted variable **classe**
```{r}
f <- table(train.trn$classe);
p <- plot(train.trn$classe, col=c(3,6,6,6,6,6), xlab="outcome variable \"classe\"", ylab="Frequency", yaxt="n");
text(p[,1], f-125, f);
legend("topright", c("Exercised correctly", "Exercised with mistakes"), pch=19, col=c(3,6));
```

## First prediction model: 
### Training Decision Tree Model
```{r}
model1 <- rpart(classe ~ ., data=train.trn, method="class")
```

### Prediction from Model
```{r}
prediction1 <- predict(model1, train.tst, type = "class")
length(prediction1)
```

### Decision Tree plotting
```{r}
rpart.plot(model1, main="Classification Tree", extra=102, under=TRUE, faclen=0);
```

### Model Evaluation
```{r}
cm1 <- confusionMatrix(prediction1, train.tst$classe)
print(cm1);
```

## Second prediction model:
### Training Random Forest Model
```{r}
model2 <- randomForest(classe ~. , data=train.trn, method="class")
```

### Prediction from Model
```{r}
prediction2 <- predict(model2, train.tst, type = "class")
```

### Model Evaluation
```{r}
cm2 <- confusionMatrix(prediction2, train.tst$classe)
print(cm2);
```

## Conclusion from TWO Models Evaluation
### Random Forests yielded better Results, as expected!
The Accuracy of this Decision Tree Model is 0.7373  
whereas, the Accuracy of Random Forest Model is 0.999
```{r}
cm1$overall
cm2$overall
```

## Using Best Model to Predict from original test data
### predict outcome levels on the original Test data set using Random Forest Model
```{r}
predict99 <- predict(model2, test, type="class")
predict99
```

## Submitting the Predicted results to Coursera
### Generating files for submission as part of Assignment
```{r}
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}


pml_write_files(predict99);
```
