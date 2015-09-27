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


```r
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

```r
train <- read.csv("pml-training.csv", na.strings=c("NA", "#DIV/0!", "", " "));
test  <- read.csv("pml-testing.csv", na.strings=c("NA", "#DIV/0!", "", " "));
```

1st 7 columns of Training data, Testing data and Last column of Training data are not relevant to our analysis

```r
train <- train[-c(1:7)];         # originally 160 variables
test  <- test[-c(1:7,160)];      # originally 160 variables
```

Originally each dataset contains 160 columns. Unnecessary variables are eliminated.

```r
dim(train);     # now 153 variables
```

```
## [1] 19622   153
```

```r
dim(test);      # now 152 variables
```

```
## [1]  20 152
```

Removing the columns which contain at least on NA value using a User Defined Function

```r
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

```r
dim(train);     # 53 variables
```

```
## [1] 19622    53
```

```r
dim(test);      # 53 variables
```

```
## [1] 20 52
```

## Subsetting Training data for Cross-Validation
The training data set contains 53 variables and 19622 obs.  
The testing data set contains 53 variables and 20 obs.  
In order to perform cross-validation, the training data set is partitioned into 2 sets: subTraining (75%) and subTest (25%).  
This will be performed using random subsampling without replacement.

```r
train.indexes <- createDataPartition(y=train$classe, p=0.75, list=FALSE);
train.trn <- train[train.indexes,];
train.tst <- train[-train.indexes,]; 
dim(train.trn);
```

```
## [1] 14718    53
```

```r
dim(train.tst);
```

```
## [1] 4904   53
```
We see there are **14716 in the train.trn** group, and **4906 in the train.tst** group.

## Charting the Predicted variable **classe**

```r
f <- table(train.trn$classe);
p <- plot(train.trn$classe, col=c(3,6,6,6,6,6), xlab="outcome variable \"classe\"", ylab="Frequency", yaxt="n");
text(p[,1], f-125, f);
legend("topright", c("Exercised correctly", "Exercised with mistakes"), pch=19, col=c(3,6));
```

![plot of chunk unnamed-chunk-8](figure/unnamed-chunk-8-1.png) 

## First prediction model: 
### Training Decision Tree Model

```r
model1 <- rpart(classe ~ ., data=train.trn, method="class")
```

### Prediction from Model

```r
prediction1 <- predict(model1, train.tst, type = "class")
length(prediction1)
```

```
## [1] 4904
```

### Decision Tree plotting

```r
rpart.plot(model1, main="Classification Tree", extra=102, under=TRUE, faclen=0);
```

![plot of chunk unnamed-chunk-11](figure/unnamed-chunk-11-1.png) 

### Model Evaluation

```r
cm1 <- confusionMatrix(prediction1, train.tst$classe)
print(cm1);
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1235  157   16   50   20
##          B   55  568   73   80  102
##          C   44  125  690  118  116
##          D   41   64   50  508   38
##          E   20   35   26   48  625
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7394          
##                  95% CI : (0.7269, 0.7516)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6697          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8853   0.5985   0.8070   0.6318   0.6937
## Specificity            0.9307   0.9216   0.9005   0.9529   0.9678
## Pos Pred Value         0.8356   0.6469   0.6313   0.7247   0.8289
## Neg Pred Value         0.9533   0.9054   0.9567   0.9296   0.9335
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2518   0.1158   0.1407   0.1036   0.1274
## Detection Prevalence   0.3014   0.1790   0.2229   0.1429   0.1538
## Balanced Accuracy      0.9080   0.7601   0.8537   0.7924   0.8307
```

## Second prediction model:
### Training Random Forest Model

```r
model2 <- randomForest(classe ~. , data=train.trn, method="class")
```

### Prediction from Model

```r
prediction2 <- predict(model2, train.tst, type = "class")
```

### Model Evaluation

```r
cm2 <- confusionMatrix(prediction2, train.tst$classe)
print(cm2);
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1394    3    0    0    0
##          B    1  944   10    0    0
##          C    0    2  843    6    0
##          D    0    0    2  798    0
##          E    0    0    0    0  901
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9951          
##                  95% CI : (0.9927, 0.9969)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9938          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9993   0.9947   0.9860   0.9925   1.0000
## Specificity            0.9991   0.9972   0.9980   0.9995   1.0000
## Pos Pred Value         0.9979   0.9885   0.9906   0.9975   1.0000
## Neg Pred Value         0.9997   0.9987   0.9970   0.9985   1.0000
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2843   0.1925   0.1719   0.1627   0.1837
## Detection Prevalence   0.2849   0.1947   0.1735   0.1631   0.1837
## Balanced Accuracy      0.9992   0.9960   0.9920   0.9960   1.0000
```

## Conclusion from TWO Models Evaluation
### Random Forests yielded better Results, as expected!
The Accuracy of this Decision Tree Model is 0.7373  
whereas, the Accuracy of Random Forest Model is 0.999

```r
cm1$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   7.393964e-01   6.697367e-01   7.268685e-01   7.516387e-01   2.844617e-01 
## AccuracyPValue  McnemarPValue 
##   0.000000e+00   6.938867e-37
```

```r
cm2$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.9951060      0.9938092      0.9927269      0.9968619      0.2844617 
## AccuracyPValue  McnemarPValue 
##      0.0000000            NaN
```

## Using Best Model to Predict from original test data
### predict outcome levels on the original Test data set using Random Forest Model

```r
predict99 <- predict(model2, test, type="class")
predict99
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

## Submitting the Predicted results to Coursera
### Generating files for submission as part of Assignment

```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}


pml_write_files(predict99);
```
