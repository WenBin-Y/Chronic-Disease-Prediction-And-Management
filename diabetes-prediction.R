# ==============================================================================
# University Coursework
# Purpose:      Predicting Diabetes With Machine Learning Models and 
#               Identifying Key Factors
# Disease:      Diabetes
# Authors:      Yeung WenBin
# DOC:          23-03-2024
# Topics:       Data Wrangling, Logistic Regression, CART, Random Forest
# Data Source:  diabetes_data.csv
# Link:         https://www.kaggle.com/datasets/prosperchuks/health-dataset
# Packages:     tidyverse, ggplot2, haven, labelled, caTools, caret, ROCR,
#               pROC, randomForest, corrplot, rpart, rpart.plot

# ==============================================================================
# ============================= TABLE OF CONTENTS ==============================

# Part 1: Data Preparation & Cleaning

# Part 2: Data Exploration & Selecting Variables
# ------- 2.1 Data Visualisation
# ------- 2.2 Kruskal Test (For Continuous Variables)
# ------- 2.3 Chi-Squared test (For Categorical variables)
# ------- 2.4 Logistic Regression (w/ backward elimination)
# ------- 2.5 Selecting Variables

# Part 3: Model Evaluation
# ------- 3.1 Train-Test Split & Set Threshold
# ------- 3.2 Model 1: Logistic Regression w Backward Elimination + Evaluation
# ------- 3.3 Model 2: CART + Evaluation
# ------- 3.4 Model 3: Random Forest + Evaluation

# ==============================================================================

# Set your working directory here :)
setwd('/Users/ywb/Desktop/Univeristy/AY 23:24/BC2407-Analytics 2/Project Work/Submission')

# Automation to install packages that might have yet to be installed
package.list <- c("tidyverse", "ggplot2", "haven", "labelled", "caTools", 
                  "caret", "ROCR", "pROC", "rpart", "rpart.plot", "randomForest")
new.packages <- package.list[!(package.list %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)


library(tidyverse)
library(ggplot2)
library(haven)
library(labelled)
library(caTools)
library(caret)
library(ROCR)
library(pROC)
library(randomForest)
library(corrplot)
library(rpart)
library(rpart.plot)




# ================== Part 1: Data Preparation & Cleaning =======================
# Load dataset, import from csv
diabetes = read.csv("diabetes_data.csv")
View(diabetes)
summary(diabetes)

## Numeric (Continuous) variables: 
## - BMI, 
## - MentHlth (days of poor mental health scale 1-30 days),
## - PhysHlth (physical illness or injury days in past 30 days scale 1-30),

## The rest of the variables are categorical. (0 or 1) and should be
## transformed to factor


# Transform factor variables to values with greater interpretability
# Referencing Kaggle data source to understand values and meaning
diabetes$Age <- factor(diabetes$Age, 
                          levels = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13),
                          labels = c("18-24", "25-29", "30-34", "35-39", "40-44",
                                     "45-49", "50-54", "55-59", "60-64", "65-69",
                                     "70-74", "75-79", "80 or older"))
## Age categories derived from original data source:
## Behavioral Risk Factor Surveillance System (BRFSS) 2015
## Link to codebook: https://www.cdc.gov/brfss/annual_data/2015/pdf/overview_2015.pdf 


diabetes$Sex <- factor(diabetes$Sex, levels = c(0,1), 
                          labels = c("female", "male"))

diabetes$HighChol <- factor(diabetes$HighChol, levels = c(0,1), 
                               labels = c("No HighChol", "HighChol"))

diabetes$CholCheck <- factor(diabetes$CholCheck, levels = c(0,1),
                                labels = c("No Check P5Y", "Checked P5Y")) 
## P5Y - Past 5 Years

diabetes$Smoker <- factor(diabetes$Smoker, levels = c(0,1),
                                labels = c("Smoke <100 Cigs", "Smoke >= 100 Cigs"))

diabetes$HeartDiseaseorAttack <- factor(diabetes$HeartDiseaseorAttack, 
                                           levels = c(0,1),
                                           labels = c("No CHD or MI", "CHD or MI"))
## CHD - Coronary Heart Disease / MI - Myocardial Infarction

diabetes$PhysActivity <- factor(diabetes$PhysActivity, levels = c(0,1),
                                labels = c("No PhysAct P30D", "PhysAct P30D"))
## P30D - Past 30 Days

diabetes$Fruits <- factor(diabetes$Fruits, levels = c(0,1),
                                labels = c("< 1 per day", ">= 1 per day"))

diabetes$Veggies <- factor(diabetes$Veggies, levels = c(0,1),
                             labels = c("< 1 per day", ">= 1 per day"))

diabetes$HvyAlcoholConsump <- factor(diabetes$HvyAlcoholConsump, 
                                        levels = c(0,1),
                                        labels = c("No", "Yes"))
## Adult men >=14 drinks per week / adult women>=7 drinks per week)

diabetes$DiffWalk <- factor(diabetes$DiffWalk, levels = c(0,1),
                              labels = c("No", "Yes"))

diabetes$Stroke <- factor(diabetes$Stroke, levels = c(0,1),
                               labels = c("No", "Yes"))

diabetes$HighBP <- factor(diabetes$HighBP, levels = c(0,1),
                               labels = c("No", "Yes"))

diabetes$Diabetes <- factor(diabetes$Diabetes, levels = c(0,1),
                               labels = c("No", "Yes"))


# Check for NA values
colSums(is.na(diabetes))
# No NA values -- All datasets are cleaned, augmented, and have balanced classes.


# Detect outliers using boxplots
boxplot(diabetes, main = "Boxplot of all variables in diabetes dataset")
# BMI, MentHlth and PhysHlth have a lot of outliers (beyond upper limit)


# Target variable: Diabetes (0 = no diabetes, 1 = diabetes)





# ============== Part 2: Data Exploration & Selecting Variables ================
# Creation of visualisations will be done to analyse the impact that certain
# variables have on diabetes outcome.

# Not all variables will be used in the dataset to train and test models.
# The team will leverage various methods to evaluate the significance of each
# variable.

# Method 1: Using domain knowledge
#           Most of the variables are important predictors of diabetes. 
#           However, we can take note of some variables that indicate about one's
#           lifestyle factors:
          
#           Fruits:   While not a direct predictor, a healthy diet rich in fruits 
#                     and vegetables can help with weight management and reduce 
#                     the risk of diabetes.

#           Veggies:  Same reason as Fruits

#           MentHlth: Mental health could be indirectly related through factors 
#                     like stress or medication use that might influence the 
#                     risk of diabetes

#           PhysHlth: Physical health can affect other factors like one's mental
#                     health, BMI and cholesterol levels

#           DiffWalk: Difficulty with walking could be a consequence of diabetes 
#                     or a risk factor depending on the cause.

#           Stroke:   While stroke itself might not directly cause diabetes, it 
#                     could be a risk factor if it is caused by uncontrolled 
#                     blood sugar.

# Method 2: Kruskal-Wallis test
#           Determines if there is statistically significant difference in
#           medians of >= 3 independent groups. Can be applied to applied to
#           categorical Y and continuous X.

# Method 3: Chi-squared test
#           Measures the association between categorical Y and X
#           If p-value > threshold, there is strong association between
#           variables.

# Method 4: Logistic Regression w Backward Elimination
#           Using logistic regression to find variables that are important 
#           indicators in predicting diabetes 





# ==============================================================================
# --------------------------- 2.1 Data Visualisation ---------------------------
# Set main colours to be used across plots
colour1 = '#F8766D' 
colour2 = '#00BFC4'


# Chart 1
# How people with Diabetes perceived their general health on a scale 1-5? 
# OR 
# How people that perceived their general health to be excellent(1) or very good(2)
# can still have Diabetes.
# 1 = excellent 2 = very good 3 = good 4 = fair 5 = poor

ggplot(data = diabetes, aes(fill=factor(Diabetes), y=..count.., 
                            x=factor(GenHlth))) + 
  geom_bar(position="stack", stat="count") +   
  stat_count(aes(y=after_stat(count), label=after_stat(count)), size = 3, 
             geom="text", position = position_stack(vjust=0.5)) + 
  labs(x = "General Health Level", y = "Number Of Diabetes", 
       title = "Diabetes Distribution based on General Health Level") +
  scale_fill_manual(values = c(colour2, colour1)) +
  theme(plot.title = element_text(hjust = 0.5)) 


# Chart 2
# Distribution of Diabetes based on BMI
ggplot(data = subset(diabetes, BMI < 50), aes(fill=factor(Diabetes), y=..count.., 
                                              x=BMI)) + 
  geom_bar(position="stack", stat="count") +   
  stat_count(aes(y=after_stat(count), label=after_stat(count)), size = 3, 
             geom="text", position = position_stack(vjust=0.5)) + 
  labs(x = "BMI", y = "Number Of Diabetes", 
       title = "Distribution of Diabetes based on BMI (BMI < 50)") +
  scale_fill_manual(values = c(colour2, colour1)) +
  theme(plot.title = element_text(hjust = 0.5)) 


# Chart 3
# Distribution of Diabetes based on High Cholesterol
ggplot(data = diabetes, aes(fill=factor(Diabetes), y=..count.., 
                            x=factor(HighChol))) + 
  geom_bar(position="stack", stat="count") +   
  stat_count(aes(y=after_stat(count), label=after_stat(count)), size = 3, 
             geom="text", position = position_stack(vjust=0.5)) + 
  labs(x = "High Cholesterol", y = "Number Of Diabetes", 
       title = "Distribution of Diabetes based on High Cholesterol") +
  scale_fill_manual(values = c(colour2, colour1)) +
  theme(plot.title = element_text(hjust = 0.5)) 





# ------------------------------ 2.2 Kruskal Test ------------------------------
# For categorical Y and continuous X
# Create dataframe to store the p-values of all continuous X variables
results.kruskal.df <- data.frame(variable = character(), 
                                 p_value = numeric(), 
                                 stringsAsFactors = FALSE)

# Create dataframe with only continuous X variables, and categorical Y
numeric.columns <- sapply(diabetes, is.numeric) 
numeric.data <- diabetes[, numeric.columns]
numeric.data["Diabates_outcome"] <- diabetes$Diabetes


# Run kruskal test for variables
for (col in colnames(numeric.data)) {
  
  # Extract the column data
  column.data <- numeric.data[[col]]
  
  # Perform Kruskal-Wallis test
  kruskal_result <- kruskal.test(Diabates_outcome ~ column.data, 
                                 data = numeric.data)
  
  # Store the results
  results.kruskal.df <- rbind(results.kruskal.df, 
                              data.frame(variable = col, 
                                         p_value = round(kruskal_result$p.value, 3)
                              ))
}


# Remove Diabates_outcome variable, View p-values of continuous X variables
results.kruskal.df <- results.kruskal.df[results.kruskal.df$variable != "Diabates_outcome",]
View(results.kruskal.df)


# Return only variables where p-value > 0.05
results.kruskal.df[results.kruskal.df$p_value <= 0.05,]
## All numeric variables have p-value = 0





# ---------------------------- 2.3 Chi-Squared Test ----------------------------
# For categorical Y and categorical X
# Create dataframe to store the p-values of all categorical X variables
results.chisq.df <- data.frame(variable = character(), 
                               p_value = numeric(), 
                               stringsAsFactors = FALSE)


# Create dataframe with only categorical X and Y variables
factor.columns <- sapply(diabetes, is.factor)
factor.data <- diabetes[, factor.columns]


# Run chi-sq test for variables
for (col in colnames(factor.data)) {
  
  # Extract the column data
  chisq.table <- table(factor.data[col])
  
  # Perform chisq test
  chisq.result <- chisq.test(chisq.table)
  
  # Store the results
  results.chisq.df <- rbind(results.chisq.df, 
                            data.frame(variable = col, 
                                       p_value = round(chisq.result$p.value, 3)
                            ))
}


# View p-values of continuous X variables
View(results.chisq.df)


# Return only variables where p-value > 0.05
results.chisq.df[results.chisq.df$p_value <= 0.05,]
## For diabetes dataset, all the p-values are 0





# ------------- 2.4 Logistic Regression (w/ backward elimination) --------------
model1 <- step(glm(Diabetes ~ ., data = diabetes, family = binomial), 
               direction = 'backward')
summary(model1)

## Certain values are quite important (e.g. Age25-29)
## The 15 variables with these important values are:
## - Age
## - Sex
## - HighChol
## - BMI
## - HeartDiseaseorAttack
## - PhysActivity
## - Fruits
## - Veggies
## - HvyAlcoholConsumpYes
## - GenHlth
## - MentlHlth
## - PhysHlth
## - DiffWalk
## - Stroke
## - HighBP





# -------------------------- 2.5 Selecting Variables ---------------------------
# Legend for explanation:
# DK - variable chosen because of domain knowledge
# KT - variable chosen because p-value < threshold in kruskal test
# CT - variable chosen because p-value < threshold in chi-sq test
# LR - variable chosen because logistic regression
diabetes2.df <- diabetes %>% select("Diabetes", # Target variable
                                 "Age",                     # DK, LR
                                 "Sex",                     # DK, LR
                                 "HighChol",                # DK, LR
                                 "CholCheck",               # DK
                                 "BMI",                     # DK, LR
                                 "Smoker",                  # DK
                                 "HeartDiseaseorAttack",    # DK, LR
                                 "PhysActivity",            # DK, LR
                                 "Fruits",                  # DK, LR
                                 "Veggies",                 # DK, LR
                                 "HvyAlcoholConsump",       # DK, LR
                                 "GenHlth",                 # DK, LR
                                 "PhysHlth",                # DK, LR
                                 "Stroke",                  # DK, LR
                                 "HighBP"                   # DK, LR
                                 )
View(diabetes2.df)
summary(diabetes2.df)    # 16 variables have been selected, inc. target variable
sum(is.na(diabetes2.df)) # No missing values





# ==================== Machine Learning Starts From Here :) ====================
# ==================== Part 3: Implement & Evaluate Models =====================

# This is the overview for Part 3:
# 3.1 - Train-Test Split & Set Threshold
# 3.2 - Model 1: Logistic Regression w Backward Elimination + Evaluation
# 3.3 - Model 2: Classification And Regression Tree (CART) + Evaluation
# 3.4 - Model 3: Random Forest + Evaluation

# All models will be evaluated using the 5 criteria:
# 1. Accuracy
# 2. Area under curve (AUC)
# 3. Precision ( true positives / [true positives + false positives] )
# 4. Recall    ( true positives / [true positives + false negatives] )
# 5. F1 Score  ( [2 * precision * recall] / [precision + recall] )

# NOTE: 
# Results that are displayed (e.g. model accuracy / important variables) were
# from testing session conducted on 26-03-24, 17000HRS.





# ==============================================================================
# ----------------- Part 3.1: Train-Test Split & Set Threshold -----------------
set.seed(2024)


# 70-30 train-test split will be applied across models
traintest_split <- sample.split(Y = diabetes2.df$Diabetes, 
                                SplitRatio = 0.7)
trainset <- subset(x = diabetes2.df, subset = traintest_split == TRUE)
testset <- subset(x = diabetes2.df, subset = traintest_split == FALSE)


# Set the threshold
threshold <- 0.5


# Create empty data frame to store test set results for models
testset.model.compare <- data.frame('Model' = c("Logistic Regression with BE", 
                                                "CART", "Random Forest"),
                                    'Accuracy' = rep(0, 3), 'AUC' = rep(0, 3),
                                    'Precision' = rep(0, 3), 'Recall' = rep(0, 3), 
                                    'F1 Score' = rep(0, 3))





# ----- Part 3.2: Logistic Regression w Backward Elimination + Evaluation ------
# Training model on trainset
log.model <- step(glm(Diabetes ~ ., family = binomial, data = trainset), 
                  direction = 'backward')


# Important variables
summary(log.model)

## Final model had values in 13 of the 15 variables:
## Age / Sex / HighChol / CholCheck / BMI / HeartDiseaseorAttack / PhysActivity /
## Veggies / HvyAlcoholConsump / GenHlth / PhysHlth / Stoke / HighBP

## Fruits / Smoker variables were not included


# Accuracy of model on trainset
trainset.prob <- predict(log.model, type = 'response')
trainset.predict <- ifelse(trainset.prob > threshold, "Yes", "No") 
## Comparing predicted probability of Diabetes with threshold

cm.trainset <- table(Actual = trainset$Diabetes , 
                     Predicted = trainset.predict, deparse.level = 2)
cm.trainset
mean(trainset.predict == trainset$Diabetes) ## Overall accuracy = 0.7493533


# Evaluating model on test set - accuracy 
testset.prob <- predict(log.model, type = 'response', newdata = testset)
testset.predict <- ifelse(testset.prob > threshold, "Yes", "No")
## Comparing predicted probability of Diabetes with threshold

cm.testset <- table(Actual = testset$Diabetes , Predicted = testset.predict, deparse.level = 2)
cm.testset
mean(testset.predict == testset$Diabetes) ## Overall accuracy = 0.7482554
testset.model.compare[1,2] <- mean(testset.predict == testset$Diabetes)


# For the prediction of developing diabetes (positive - have diabetes, negative - no diabetes)
# Assign True Positive (TP) / True Negative (TN) / False Positive (FP) / False Negative (FN)
TP <- cm.testset[2,2]
FP <- cm.testset[1,2]
FN <- cm.testset[2,1]
TN <- cm.testset[1,1]


# Evaluating model on test set - precision and recall
precision = (TP) / (TP + FP)
recall = (TP) / (TP + FN)
testset.model.compare[1,4] <- precision
testset.model.compare[1,5] <- recall
testset.model.compare[1,6] <- (2 * precision * recall) / (precision + recall)


# Evaluating model on test set - AUC
pred = prediction(testset.prob, testset$Diabetes)
auc = performance(pred, measure = "auc")
testset.model.compare[1,3] <- auc@y.values


# Evaluating model on test set - plot AUC
testset.predict.roc <- ifelse(testset.prob > threshold, 1, 0)
roc_score = roc(testset$Diabetes, testset.predict.roc)
plot(roc_score, main = "AUC for Logistic Regression w BE")


# Q-Q Plot
par(mfrow= c(2,2))
plot(log.model)
par(mfrow= c(1,1))

## top left = model w outlier
## top right = plots assumption 2, that error has normal distribution (w/o outlier)
## bottom left = assumption 3, if variance is constant (w outlier)
## bottom right = influential outlier (regression w/o outlier)

## From top left graph, assumption 1 does not apply to logistic regression model, as
## there are outliers

## From bottom left graph, variance or residuals is not constant





# ------------------------ Part 3.3: CART + Evaluation -------------------------
# Training model on train set
cart.model <- rpart(Diabetes ~  ., data = trainset, method = 'class',
                    control = rpart.control(minsplit = 2, cp = 0)) 

## to grow max tree, minsplit = 2 and cp = 0


# Plot maximal tree
# Might take some time to fully run
rpart.plot(cart.model, nn=T, main = "Maximal Tree For Trainset")


# Display pruning sequence and 10-fold CV errors as a chart
plotcp(cart.model, main ="Subtrees in diabetes") 


# Display pruning sequence and 10-fold CV errors as a table
printcp(cart.model)


# Source: Prof. Neumann Chew's Textbook, Chapter 8 - CART
# Create function to find the optimal CP
optimal.cp <- function(cartmodel) {
  
  # Compute min CVerror + 1SE in maximal tree cart.model
  CVerror.cap <- cartmodel$cptable[which.min(cartmodel$cptable[,"xerror"]), "xerror"] + 
    cartmodel$cptable[which.min(cartmodel$cptable[,"xerror"]), "xstd"]
  
  # Find the optimal CP region where CV error is just below CVerror.cap in maximal tree cart model
  i <- 1; j<- 4
  while (cartmodel$cptable[i,j] > CVerror.cap) {
    i <- i + 1
  }
  
  # Get geometric mean of the two identified CP values in the optimal region if optimal tree has at least one split.
  cp.optimal = ifelse(i > 1, sqrt(cartmodel$cptable[i,1] * cartmodel$cptable[i-1,1]), 1) #calculate geometric mean here
  
  return(cp.optimal)
}


# Find the optimal CP
cp.opt <- optimal.cp(cart.model)


# Prune the max tree
prune.cart.model <- prune(cart.model, cp = cp.opt)
print(prune.cart.model)
rpart.plot(prune.cart.model, nn=T, main = "Optimal Tree for Trainset")


# Accuracy of model on train set
cart.yhat <- predict(prune.cart.model, type= 'class')
cm.trainset <- table(Actual = trainset$Diabetes , 
                     Predicted = cart.yhat, deparse.level = 2)
cm.trainset
mean(cart.yhat == trainset$Diabetes) ## Overall Accuracy = 0.7447256


# Evaluating model on test set - accuracy 
prune.cart.model.yhat <- predict(prune.cart.model, type= 'class', 
                                 newdata = testset)
cm.testset <- table(Actual = testset$Diabetes, 
                    Predicted = prune.cart.model.yhat, deparse.level = 2)
cm.testset
mean(prune.cart.model.yhat == testset$Diabetes) ## Overall accuracy = 0.7384006
testset.model.compare[2,2] <- mean(prune.cart.model.yhat == testset$Diabetes)


# For the prediction of developing diabetes (positive - have diabetes, negative - no diabetes)
# Assign True Positive (TP) / True Negative (TN) / False Positive (FP) / False Negative (FN)
TP <- cm.testset[2,2]
FP <- cm.testset[1,2]
FN <- cm.testset[2,1]
TN <- cm.testset[1,1]


# Evaluating model on testset - precision and recall
precision = (TP) / (TP + FP)
recall = (TP) / (TP + FN)
testset.model.compare[2,4] <- precision
testset.model.compare[2,5] <- recall
testset.model.compare[2,6] <- (2 * precision * recall) / (precision + recall)

# Evaluating model on test set - AUC 
testset.prob <- as.data.frame(predict(prune.cart.model, type= 'prob', newdata = testset))
pred <- prediction(testset.prob$Yes, testset$Diabetes)
auc <- performance(pred, measure = "auc")
testset.model.compare[2,3] <- auc@y.values


# Evaluating model on testset - plot AUC
testset.predict.roc <- ifelse(testset.prob$Yes > threshold, 1, 0)
roc_score = roc(testset$Diabetes, testset.predict.roc)
plot(roc_score, main = "AUC for CART")





# ========================== Part 3.4.1: Random Forest ===========================
# Training model on trainset
rf.model <- randomForest(Diabetes ~ . , data = trainset, 
                         na.action = na.omit, 
                         importance = TRUE,
                         keep.inbag = TRUE)


# Accuracy of model on trainset
rf.model

## OOB estimate of  error rate: 25.15%
## Confusion matrix:
##        No   Yes class.error
## No  17346  7396   0.2989249
## Yes  5048 19694   0.2040255

print((17346+19694) / (17346+19694+7396+5048)) ## Overall accuracy = 0.7485248


# Evaluating OOB Error Rate
plot(rf.model,
     main = "OOB Error Rates Of Random Forest On Diabetes Dataset
     Up Till 500 Trees")
## OOB error rate stabilised before 500 trees


# Evaluating model on testset - accuracy 
rf.model.yhat <- predict(rf.model, newdata = testset)
cm.testset <- table(Actual = testset$Diabetes , Predicted = rf.model.yhat, deparse.level = 2)
cm.testset
mean(rf.model.yhat == testset$Diabetes) # Overall accuracy = 0.7454734
testset.model.compare[3,2] <- mean(rf.model.yhat == testset$Diabetes)


# For the prediction of developing diabetes (positive - have diabetes, negative - no diabetes)
# Assign True Positive (TP) / True Negative (TN) / False Positive (FP) / False Negative (FN)
TP <- cm.testset[2,2]
FP <- cm.testset[1,2]
FN <- cm.testset[2,1]
TN <- cm.testset[1,1]


# Evaluating model on testset - precision and recall
precision = (TP) / (TP + FP)
recall = (TP) / (TP + FN)
testset.model.compare[3,4] <- precision
testset.model.compare[3,5] <- recall
testset.model.compare[3,6] <- (2 * precision * recall) / (precision + recall)

# Evaluating model on test set - AUC 
testset.prob <- as.data.frame(predict(rf.model, type = 'prob', newdata = testset))
pred = prediction(testset.prob$Yes, testset$Diabetes)
auc = performance(pred, measure = "auc")
testset.model.compare[3,3] <- auc@y.values

# Evaluating model on testset - plot AUC
testset.predict.roc <- ifelse(testset.prob$Yes > threshold, 1, 0)
roc_score = roc(testset$Diabetes, testset.predict.roc)
plot(roc_score, main = "AUC for Random Forest")

# Evaluation of models
testset.model.compare

##              Model             Accuracy    AUC     Precision Recall  F1.Score
## 1 Logistic Regression with BE 0.7482554 0.8227512 0.7384727 0.7687665 0.7533152
## 2                        CART 0.7384006 0.7924656 0.7195207 0.7814032 0.7491863
## 3               Random Forest 0.7454734 0.8153717 0.7254070 0.7899849 0.7563200

## In healthcare, a false negative is more detrimental than a false positive.
## As recall performance of Random Forest is slightly better than CART and
## Logistic Regression w BE, Random Forest is preferred in this case.


## Based on the evaluation of other metrics for THIS DATASET, and 
## considering the real life conditions, Random Forest would be the preferred
## model. It is because in real life applicatons, there would be even more 
## features present in the dataset and the dataset would become more complex
## with more variables.





# ============== Part 3.4.2: Random Forest (Using Entire Dataset) ==============
# Apply RF on entire diabetes dataset
rf.model <- randomForest(Diabetes ~ . , data = diabetes, 
                         na.action = na.omit, 
                         importance = TRUE,
                         keep.inbag = TRUE)


# Accuracy of model
rf.model

## OOB estimate of  error rate: 25.49%
## Confusion matrix:
##        no   yes class.error
## No  24733 10613   0.3002603
## Yes  7403 27943   0.2094438

## Compared to previous RF error rate (25.15%), current RF error rate is
## slightly more (25.49%)

print((24733+27943) / (24733+27943+10613+7403)) ## Overall accuracy = 0.745148
print(27943 / (27943 + 7403)) ## Recall = 0.7905562

testset.model.compare[3,2] ## Overall accuracy = 0.7454734
testset.model.compare[3,5] ## Recall = 0.7899849

## Recall is higher in new RF model compared to original, but overall accuracy
## of the original model is slightly higher.
## In this case, no particular preference of the RF model.


# Evaluating OOB Error Rate
plot(rf.model,
     main = "OOB Error Rates Of Random Forest On Diabetes (Whole) Dataset
     Up Till 500 Trees")
## OOB error rate stabilised before 500 trees


# Evaluating Other Properties Of Random Forest
rf.model$oob.times # Check how many times each sample has been OOB

inbag <- rf.model$inbag # Check each case, whether it is inbag or OOB in which tree
View(inbag)

rf.model$votes


# Important variables
var.impt <- importance(rf.model)
varImpPlot(rf.model, type = 1)

## Comparing variable importance of different models:

## Logistic Regression w BE used 13 variables:
## Age / Sex / HighChol / CholCheck / BMI / HeartDiseaseorAttack / PhysActivity /
## Veggies / HvyAlcoholConsump / GenHlth / PhysHlth / Stoke / HighBP

## CART used 7 variables:
## GenHlth /  HighBP / BMI / Age / HighChol / HvyAlcoholConsump / 
## HeartDiseaseorAttack

## Top 6 variables w most importance in Random Forest:
## GenHlth / BMI / HighBP / Age / HighChol / HeartDiseaseorAttack

## The 3 models applied all identified common variables as important factors 
## these factors have to do with the lifestyle or parameters that might affect
## the probability of getting Diabetes more.



# ===================== Thank You For Running This Script! =====================
# ======================= Have A Wonderful Day Ahead :) ========================
# ==================================== End =====================================