# ==============================================================================
# BC2407 Analytics 2
# Purpose:      Predicting Dyslipidemia With Machine Learning Models and 
#               Identifying Key Factors
# Disease:      Chronic Kidney Disease
# Authors:      Amirul Haziq Bin Muhammad Effendy (U2210956K), 
#               Andrhea Angelina Therese Gaerlan San Gabriel (U2210291B),
#               Charan Kumar Velu (U2110392E),
#               Muhammad Wisnu Darmawan (U2220618B),
#               Yeung WenBin (U2110374B)
# DOC:          13-03-2024
# Topics:       Data Wrangling, Logistic Regression, CART, Random Forest
# Data Source:  kidney_disease_train.csv
# Link:         https://www.kaggle.com/datasets/colearninglounge/chronic-kidney-disease
# Packages:     tidyverse, ggplot2, haven, labelled, caTools, caret, ROCR, pROC,
#               rpart, rpart.plot, randomForest

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
setwd("/Users/ywb/Desktop/Univeristy/AY 23:24/BC2407-Analytics 2/Project Work/Submission")


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
library(rpart)
library(rpart.plot)
library(randomForest)





# ================== Part 1: Data Preparation & Cleaning =======================
# Read .csv file into R
ckd.df <- read.csv('kidney_disease_train.csv')
View(ckd.df)
summary(ckd.df)


# Check the datatype of each variable
str(ckd.df)


# Remove id variable from dataset
ckd.df <- ckd.df[, -which(names(ckd.df) == "id")]


# Convert certain variables with 'chr' datatype to numerical
ckd.df$wc <- as.numeric(ckd.df$wc)
ckd.df$rc <- as.numeric(ckd.df$rc)
colSums(is.na(ckd.df))
## NAs introduced because of converting "" values


# After viewing the data frame, there are 2 kinds of missing values that exist
# 1. Cells that are NA
# 2. Cells that have "" value

# To clean data...
# 1. Assign cells that have "" value to NA
# 2. Handle NA values in each variable


# Assigning cells that have "" value to NA
ckd.df$rbc <- na_if(ckd.df$rbc, "")
ckd.df$pc <- na_if(ckd.df$pc, "")
ckd.df$pcc <- na_if(ckd.df$pcc, "")
ckd.df$ba <- na_if(ckd.df$ba, "")
ckd.df$htn <- na_if(ckd.df$htn, "")
ckd.df$dm <- na_if(ckd.df$dm, "")
ckd.df$cad <- na_if(ckd.df$cad, "")
ckd.df$htn <- na_if(ckd.df$htn, "")
ckd.df$dm <- na_if(ckd.df$dm, "")
ckd.df$cad <- na_if(ckd.df$cad, "")
ckd.df$appet <- na_if(ckd.df$appet, "")
ckd.df$pe <- na_if(ckd.df$pe, "")
ckd.df$ane <- na_if(ckd.df$ane, "")
colSums(is.na(ckd.df)) ## Identify columns that require data cleaning


# Create function that replaces NA values in variable with mode
insert.mode <- function(data, variable) {
  
  # Calculate frequency of each value in categorical variable
  freq.table <- table(data[[variable]])
  
  # Find mode of variable
  mode.value <- names(freq.table)[which.max(freq.table)]
  
  # Replace missing values in variable with mode
  data[[variable]][is.na(data[[variable]])] <- mode.value
}


# Handle NA values
# For numeric variables, assign median
# For categorical variables, assign mode
ckd.df$age[is.na(ckd.df$age)] <- median(ckd.df$age, na.rm = TRUE)
ckd.df$bp[is.na(ckd.df$bp)] <- median(ckd.df$bp, na.rm = TRUE)
ckd.df$sg[is.na(ckd.df$sg)] <- insert.mode(ckd.df, "sg")
ckd.df$al[is.na(ckd.df$al)] <- insert.mode(ckd.df, "al")
ckd.df$su[is.na(ckd.df$su)] <- insert.mode(ckd.df, "su")
ckd.df$rbc[is.na(ckd.df$rbc)] <- insert.mode(ckd.df, "rbc")
ckd.df$pc[is.na(ckd.df$pc)] <- insert.mode(ckd.df, "pc")
ckd.df$pcc[is.na(ckd.df$pcc)] <- insert.mode(ckd.df, "pcc")
ckd.df$ba[is.na(ckd.df$ba)] <- insert.mode(ckd.df, "ba")
ckd.df$bgr[is.na(ckd.df$bgr)] <- mean(ckd.df$bgr, na.rm = TRUE)
ckd.df$bu[is.na(ckd.df$bu)] <- mean(ckd.df$bu, na.rm = TRUE)
ckd.df$sc[is.na(ckd.df$sc)] <- mean(ckd.df$sc, na.rm = TRUE)
ckd.df$sod[is.na(ckd.df$sod)] <- mean(ckd.df$sod, na.rm = TRUE)
ckd.df$pot[is.na(ckd.df$pot)] <- mean(ckd.df$pot, na.rm = TRUE)
ckd.df$hemo[is.na(ckd.df$hemo)] <- mean(ckd.df$hemo, na.rm = TRUE)
ckd.df$pcv[is.na(ckd.df$pcv)] <- mean(ckd.df$pcv, na.rm = TRUE)
ckd.df$wc[is.na(ckd.df$wc)] <- mean(ckd.df$wc, na.rm = TRUE)
ckd.df$rc[is.na(ckd.df$rc)] <- mean(ckd.df$rc, na.rm = TRUE)
ckd.df$htn[is.na(ckd.df$htn)] <- insert.mode(ckd.df, "htn")
ckd.df$dm[is.na(ckd.df$dm)] <- insert.mode(ckd.df, "dm")
ckd.df$cad[is.na(ckd.df$cad)] <- insert.mode(ckd.df, "cad")

colSums(is.na(ckd.df)) ## No more NAs


# Convert specific numeric variables into categorical
# sg - specific gravity
# al - albumin
# su - sugar
# Define breaks and labels
breaks1 <- c(1.005,1.010,1.015,1.020,1.025)
labels1 <- c("1.005", "1.010", "1.015", "1.020", "1.025")
breaks2 <- c(0,1,2,3,4,5)
labels2 <- c("0", "1", "2", "3", "4", "5")
breaks3 <- c(0,1,2,3,4,5)
labels3 <- c("0", "1", "2", "3", "4", "5")


# Create categorical variable using factor function with labels
ckd.df$sg <- factor(ckd.df$sg, levels = breaks1, labels = labels1)
ckd.df$al <- factor(ckd.df$al, levels = breaks2, labels = labels2)
ckd.df$su <- factor(ckd.df$su, levels = breaks3, labels = labels3)


# Convert certain variables with 'chr' datatype to categorical
ckd.df$rbc <- factor(ckd.df$rbc)
ckd.df$pc <- factor(ckd.df$pc)
ckd.df$pcc <- factor(ckd.df$pcc)
ckd.df$ba <- factor(ckd.df$ba)
ckd.df$htn <- factor(ckd.df$htn)
ckd.df$dm <- factor(ckd.df$dm)
ckd.df$cad <- factor(ckd.df$cad)
ckd.df$appet <- factor(ckd.df$appet)
ckd.df$pe <- factor(ckd.df$pe)
ckd.df$ane <- factor(ckd.df$ane)
ckd.df$classification <- factor(ckd.df$classification)


# Target variable is: classification





# ============== Part 2: Data Exploration & Selecting Variables ================
# Creation of visualisations will be done to analyse the impact that certain
# variables have on dyslipidemia outcome.

# Not all variables will be used in the dataset to train and test models.
# The team will leverage various methods to evaluate the significance of each
# variable.

# Method 1: Using domain knowledge
#           e.g. blood pressure (bp) is a known leading cause of chronic kidney
#           disease (ckd), and should thus be included in the dataset.

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
# Distribution of Ages in individuals with Chronic kidney disease
ckd.df$age_group <- cut(ckd.df$age, breaks = seq(0, 100, by = 10), labels = seq(0, 90, by = 10))

# Plot
ggplot(data = ckd.df, aes(fill = classification, y = ..count.., x = age_group)) + 
  geom_bar(position = "stack", stat = "count") +   
  stat_count(aes(y = after_stat(count), label = after_stat(count)), size = 3, 
             geom = "text", position = position_stack(vjust = 0.5)) + 
  labs(x = "Age", y = "Number Of Chronic Kidney Disease", 
       title = "Chronic Kidney Disease based on Age") +
  scale_x_discrete(labels = function(x) as.integer(gsub("\\D", "", x))) + 
  scale_fill_manual(values = c(colour2, colour1)) +
  theme(plot.title = element_text(hjust = 0.5))

# Chart 2
# Showing the distribution of the individuals based on their sodium (intake) level. To examine
# whether higher sodium level leads to higher chance of chronic kidney disease
ggplot(data = subset(ckd.df, sod >= 130 & sod <= 150), aes(x = sod, color = classification, fill = classification)) +
  geom_density(alpha = 0.5) +
  labs(x = "Sodium (mEq/L)", y = "Density", 
       title = "Density Plot of Chronic Kidney Disease based on Sodium Levels") +
  scale_color_manual(values = c(colour2, colour1)) +
  scale_fill_manual(values = c(colour2, colour1)) +
  theme(plot.title = element_text(hjust = 0.5))

# Chart 3
# Examining the potassium levels on the individuals. To examine whether abnormal 
# potassium level can lead to chronic kidney disease.
ggplot(data = subset(ckd.df, pot >= 2.5 & pot <= 8), aes(x = pot, color = classification, fill = classification)) +
  geom_density(alpha = 0.5) +
  labs(x = "Potassium (mEq/L)", y = "Density", 
       title = "Density Plot of Chronic Kidney Disease based on Potassium Levels") +
  scale_color_manual(values = c(colour2, colour1)) +
  scale_fill_manual(values = c(colour2, colour1)) +
  theme(plot.title = element_text(hjust = 0.5))





# ------------------------------ 2.2 Kruskal Test ------------------------------
# For categorical Y and continuous X
# Create dataframe to store the p-values of all continuous X variables
results.kruskal.df <- data.frame(variable = character(), 
                                 p_value = numeric(), 
                                 stringsAsFactors = FALSE)


# Create dataframe with only continuous X variables, and categorical Y
numeric.columns <- sapply(ckd.df, is.numeric)
numeric.data <- ckd.df[, numeric.columns]
numeric.data["classification"] <- ckd.df$classification


# Run kruskal test for variables
for (col in colnames(numeric.data)) {
  
  # Extract the column data
  column.data <- numeric.data[[col]]
  
  # Perform Kruskal-Wallis test
  kruskal_result <- kruskal.test(classification ~ column.data, 
                                 data = numeric.data)
  
  # Store the results
  results.kruskal.df <- rbind(results.kruskal.df, 
                              data.frame(variable = col, 
                                         p_value = round(kruskal_result$p.value, 3)
                              ))
}


# Remove classification variable, View p-values of continuous X variables
results.kruskal.df <- results.kruskal.df[results.kruskal.df$variable != "classification",]
View(results.kruskal.df)


# Return only variables where p-value < 0.05
results.kruskal.df[results.kruskal.df$p_value < 0.05,]

## 9 variables have p-value < 0.05
## age / bp / sc / sod / pot / hemo / pcv / wc / rc





# ---------------------------- 2.3 Chi-Squared Test ----------------------------
# For categorical Y and categorical X
# Create data frame to store the p-values of all categorical X variables
results.chisq.df <- data.frame(variable = character(), 
                               p_value = numeric(), 
                               stringsAsFactors = FALSE)


# Create dataframe with only categorical X and Y variables
factor.columns <- sapply(ckd.df, is.factor)
factor.data <- ckd.df[, factor.columns]


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
results.chisq.df <- results.chisq.df[results.chisq.df$variable != "classification",]
View(results.chisq.df)


# Return only variables where p-value < 0.05
results.chisq.df[results.chisq.df$p_value < 0.05,]

## all variables have p-value < 0.05 




# ------------- 2.4 Logistic Regression (w/ backward elimination) --------------
ckdmodel <- step(glm(classification ~ ., data = ckd.df, family = binomial), 
                 direction = 'backward')
summary(ckdmodel)

## Final 6 variables used:
## age / sg / pcv / htn / appet / pe





# -------------------------- 2.5 Selecting Variables ---------------------------
# Legend for explanation:
# DK - variable chosen because of domain knowledge
# KT - variable chosen because p-value < threshold in kruskal test
# CT - variable chosen because p-value < threshold in chi-sq test
# LR - variable chosen because logistic regression
ckd2.df <- ckd.df %>% select("classification", # Target Variable
                             "age",            #DK, KT
                             "bp",             #DK, KT
                             "sg",             #DK, CT, LR
                             "pcv",            #KT, LR
                             "htn",            #DK, CT, LR
                             "appet",          #CT, LR
                             "pe",             #CT, LR
                             "dm",             #DK, CT 
                             "cad",            #DK, CT, LR
                             "wc",             #KT
                             "rc",             #KT
                             "ane"             #CT
                             )
View(ckd2.df)
summary(ckd2.df) # 13 variables have been selected, inc. target variable





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
# from testing session conducted on 27-03-24, 0100HRS.





# ==============================================================================
# ----------------- Part 3.1: Train-Test Split & Set Threshold -----------------
set.seed(2024)


# 70-30 train-test split will be applied across models
traintest_split <- sample.split(Y = ckd2.df$classification, 
                                SplitRatio = 0.7)
trainset <- subset(x = ckd2.df, subset = traintest_split == TRUE)
testset <- subset(x = ckd2.df, subset = traintest_split == FALSE)


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
log.model <- step(glm(classification ~ ., 
                      family = binomial, 
                      data = trainset), direction = 'backward')


# Important variables
summary(log.model)

## Final model used 5 variables:
## age / sg / pcv / htn / appet


# Accuracy of model on train set
trainset.prob <- predict(log.model, type = 'response')
trainset.predict <- ifelse(trainset.prob > threshold, "notckd", "ckd") 
## Comparing predicted probability of dyslipidemia_outcome with threshold

cm.trainset <- table(Actual = trainset$classification , 
                     Predicted = trainset.predict, deparse.level = 2)
cm.trainset
mean(trainset.predict == trainset$classification) ## Overall accuracy = 1.0


# Evaluating model on test set - accuracy 
testset.prob <- predict(log.model, type = 'response', newdata = testset)
testset.predict <- ifelse(testset.prob > threshold, "notckd", "ckd")
## Comparing predicted probability of dyslipidemia_outcome with threshold

cm.testset <- table(Actual = testset$classification , Predicted = testset.predict, deparse.level = 2)
cm.testset
mean(testset.predict == testset$classification) ## Overall accuracy = 0.9166667
testset.model.compare[1,2] <- mean(testset.predict == testset$classification)


# For the prediction of developing ckd (positive - have ckd, negative - no ckd)
# Assign True Positive (TP) / True Negative (TN) / False Positive (FP) / False Negative (FN)
TP <- cm.testset[1,1]
FN <- cm.testset[1,2]
FP <- cm.testset[2,1]
TN <- cm.testset[2,2]


# Evaluating model on test set - precision and recall
precision = (TP) / (TP + FP)
recall = (TP) / (TP + FN)
testset.model.compare[1,4] <- precision
testset.model.compare[1,5] <- recall
testset.model.compare[1,6] <- (2 * precision * recall) / (precision + recall)


# Evaluating model on test set - AUC
pred = prediction(testset.prob, testset$classification)
auc = performance(pred, measure = "auc")
testset.model.compare[1,3] <- auc@y.values


# Evaluating model on test set - plot AUC
testset.predict.roc <- ifelse(testset.prob > threshold, 1, 0)
roc_score = roc(testset$classification, testset.predict.roc)
plot(roc_score, main = "AUC for Logistic Regression w BE")


# Q-Q Plot
par(mfrow= c(2,2))
plot(log.model)
par(mfrow= c(1,1))

## top left = model w outlier
## top right = plots assumption 2, that error has normal distribution (w/o outlier)
## bottom left = assumption 3, if variance is constant (w outlier)
## bottom right = influential outlier (regression w/o outlier)

## From top right graph, assumption 2 does not apply to logistic regression model, as
## The residuals do not follow normal distribution





# ------------------------ Part 3.3: CART + Evaluation -------------------------
# Training model on train set
cart.model <- rpart(classification ~ . , data = trainset, method = 'class',
                    control = rpart.control(minsplit = 2, cp = 0)) 

## to grow max tree, minsplit = 2 and cp = 0


# Plot maximal tree
rpart.plot(cart.model, nn=T, main = "Maximal Tree For Trainset")


# Display pruning sequence and 10-fold CV errors as a chart
plotcp(cart.model, main ="Subtrees in chronic kidney disease") 


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
  cp.optimal = ifelse(i > 1, sqrt(cartmodel$cptable[i,1] * cartmodel$cptable[i-1,1]), 1) # calculate geometric mean here
  
  return(cp.optimal)
}


# Find the optimal CP
cp.opt <- optimal.cp(cart.model)


# Prune the max tree
prune.cart.model <- prune(cart.model, cp = cp.opt)
print(prune.cart.model)
rpart.plot(prune.cart.model, nn=T, main = "Optimal Tree for Trainset")

## 3 variables used: pcv / sg / htn


# Accuracy of model on train set
cart.yhat <- predict(prune.cart.model, type= 'class')
cm.trainset <- table(Actual = trainset$classification , 
                     Predicted = cart.yhat, deparse.level = 2)
cm.trainset
mean(cart.yhat == trainset$classification) ## Overall Accuracy - 0.9795918


# Evaluating model on test set - accuracy 
prune.cart.model.yhat <- predict(prune.cart.model, type= 'class', newdata = testset)
cm.testset <- table(Actual = testset$classification , 
                    Predicted = prune.cart.model.yhat, deparse.level = 2)
cm.testset
mean(prune.cart.model.yhat == testset$classification) ## Overall accuracy = 0.9761905
testset.model.compare[2,2] <- mean(prune.cart.model.yhat == testset$classification)


# For the prediction of developing ckd (positive - have ckd, negative - no ckd)
# Assign True Positive (TP) / True Negative (TN) / False Positive (FP) / False Negative (FN)
TP <- cm.testset[1,1]
FN <- cm.testset[1,2]
FP <- cm.testset[2,1]
TN <- cm.testset[2,2]


# Evaluating model on testset - precision and recall
precision = (TP) / (TP + FP)
recall = (TP) / (TP + FN)
testset.model.compare[2,4] <- precision
testset.model.compare[2,5] <- recall
testset.model.compare[2,6] <- (2 * precision * recall) / (precision + recall)


# Evaluating model on test set - AUC 
testset.prob <- as.data.frame(predict(prune.cart.model, type= 'prob', newdata = testset))
pred = prediction(testset.prob$notckd, testset$classification)
auc = performance(pred, measure = "auc")
testset.model.compare[2,3] <- auc@y.values


# Evaluating model on testset - plot AUC
testset.predict.roc <- ifelse(testset.prob$notckd > threshold, 1, 0)
roc_score = roc(testset$classification, testset.predict.roc)
plot(roc_score, main = "AUC for CART")





# ------------------- Part 3.4: Random Forest + Evaluation ---------------------
# Training model on train set
rf.model <- randomForest(classification ~ . , data = trainset, 
                         na.action = na.omit, 
                         importance = TRUE,
                         keep.inbag = TRUE)


# Accuracy of model on train set
rf.model

## OOB estimate of  error rate: 2.04%
## Confusion matrix:
##        ckd notckd class.error
## ckd    121      1 0.008196721
## notckd   3     71 0.040540541

print((121+71) / (121+1+3+71)) ## Overall accuracy = 0.9795918


# Evaluating OOB Error Rate
plot(rf.model,
     main = "OOB Error Rates Of Random Forest On Dyslipidemia Dataset
     Up Till 500 Trees")
## OOB error rate stabilised before 500 trees


# Evaluating model on test set - accuracy 
rf.model.yhat <- predict(rf.model, newdata = testset)
cm.testset <- table(Actual = testset$classification , 
                    Predicted = rf.model.yhat, deparse.level = 2)
cm.testset
mean(rf.model.yhat == testset$classification) ## Overall accuracy = 0.9761905
testset.model.compare[3,2] <- mean(rf.model.yhat == testset$classification)


# For the prediction of developing ckd (positive - have ckd, negative - no ckd)
# Assign True Positive (TP) / True Negative (TN) / False Positive (FP) / False Negative (FN)
TP <- cm.testset[1,1]
FN <- cm.testset[1,2]
FP <- cm.testset[2,1]
TN <- cm.testset[2,2]


# Evaluating model on testset - precision and recall
precision = (TP) / (TP + FP)
recall = (TP) / (TP + FN)
testset.model.compare[3,4] <- precision
testset.model.compare[3,5] <- recall
testset.model.compare[3,6] <- (2 * precision * recall) / (precision + recall)


# Evaluating model on test set - AUC 
testset.prob <- as.data.frame(predict(rf.model, type = 'prob', newdata = testset))
pred = prediction(testset.prob$notckd, testset$classification)
auc = performance(pred, measure = "auc")
testset.model.compare[3,3] <- auc@y.values


# Evaluating model on testset - plot AUC
testset.predict.roc <- ifelse(testset.prob$notckd > threshold, 1, 0)
roc_score = roc(testset$classification, testset.predict.roc)
plot(roc_score, main = "AUC for Random Forest")


# Evaluation of models
testset.model.compare

##                         Model  Accuracy       AUC Precision    Recall  F1.Score
## 1 Logistic Regression with BE 0.9166667 0.9393029 0.9787234 0.8846154 0.9292929
## 2                        CART 0.9761905 0.9765625 0.9807692 0.9807692 0.9807692
## 3               Random Forest 0.9880952 1.0000000 1.0000000 0.9807692 0.9902913

## In healthcare, a false negative is more detrimental than a false positive.
## Therefore, the Random Forest or CART is preferred to Logistic Regression w BE.

## While value of recall is the same between Random Forest and CART, the accuracy
## of the Random Forest is slightly higher than that of CART. Therefore for this
## dataset, the Random Forest is preferred.





# -------------- Part 3.4.2: Random Forest (Using Entire Dataset) --------------
# Apply RF on entire ckd2.df dataset
rf.model <- randomForest(classification ~ . , data = ckd2.df, 
                         na.action = na.omit, 
                         importance = TRUE,
                         keep.inbag = TRUE)


# Accuracy of model
rf.model

##         OOB estimate of  error rate: 1.43%
## Confusion matrix:
##        ckd notckd class.error
## ckd    173      1 0.005747126
## notckd   3    103 0.028301887

## Compared to the previous RF's error rate (2.04%), the new RF has a lower
## error rate (1.43%)

print((173+103) / (173+1+3+103)) ## Overall accuracy = 0.9857143
print(173 / (173 + 1)) ## Recall = 0.9942529

testset.model.compare[3,2] ## Overall accuracy = 0.9880952
testset.model.compare[3,5] ## Recall = 0.9807692

## While accuracy of new RF is slightly lower than original RF, the recall is
## higher. The new RF model would be preferred over the original model.


# Evaluating OOB Error Rate
plot(rf.model,
     main = "OOB Error Rates Of Random Forest On Dyslipidemia Dataset
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

## Logistic Regression w BE used 5 variables:
## age / sg / pcv / htn / appet

## CART used 3 variables:
## pcv / sg / htn

## Top 5 variables w most importance in Random Forest:
## sg / pcv / htn / dm / rc

## While there is some variance, 
## The 3 models identified common variables as important factors (sg / pcv / htn)
## sg - specific gravity - density of substance w.r.t. density of water
## pcv - packed cell volume - a measurement of the proportion of blood that is made up of cells
## htn - hypertension
## lifestyle factors like physical activity and sleep are known to influence
## hypertension and packed cell volume.





# ===================== Thank You For Running This Script! =====================
# ======================= Have A Wonderful Day Ahead :) ========================
# ==================================== End =====================================